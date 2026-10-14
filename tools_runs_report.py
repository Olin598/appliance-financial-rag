#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 evaluation/runs/Q*.json 生成增强版逐题运行记录 —— evaluation/逐题运行记录.md

与 evaluation/逐题评测.md 的区别：本文件保留**可复核的原始证据**，每题给出
  1) 召回证据（候选块的 chunk_id / 公司 / 章节 / PDF页码 / 文字或表格 / 向量排名 / BM25排名 / 相似度 / 原文片段）
  2) 单路召回对照（纯向量 vs 纯 BM25）
  3) 答案判定（正确 / 部分正确 / 错误，附理由）
  4) 未中原因（逐一解释参考答案里没被答出的数字）

用法：python tools_runs_report.py
"""
from __future__ import annotations

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS_DIR = os.path.join(HERE, 'evaluation', 'runs')
OUT = os.path.join(HERE, 'evaluation', '逐题运行记录.md')

# ── 逐题人工判定与错误归因（对 runs/Q*.json 的人工复核结论）──────────────────
# hits / misses 取自 evaluation/逐题评测.md（同一次运行、同一口径，不重算，避免口径漂移）
REVIEW = {
    'Q01': dict(
        hits=['456,451,731 千元', '4,564.52 亿元', '12.11%'],
        misses=['4,071.50 亿元', '407,149,600 千元'],
        verdict='✅ 正确',
        why='题干两问（本期营业收入是多少 / 同比增长多少）均已答对，且数字与「主要会计数据」表一致。',
        miss_reason='两个未中项都是**上年同期对照数**（4,071.50 亿元 = 407,149,600 千元）。题干没有问上期数，模型只答了本期与增长率。属于「未主动补充题干外信息」，不是答错。'),
    'Q02': dict(
        hits=['29,003,103,411.66 元', '290.03 亿元'],
        misses=['321.85 亿元'],
        verdict='✅ 正确',
        why='题干两问（净利润是多少 / 同比变化如何）均已答对，同比方向与幅度（-9.89%）与年报一致。',
        miss_reason='未中项是**上年同期数**（321.85 亿元）。模型答了本期数值与同比降幅，未回填上年基数。同样属于题干外信息。'),
    'Q03': dict(
        hits=['124.22 亿元', '135.91 亿元', '118.70 亿元', '60.62 亿元',
              '12,422,233', '13,591,457', '11,869,693', '6,062,028 千元'],
        misses=['439.45 亿元'],
        verdict='✅ 正确',
        why='四个季度数值全部答对（且同时给出千元原值与亿元换算），跨列读取正确，无串列。',
        miss_reason='未中项是**全年合计**（439.45 亿元）。题干问的是「四个季度分别是多少」，合计并非被问项；且该数在 Q09 里模型已正确给出。'),
    'Q04': dict(
        hits=['16.95%'],
        misses=['139,157.73 万元'],
        verdict='✅ 正确',
        why='题干问的是比率型指标「销售费用率」，模型给出的 16.95% 与参考答案一致。',
        miss_reason='未中项是**销售费用绝对值**（139,157.73 万元），是计算费用率的分子，题干未要求给出。'),
    'Q05': dict(
        hits=['19,552,798,222.85 元', '195.53 亿元', '187.31 亿元'],
        misses=[],
        verdict='✅ 正确',
        why='口语化提问（「去年赚了多少钱」）经查询改写后命中「归属于上市公司股东的净利润」，本期与上期数字、同比方向（增加 4.39%）全部答对。',
        miss_reason='无。'),
    'Q06': dict(
        hits=['101.16 亿元', '12.56 亿元', '20.38%'],
        misses=[],
        verdict='✅ 正确',
        why='需要跨段归纳的「为什么下降」，模型既给对了两个下降幅度，也归纳出了报告里陈述的行业需求下行原因。',
        miss_reason='无。'),
    'Q07': dict(
        hits=['967,536,650.87 元'],
        misses=['20.73%'],
        verdict='✅ 正确',
        why='题干两问（研发投入金额 / 占营业收入比例）均已答对：9.68 亿元、占营业收入 3.18%。',
        miss_reason='未中项是**研发投入的同比增幅**（20.73%），题干未问，模型未主动补充。'),
    'Q08': dict(
        hits=['4,564.52 亿元', '770,536,064.78 元'],
        misses=[],
        verdict='✅ 正确',
        why='跨 12 家公司全景题，最高（美的集团）与最低（火星人）判断正确、数值正确。走的是「逐公司召回」策略，12 家全部覆盖。',
        miss_reason='无。'),
    'Q09': dict(
        hits=['439.45 亿元', '43,945,411 千元', '-296,097,562.49 元'],
        misses=['2,755.29%'],
        verdict='✅ 正确',
        why='跨 12 家公司全景题，净利润最高（美的集团 439.45 亿元）与最低（火星人 -2.96 亿元）判断与数值均正确，并正确识别出海尔等公司的量级差异。',
        miss_reason='未中项是火星人净利润的**同比下降幅度**（-2,755.29%），题干只问「最高和最低分别是谁、是多少」，降幅未被问及。'),
    'Q10': dict(
        hits=[],
        misses=[],
        verdict='✅ 正确（拒答）',
        why='语料只到 2025 年年度报告，无 2026 年半年报。模型明确回答「资料中未找到」，**未编造任何数字**，引用条数为 0。这是库外题的正确行为。',
        miss_reason='不适用（本题考的就是「不该答」）。'),
}


def frag(t, n=150):
    s = ' / '.join(x.strip() for x in str(t or '').splitlines() if x.strip())
    return (s[:n] + '…') if len(s) > n else s


def cell(x, dash='—'):
    return dash if x is None or x == '' else str(x)


def build():
    questions = json.load(open(os.path.join(HERE, 'evaluation', 'questions.json'), encoding='utf-8'))['questions']
    lines = []
    A = lines.append

    A('# 作业3（方向A）· 逐题运行记录（可复核版）')
    A('')
    A('> 本文件与 `逐题评测.md` 同源（同一次运行、同一批冻结题目），但保留**原始证据**：')
    A('> 每题都给出检索实际召回的候选块（含 `chunk_id`、公司、章节、PDF 页码、文字/表格类型、')
    A('> 向量排名与 BM25 排名、相似度、原文片段）、纯向量与纯 BM25 的单路对照，')
    A('> 以及人工复核后的**答案判定**与**未中原因**。')
    A('')
    A('题目冻结时间：2026-10-13T21:05:00+08:00　｜　原始运行记录：`evaluation/runs/Q01.json` … `Q10.json`')
    A('')
    A('十题合计关键数值命中：**26/33 = 78.8%**；其中 **7 个未中项经逐条复核全部为题干未问的次要对照数，无一处答案错误**。')
    A('')
    A('---')
    A('')

    for q in questions:
        qid = q['id']
        run = json.load(open(os.path.join(RUNS_DIR, f'{qid}.json'), encoding='utf-8'))
        rv = REVIEW[qid]
        n_hit, n_all = len(rv['hits']), len(rv['hits']) + len(rv['misses'])

        A(f"## {qid}　{q['type']}")
        A('')
        A(f"**问：** {q['q']}")
        A('')
        src = q.get('ref_source') or '—'
        page = f"第 {q['ref_page']} 页" if q.get('ref_page') else '跨全表'
        A(f"**参考答案（人工核对 · 来源：{src} · PDF {page}）：**")
        A('')
        A(f"> {q['ref']}")
        A('')
        A(f"**模型回答**（检索模式 `{run['mode']}`，公司范围：{cell(run.get('company_scope'), '全库')}"
          f"{'，全景逐公司召回' if run.get('panorama') else ''}，耗时 {run.get('elapsed')} s）：")
        A('')
        A('```')
        A(str(run.get('answer', '')).strip())
        A('```')
        A('')

        if run.get('query_rewritten') and run['query_rewritten'].strip('？?') != q['q'].strip('？?'):
            A(f"查询改写：`{q['q']}` → `{run['query_rewritten']}`")
            A('')

        A(f"### 答案判定：{rv['verdict']}")
        A('')
        A(f"- **判定理由**：{rv['why']}")
        if n_all:
            A(f"- **关键数值命中**：{n_hit}/{n_all}"
              f"　命中 `{rv['hits']}`"
              + (f"　未中 `{rv['misses']}`" if rv['misses'] else ''))
        else:
            A('- **关键数值命中**：不适用（本题考「应当拒答」）')
        A(f"- **未中原因**：{rv['miss_reason']}")
        A('')

        cites = run.get('citations') or []
        A('### 引用来源（模型在答案里标注的出处）')
        A('')
        if cites:
            A('| # | chunk_id | 公司 | 章节 | PDF 页 | 类型 | 官方链接 |')
            A('|---|---|---|---|---:|---|---|')
            for c in cites:
                A(f"| [{c.get('n')}] | `{c.get('chunk_id')}` | {cell(c.get('name'))} | {cell(c.get('section'))} "
                  f"| {cell(c.get('pdf_page'))} | {'表格' if c.get('kind') == 'table' else '正文'} "
                  f"| [年报原文]({c.get('url')}) |")
        else:
            A('（无引用 —— 库外题按设计拒绝给出处。）')
        A('')

        ret = run.get('retrieved') or []
        A(f'### 召回证据（实际检索到的候选块，共 {len(ret)} 条）')
        A('')
        if ret:
            A('| 排名 | chunk_id | 公司 | 章节 | 页 | 类型 | 向量排名 | BM25排名 | 余弦 | BM25 | 原文片段 |')
            A('|---:|---|---|---|---:|---|---:|---:|---:|---:|---|')
            for r in ret:
                cos_v, bm_v = r.get('cos'), r.get('bm25')
                cos_s = f'{cos_v:.4f}' if cos_v is not None else '—'
                bm_s = f'{bm_v:.3f}' if bm_v is not None else '—'
                kind = '表格' if r.get('kind') == 'table' else '正文'
                A(f"| {r.get('rank')} | `{r.get('chunk_id')}` | {cell(r.get('name'))} | {cell(r.get('section'))} "
                  f"| {cell(r.get('pdf_page'))} | {kind} "
                  f"| {cell(r.get('vec_rank'))} | {cell(r.get('bm25_rank'))} "
                  f"| {cos_s} | {bm_s} | {frag(r.get('text_head'))} |")
        else:
            A('（无召回结果。）')
        A('')

        rc = run.get('retrieval_comparison') or {}
        if rc:
            A('### 单路召回对照（top-6）')
            A('')
            for label, key in (('只用向量', 'vector'), ('只用 BM25', 'bm25')):
                items = rc.get(key) or []
                if not items:
                    continue
                desc = '、'.join(
                    f"{cell(i.get('name'))}p{i.get('pdf_page')}"
                    + (f"（{i.get('chunk_id')}）" if i.get('rank') == 1 else '')
                    for i in items)
                A(f"- **{label}**：{desc}")
            A('')

        A('---')
        A('')

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f'✅ 已生成 {OUT}（{len(lines)} 行）')


if __name__ == '__main__':
    build()
