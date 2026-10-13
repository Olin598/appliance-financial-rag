#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
作业3（方向A）· 把 evaluation/runs/Q*.json 汇总成逐题评测报告

评分方式（尽量自动化、可复核）
- 从冻结的参考答案里自动抽出「关键数值」（带千分位/小数点/百分号的数字），
  检查这些数值是否出现在模型答案中 → 命中率。
- 召回侧的另一个指标：含答案的那一块是否被检索进来（用关键数值在召回块文本里找）。
- 报告里同时保留模型原答与具体引用来源，便于人工复核。
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

RE_NUM = re.compile(r'(-?\d[\d,]*(?:\.\d+)?)\s*[（(]?\s*(%|亿元|万元|千元|元|亿|万)?')

# 单位 → 换算到「元」的倍数；百分比单独处理
UNIT_FACTOR = {'元': 1.0, '千元': 1e3, '万元': 1e4, '亿元': 1e8,
               '千': 1e3, '万': 1e4, '亿': 1e8}


SCALES = (1.0, 1e3, 1e4, 1e8)      # 元 / 千元 / 万元 / 亿元


def parse_numbers(text: str) -> tuple[set[float], set[float]]:
    """解析文本里的数字，返回 (金额候选值集合, 百分比集合)。

    金额做「多尺度」展开：同一个数字可能以 元/千元/万元/亿元 四种单位出现
    （「12,422,233（千元）」与「124.22 亿元」是同一个数），所以把一个数按四种
    单位都展开进集合，再按 1% 容差比对 —— 这样就不会因为答案换了单位而误判成答错。
    """
    money: set[float] = set()
    pct: set[float] = set()
    for m in RE_NUM.finditer(text or ''):
        raw = m.group(1).replace(',', '')
        unit = m.group(2) or ''
        try:
            v = float(raw)
        except ValueError:
            continue
        if unit == '%':
            pct.add(v)
            continue
        base = v * UNIT_FACTOR.get(unit, 1.0)
        for s in SCALES:
            money.add(base * s)
            if v:
                money.add(v * s)
    return money, pct


def key_numbers(text: str) -> list[str]:
    """从参考答案里挑出可用于核对的数值（要求带千分位 / 小数 / 单位，滤掉年份之类的裸整数）"""
    out = []
    for m in RE_NUM.finditer(text or ''):
        s = m.group(0).strip()
        digits = re.sub(r'\D', '', s.split('.')[0])
        if (',' in s or '.' in s or m.group(2)) and len(digits) >= 2:
            out.append(s)
    return list(dict.fromkeys(out))


def _close(a: float, b: float) -> bool:
    return abs(a - b) <= max(abs(b) * 0.01, 1e-9)


def hit_rate(ans: str, keys: list[str]) -> tuple[int, list[str], list[str]]:
    """参考答案里的每个关键数值，是否能在模型答案里找到（按数值+单位换算后比对，容差 1%）"""
    a_money, a_pct = parse_numbers(ans)
    got, miss = [], []
    for k in keys:
        k_money, k_pct = parse_numbers(k)
        ok = (any(_close(x, y) for x in a_pct for y in k_pct)
              or any(_close(x, y) for x in a_money for y in k_money))
        (got if ok else miss).append(k)
    return len(got), got, miss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--questions', default='evaluation/questions.json')
    ap.add_argument('--runs', default='evaluation/runs')
    ap.add_argument('--out', default='evaluation/逐题评测.md')
    ap.add_argument('--emb', default='auto')
    args = ap.parse_args()

    qs = json.load(open(args.questions, encoding='utf-8'))
    meta, questions = qs, qs['questions']
    runs = {}
    for q in questions:
        p = Path(args.runs) / f"{q['id']}.json"
        if p.exists():
            runs[q['id']] = json.loads(p.read_text(encoding='utf-8'))

    L = ['# 作业3（方向A）· 十道题逐题评测', '',
         f"题目冻结时间：{meta.get('frozen_at', '')}",
         '',
         f"语料：{meta.get('corpus', '')}",
         '',
         '> 说明：先冻结 10 道题与参考答案（答案逐条对照年报 PDF 原文人工核对、标注了页码），',
         '> 再跑评测，避免事后凑答案。下表「关键数值命中」= 参考答案里的关键数字有多少出现在模型答案中。',
         '']

    # 总览
    tot_key = tot_hit = 0
    rows = []
    for q in questions:
        r = runs.get(q['id'])
        keys = key_numbers(q['ref'])
        if keys and r:
            h, got, miss = hit_rate(r.get('answer', ''), keys)
            tot_key += len(keys)
            tot_hit += h
            rate = f'{h}/{len(keys)}'
        else:
            got, miss, rate = [], keys, '—'
        cite_n = len(r.get('citations', [])) if r else 0
        sec = r.get('retrieval_seconds') or r.get('elapsed') or '—' if r else '—'
        rows.append((q, r, rate, got, miss, cite_n, sec))

    if tot_key:
        L += [f'**十题合计关键数值命中：{tot_hit}/{tot_key} = {tot_hit / tot_key * 100:.1f}%**', '']

    L += ['## 总览', '',
          '| 题号 | 类型 | 关键数值命中 | 引用条数 | 耗时(s) |',
          '|---|---|---:|---:|---:|']
    for q, r, rate, got, miss, cite_n, sec in rows:
        L.append(f"| {q['id']} | {q['type']} | {rate} | {cite_n} | {sec} |")
    L += ['']

    by_id = {q['id']: (q, r, rate, got, miss) for q, r, rate, got, miss, _, _ in rows}
    for q in questions:
        q, r, rate, got, miss = by_id[q['id']]
        L += [f"---", '', f"## {q['id']}　{q['type']}", '',
              f"**问：** {q['q']}", '',
              f"**参考答案（人工核对，来源 {q.get('ref_source', '')}"
              + (f"，PDF 第 {q['ref_page']} 页" if q.get('ref_page') else '') + '）：**',
              '', f"> {q['ref']}", '']
        if not r:
            L += ['**答：** （该题尚未运行评测）', '']
            continue
        L += [f"**模型答（{r.get('mode')} 检索，公司范围："
              f"{r.get('company_scope') or '全体'}）：**", '', r.get('answer', ''), '']
        L += [f"关键数值命中：**{rate}**"
              + (f"　命中 {got}" if got else '')
              + (f"　未中 {miss}" if miss else ''), '']
        if r.get('citations'):
            L += ['**引用来源：**', '',
                  '| # | 公司 | 章节 | PDF 页 | 类型 | 官方链接 |', '|---|---|---|---|---|---|']
            for c in r['citations']:
                L.append(f"| [{c['n']}] | {c['name']} | {c['section'][:28]} | {c['pdf_page']} | "
                         f"{'表格' if c['kind'] == 'table' else '正文'} | [年报原文]({c['url']}) |")
            L += ['']
        # 召回的 top-6
        if r.get('retrieved'):
            L += ['<details><summary>召回证据 top-6（点击展开）</summary>', '',
                  '| 排名 | 公司 | 章节 | 页 | 类型 | 向量排名 | BM25排名 | 片段 |',
                  '|---|---|---|---|---|---:|---:|---|']
            for h in r['retrieved']:
                frag = (h.get('text') or '')[:60].replace('\n', ' ').replace('|', '｜')
                L.append(f"| {h.get('rank')} | {h.get('name')} | {str(h.get('section'))[:20]} | "
                         f"{h.get('pdf_page')} | {'表格' if h.get('kind') == 'table' else '正文'} | "
                         f"{h.get('rank_in_vector_topk') or '—'} | {h.get('rank_in_bm25_topk') or '—'} | "
                         f"{frag} |")
            L += ['', '</details>', '']
        if r.get('retrieval_comparison'):
            v = r['retrieval_comparison'].get('vector', [])
            b = r['retrieval_comparison'].get('bm25', [])
            L += ['**单路召回对照（top-6）：**', '',
                  '- 只走向量：' + '、'.join(
                      f"{x['name']}p{x['pdf_page']}" for x in v), '',
                  '- 只走 BM25：' + '、'.join(
                      f"{x['name']}p{x['pdf_page']}" for x in b), '']

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text('\n'.join(L), encoding='utf-8')
    print(f'写入 {args.out}（{len(questions)} 题，已跑 {len(runs)} 题）')
    if tot_key:
        print(f'关键数值命中 {tot_hit}/{tot_key} = {tot_hit / tot_key * 100:.1f}%')


if __name__ == '__main__':
    main()
