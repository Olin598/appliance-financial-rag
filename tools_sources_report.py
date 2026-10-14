#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成两个交付辅助文件：

1. SOURCES.md              —— 12 家公司 2025 年年报的**官方全文来源**（巨潮资讯网原始链接、
                              披露日期、页数、表格区域数、切块数、SHA-256 校验值）
2. data/chunks_sample.jsonl —— 知识块样本（每家公司抽 2 条正文 + 2 条表格），
                              用来证明「文字/表格切块」与「公司/章节/页码」元数据真实存在

用法：python tools_sources_report.py
"""
from __future__ import annotations

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SOURCES = os.path.join(HERE, 'data', 'sources.json')
STATS = os.path.join(HERE, 'data', 'corpus_stats.json')
CHUNKS = os.path.join(HERE, 'data', 'chunks.jsonl')
SAMPLE = os.path.join(HERE, 'data', 'chunks_sample.jsonl')
OUT = os.path.join(HERE, 'SOURCES.md')

PER_COMPANY_TEXT = 2
PER_COMPANY_TABLE = 2


def build_sources_md():
    src = json.load(open(SOURCES, encoding='utf-8'))
    stats = json.load(open(STATS, encoding='utf-8'))
    per = {r['code']: r for r in stats['per_report']}

    reports = src['reports']
    years = sorted({str(r.get('year') or src.get('year')) for r in reports})

    L = []
    A = L.append
    A('# 语料来源：官方年报全文')
    A('')
    A(f'本作业的语料是 **{len(reports)} 家家用电器行业上市公司** 的年报全文 PDF，'
      f'全部来自 **巨潮资讯网**（<http://www.cninfo.com.cn/>）——'
      f'中国证监会指定的上市公司信息披露网站，即交易所官方披露渠道。')
    A('')
    A('> 只下载「年度报告」正式版；摘要、英文版、更正版、H 股版、债券相关文件均排除。')
    A('> 每份 PDF 下载后计算 SHA-256，写入 `data/sources.json`，用于校验文件未被替换。')
    A('')
    A('## 12 家公司的官方全文链接')
    A('')
    A('| # | 代码 | 公司 | 报告标题 | 披露日期 | 页数 | 表格区域 | 知识块 | 官方全文链接 |')
    A('|---:|---|---|---|---|---:|---:|---:|---|')
    tot_pages = tot_tables = tot_chunks = 0
    for i, r in enumerate(reports, 1):
        st = per.get(r['code'], {})
        pages = st.get('pages', '—')
        tables = st.get('tables', '—')
        chunks = int(st.get('text_chunks', 0)) + int(st.get('table_chunks', 0))
        tot_pages += pages if isinstance(pages, int) else 0
        tot_tables += tables if isinstance(tables, int) else 0
        tot_chunks += chunks
        A(f"| {i} | `{r['code']}` | {r['name']} | {r['title']} | {r.get('disclosed', '—')} "
          f"| {pages} | {tables} | {chunks} | [PDF]({r['url']}) |")
    A(f'| **合计** | — | **12 家** | — | — | **{tot_pages}** | **{tot_tables}** | **{tot_chunks}** | — |')
    A('')
    A(f'→ 满足「至少 10 家官方全文来源」的要求（实为 12 家，覆盖白电、黑电、厨电、小家电）。')
    A('')
    A('## 文件完整性校验（SHA-256）')
    A('')
    A('| 代码 | 公司 | 字节数 | SHA-256 |')
    A('|---|---|---:|---|')
    for r in reports:
        A(f"| `{r['code']}` | {r['name']} | {r.get('bytes', 0):,} | `{r.get('sha256', '')}` |")
    A('')
    A('## 怎么复现')
    A('')
    A('```bash')
    A('# 1) 从巨潮资讯重新下载全部 12 份年报，并写回 data/sources.json')
    A('python download_reports.py')
    A('')
    A('# 2) 抽取正文与表格')
    A('python extract_reports.py')
    A('')
    A('# 3) 切块（正文按段落/章节，表格按「列名: 数值」语义化）')
    A('python build_chunks.py')
    A('')
    A('# 4) 建索引（BM25 + 本地 BGE ONNX 向量）')
    A('python build_index.py')
    A('')
    A('# 5) 起问答服务')
    A('python app.py          # 打开 http://127.0.0.1:8765')
    A('```')
    A('')
    A('> 原始 PDF（约 60 MB）与索引产物（约 67 MB）体积较大，未放进仓库；')
    A('> 按上面 4 条命令可完整重建，结果与仓库内 `evaluation/` 下记录的一致。')
    A('')

    with open(OUT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(L))
    print(f'✅ 已生成 {OUT}（{len(L)} 行，{len(reports)} 家公司）')


def build_chunks_sample():
    counters = {}
    kept = []
    with open(CHUNKS, encoding='utf-8') as f:
        for line in f:
            c = json.loads(line)
            code, kind = c['code'], c.get('kind', 'text')
            k = (code, kind)
            if counters.get(k, 0) >= (PER_COMPANY_TABLE if kind == 'table' else PER_COMPANY_TEXT):
                continue
            counters[k] = counters.get(k, 0) + 1
            kept.append({
                'chunk_id': c['chunk_id'],
                'code': code,
                'name': c['name'],
                'title': c['title'],
                'section': c['section'],
                'pdf_page': c['pdf_page'],
                'kind': kind,
                'url': c['url'],
                'text': c['text'],
                'index_text': c.get('index_text') or c['text'],
            })

    with open(SAMPLE, 'w', encoding='utf-8') as f:
        for c in kept:
            f.write(json.dumps(c, ensure_ascii=False) + '\n')
    print(f'✅ 已生成 {SAMPLE}（{len(kept)} 条样本，覆盖 {len({c["code"] for c in kept})} 家公司）')


if __name__ == '__main__':
    build_sources_md()
    build_chunks_sample()
