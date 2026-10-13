#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
作业3（方向A）· 第 1 步：从年报 PDF 抽取正文与表格

输出 data/extracted/<code>.json，结构：
{
  "code": "000333", "name": "美的集团", "title": "...", "url": "...",
  "total_pages": 210,
  "pages": [
    {"page": 12,
     "text": "本页纯文本（保留阅读顺序）",
     "lines": [{"text": "...", "size": 12.0, "bold": false}],
     "tables": [{"bbox": [...], "rows": [["项目","2025年","2024年"], ...]}]}
  ]
}

要点：
- 页码 page 一律为 PDF 物理页码（从 1 开始），与 citation 的 #page= 一致；
- 表格通过 PyMuPDF 的 find_tables 检测，保留原始二维单元格矩阵，空单元格保留为空串；
- 同时记录每行的字号 / 是否加粗，供后续章节归属使用。
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import fitz  # PyMuPDF


def extract_page(page: fitz.Page) -> dict:
    lines = []
    d = page.get_text('dict')
    for block in d.get('blocks', []):
        if block.get('type') != 0:  # 0 = 文本块
            continue
        for line in block.get('lines', []):
            txt = ''.join(s.get('text', '') for s in line.get('spans', [])).strip()
            if not txt:
                continue
            size = max((s.get('size', 0) for s in line.get('spans', [])), default=0)
            bold = any((s.get('flags', 0) & 16) for s in line.get('spans', []))
            lines.append({'text': txt, 'size': round(size, 1), 'bold': bool(bold)})

    tables = []
    try:
        for t in page.find_tables():
            rows = []
            for row in t.extract():
                rows.append(['' if c is None else str(c).replace('\n', ' ').strip()
                             for c in row])
            if rows:
                tables.append({'bbox': [round(v, 1) for v in t.bbox], 'rows': rows})
    except Exception:
        pass

    return {
        'page': page.number + 1,
        'text': page.get_text('text').strip(),
        'lines': lines,
        'tables': tables,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pdf-dir', default='data/pdf')
    ap.add_argument('--out', default='data/extracted')
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()

    meta = json.load(open('data/sources.json', encoding='utf-8'))
    os.makedirs(args.out, exist_ok=True)

    for rep in meta['reports']:
        code = rep['code']
        dest = os.path.join(args.out, f'{code}.json')
        if os.path.exists(dest) and not args.force:
            print(f'[跳过] {rep["name"]} 已抽取')
            continue
        doc = fitz.open(rep['file'])
        pages = [extract_page(p) for p in doc]
        doc.close()
        n_tables = sum(len(p['tables']) for p in pages)
        out = {
            'code': code, 'name': rep['name'], 'title': rep['title'],
            'url': rep['url'], 'disclosed': rep['disclosed'],
            'total_pages': len(pages), 'n_tables': n_tables,
            'pages': pages,
        }
        json.dump(out, open(dest, 'w', encoding='utf-8'), ensure_ascii=False)
        print(f'[完成] {rep["name"]:<6} {len(pages):>4} 页  表格区域 {n_tables:>3} 个')


if __name__ == '__main__':
    sys.exit(main())
