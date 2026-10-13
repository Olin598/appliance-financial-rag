#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
作业3（方向A）· 第 2 步：切块

把 data/extracted/*.json 变成 data/chunks.jsonl。每块都带完整溯源元数据：
公司、证券代码、报告标题、章节、PDF 物理页码、官方链接（含 #page=）、块类型。

切块规则
- 正文：目标约 400 字，相邻块重叠 60 字，尽量在句末断开；
  长段落按句切分后再组合，避免把一整页压成一块。
- 表格：按「数据行」切，每块都重复该表表头（含多级表头行），
  这样单独一块也能看懂列的含义；保留空单元格，不猜测数值。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys

CHUNK_SIZE = 400
CHUNK_OVERLAP = 60

# 章节标题的常见形态
RE_CHAPTER = re.compile(r'^第[一二三四五六七八九十百]+[节章]\s*\S')
RE_PART = re.compile(r'^第[一二三四五六七八九十百]+部分\s*\S')
RE_NUM_HEAD = re.compile(r'^\d{1,2}(\.\d{1,2}){0,2}\s*[^\d\s].{0,30}$')
RE_CN_HEAD = re.compile(r'^[一二三四五六七八九十]+、\s*\S')
RE_SENT_END = re.compile(r'(?<=[。！？；])')


def is_heading(line: dict, body_size: float) -> bool:
    t = line['text']
    if len(t) > 45:
        return False
    # 纯数字/金额（如表里的「4.92」「19.70%」）不是章节标题
    if re.fullmatch(r'[-—()（）\d.,%万亿千百元\s/]+', t):
        return False
    if RE_CHAPTER.match(t) or RE_PART.match(t) or RE_CN_HEAD.match(t) or RE_NUM_HEAD.match(t):
        return True
    return line['size'] >= body_size + 1.5 and len(t) <= 30 and len(t) >= 3


def body_size_of(lines: list[dict]) -> float:
    if not lines:
        return 12.0
    sizes = [round(l['size'] * 2) / 2 for l in lines]
    freq: dict[float, int] = {}
    for s in sizes:
        freq[s] = freq.get(s, 0) + 1
    return max(freq.items(), key=lambda kv: (kv[1], kv[0]))[0]


def split_text(text: str) -> list[str]:
    """按 ~CHUNK_SIZE 字切分，带重叠，尽量在句末断开"""
    out = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + CHUNK_SIZE, n)
        if end < n:
            window = text[start:end + 40]
            parts = RE_SENT_END.split(window)
            acc = ''
            for p in parts:
                if len(acc) + len(p) > CHUNK_SIZE:
                    break
                acc += p
            if len(acc) >= CHUNK_SIZE * 0.5:
                end = start + len(acc)
        out.append(text[start:end].strip())
        if end >= n:
            break
        start = max(end - CHUNK_OVERLAP, start + 1)
    return [c for c in out if len(c) >= 30]


def col_labels(header_rows: list[list[str]], ncols: int) -> list[str]:
    """按列拼接多级表头，得到每列的列名（同一列里重复的上一级表头只留一次）。"""
    labels = []
    for j in range(ncols):
        parts = []
        for r in header_rows:
            if j < len(r) and r[j].strip() and r[j].strip() not in ('—', '-'):
                v = r[j].strip()
                if not parts or parts[-1] != v:
                    parts.append(v)
        labels.append(' '.join(parts))
    return labels


def render_table_row(labels: list[str], row: list[str]) -> str:
    """把一行单元格渲染成「列名: 数值」的语义化文本。

    这一步很关键：原始 PDF 表格抽出来是「空单元格 + 竖线」的形态，
    既不便于人看，向量模型也读不出结构（例如「主要会计数据」表里
    根本没有出现公司名和年份，导致检索时被别的段落压过去）。
    """
    n = max(len(labels), len(row))
    parts = []
    for j in range(n):
        lab = labels[j] if j < len(labels) else ''
        val = row[j].strip() if j < len(row) else ''
        if not val:
            continue
        if lab and lab != val:
            parts.append(f'{lab}: {val}')
        else:
            parts.append(val)
    return '；'.join(parts)


def table_rows_to_chunks(rows: list[list[str]]) -> list[list[str]]:
    """把表格按数据行分组，每组都带上表头（渲染成「列名: 数值」）。"""
    if not rows:
        return []
    # 表头：取前 1-2 行中，非数字占比高、且列数与数据行一致者
    head_rows = 1
    if len(rows) >= 3:
        r0, r1 = rows[0], rows[1]
        def numeric_ratio(r):
            cells = [c for c in r if c != '']
            if not cells:
                return 0.0
            n = sum(1 for c in cells if re.fullmatch(r'[-—()（）\d.,%万亿千百元/年月日\s]+', c))
            return n / len(cells)
        if numeric_ratio(r0) < 0.4 and numeric_ratio(r1) > 0.4:
            head_rows = 1
        elif numeric_ratio(r0) < 0.4 and numeric_ratio(r1) < 0.4:
            head_rows = 2
    header = rows[:head_rows]
    body = rows[head_rows:]
    if not body:
        body = rows
        header = []

    ncols = max((len(r) for r in rows), default=0)
    labels = col_labels(header, ncols)

    groups, cur, cur_len = [], [], 0
    for r in body:
        line = render_table_row(labels, r)
        if not line:
            continue
        if cur and cur_len + len(line) > CHUNK_SIZE:
            groups.append(cur)
            cur, cur_len = [], 0
        cur.append(line)
        cur_len += len(line) + 1
    if cur:
        groups.append(cur)

    head_line = '；'.join(x for x in labels if x.strip())
    out = []
    for g in groups:
        buf = []
        if head_line:
            buf.append('列名：' + head_line)
        buf.extend(g)
        out.append(buf)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--extracted', default='data/extracted')
    ap.add_argument('--out', default='data/chunks.jsonl')
    args = ap.parse_args()

    meta = json.load(open('data/sources.json', encoding='utf-8'))
    by_code = {r['code']: r for r in meta['reports']}

    total = 0
    stats = []
    with open(args.out, 'w', encoding='utf-8') as fo:
        for code in [r['code'] for r in meta['reports']]:
            path = os.path.join(args.extracted, f'{code}.json')
            if not os.path.exists(path):
                print(f'[跳过] {code} 未抽取')
                continue
            doc = json.load(open(path, encoding='utf-8'))
            rep = by_code[code]
            base_url = rep['url']
            section = '文首'
            n_text = n_table = 0

            for pg in doc['pages']:
                lines = pg['lines']
                bs = body_size_of(lines)
                # 逐行推进章节；同时按顺序累积正文
                buf: list[str] = []
                seq = 0

                def flush_text():
                    nonlocal seq, n_text, total
                    if not buf:
                        return
                    txt = '\n'.join(buf).strip()
                    buf.clear()
                    if len(txt) < 30:
                        return
                    for c in split_text(txt):
                        seq += 1
                        cid = f"{code}-p{pg['page']}-t{seq}"
                        head = (f"{doc['name']}（{code}）{doc['title']}"
                                f"｜{section}｜第 {pg['page']} 页")
                        fo.write(json.dumps({
                            'chunk_id': cid, 'code': code, 'name': doc['name'],
                            'title': doc['title'], 'section': section,
                            'pdf_page': pg['page'], 'kind': 'text',
                            'url': f"{base_url}#page={pg['page']}", 'text': c,
                            # index_text 只用于建索引（带公司/章节/页码，弥补表格块缺公司名的短板）
                            'index_text': f'{head}\n{c}',
                        }, ensure_ascii=False) + '\n')
                        total += 1
                        n_text += 1

                for ln in lines:
                    if is_heading(ln, bs):
                        flush_text()
                        section = ln['text'].strip()[:60]
                    else:
                        buf.append(ln['text'])
                flush_text()

                # 表格
                for ti, tb in enumerate(pg['tables'], 1):
                    for gi, grp in enumerate(table_rows_to_chunks(tb['rows']), 1):
                        n_table += 1
                        cid = f"{code}-p{pg['page']}-tb{ti}-{gi}"
                        body = '\n'.join(grp)
                        head = (f"{doc['name']}（{code}）{doc['title']}"
                                f"｜{section}｜第 {pg['page']} 页｜表格")
                        fo.write(json.dumps({
                            'chunk_id': cid, 'code': code, 'name': doc['name'],
                            'title': doc['title'], 'section': section,
                            'pdf_page': pg['page'], 'kind': 'table',
                            'url': f"{base_url}#page={pg['page']}",
                            'text': body,
                            'index_text': f'{head}\n{body}',
                        }, ensure_ascii=False) + '\n')
                        total += 1

            stats.append({'code': code, 'name': doc['name'], 'pages': doc['total_pages'],
                          'tables': doc['n_tables'], 'text_chunks': n_text,
                          'table_chunks': n_table})
            print(f"[完成] {doc['name']:<6} 页 {doc['total_pages']:>4} | "
                  f"正文块 {n_text:>5} | 表格块 {n_table:>5}")

    json.dump({'chunk_size': CHUNK_SIZE, 'overlap': CHUNK_OVERLAP,
               'total_chunks': total, 'per_report': stats},
              open('data/corpus_stats.json', 'w', encoding='utf-8'),
              ensure_ascii=False, indent=2)
    print(f'\n总块数 {total}，统计写入 data/corpus_stats.json')


if __name__ == '__main__':
    sys.exit(main())
