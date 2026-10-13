#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""在抽取结果里定位标准答案（人工核对用，不参与线上流程）"""
from __future__ import annotations
import json
import os
import re
import sys

EX = 'data/extracted'


def search(code: str, pattern: str, limit: int = 12, ctx: int = 0):
    doc = json.load(open(os.path.join(EX, f'{code}.json'), encoding='utf-8'))
    rx = re.compile(pattern)
    hits = 0
    for pg in doc['pages']:
        for ln in pg['lines']:
            if rx.search(ln['text']):
                print(f"  [p{pg['page']}] {ln['text'][:220]}")
                hits += 1
                if hits >= limit:
                    return
        for ti, tb in enumerate(pg['tables'], 1):
            for ri, row in enumerate(tb['rows']):
                txt = ' | '.join(row)
                if rx.search(txt):
                    print(f"  [p{pg['page']} 表{ti}] {txt[:260]}")
                    hits += 1
                    if hits >= limit:
                        return


if __name__ == '__main__':
    code, pat = sys.argv[1], sys.argv[2]
    limit = int(sys.argv[3]) if len(sys.argv) > 3 else 12
    print(f'=== {code}  /{pat}/ ===')
    search(code, pat, limit)
