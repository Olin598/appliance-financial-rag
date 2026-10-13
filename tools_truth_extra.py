#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import os
import re

EX = 'data/extracted'


def rows(code, pat, maxp=40, only_table=False, limit=25):
    doc = json.load(open(os.path.join(EX, f'{code}.json'), encoding='utf-8'))
    rx = re.compile(pat)
    n = 0
    for pg in doc['pages']:
        if pg['page'] > maxp:
            break
        if not only_table:
            for ln in pg['lines']:
                if rx.search(ln['text']):
                    print(f'  [p{pg["page"]} 文] {ln["text"][:250]}')
                    n += 1
                    if n >= limit:
                        return
        for ti, tb in enumerate(pg['tables'], 1):
            for row in tb['rows']:
                t = ' | '.join(row)
                if rx.search(t):
                    print(f'  [p{pg["page"]} 表{ti}] {t[:250]}')
                    n += 1
                    if n >= limit:
                        return


print('### 美的集团 分季度主要财务指标（p11 附近的表）')
rows('000333', r'分季度|第一季度|第二季度', maxp=14, limit=30)

print('\n### 老板电器 经营情况讨论：收入/利润下降原因')
rows('002508', r'同比(下降|减少|上升|增长)|变动原因|主要原因|受.{0,12}影响', maxp=30, limit=20)

print('\n### 长虹美菱 研发投入')
rows('000521', r'研发投入', maxp=30, limit=20)

print('\n### 九阳股份 销售费用率')
rows('002242', r'销售费用率', maxp=20, limit=10)

print('\n### TCL智家 营业收入（正文口径）')
rows('002668', r'实现营业收入|营业收入.{0,10}亿元', maxp=25, limit=8)
