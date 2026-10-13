#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""批量定位 10 道题的标准答案（人工核对用）"""
import json
import os
import re

EX = 'data/extracted'


def lines(code):
    doc = json.load(open(os.path.join(EX, f'{code}.json'), encoding='utf-8'))
    out = []
    for pg in doc['pages']:
        for ln in pg['lines']:
            out.append((pg['page'], 'text', ln['text']))
        for ti, tb in enumerate(pg['tables'], 1):
            for row in tb['rows']:
                out.append((pg['page'], f'tb{ti}', ' | '.join(row)))
    return out


def show(code, name, pat, limit=6, only=None):
    print(f'--- {name}({code}) /{pat}/ ---')
    rx = re.compile(pat)
    n = 0
    for p, kind, t in lines(code):
        if only and only not in kind:
            continue
        if rx.search(t):
            print(f'   [p{p} {kind}] {t[:240]}')
            n += 1
            if n >= limit:
                break
    if n == 0:
        print('   （无匹配）')


CASES = [
    ('000333', '美的集团', r'归属于上市公司股东的净利润'),
    ('000651', '格力电器', r'归属于上市公司股东的净利润'),
    ('600690', '海尔智家', r'归属于上市公司股东的净利润'),
    ('000921', '海信家电', r'归属于上市公司股东的净利润'),
    ('002668', 'TCL智家', r'归属于上市公司股东的净利润'),
    ('002032', '苏泊尔', r'炊具|电器'),
    ('002242', '九阳股份', r'销售费用'),
    ('002508', '老板电器', r'营业收入.*变动|变动原因|营业收入同比增长|营业收入同比下降'),
    ('000521', '长虹美菱', r'研发投入'),
    ('002543', '万和电气', r'境外|国外|出口'),
    ('300894', '火星人', r'集成灶'),
    ('603355', '莱克电气', r'境外|出口|国外'),
]

for code, name, pat in CASES:
    show(code, name, pat)
    print()
