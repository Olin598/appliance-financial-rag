#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""抽取 12 家公司「主要会计数据」里的营业收入与归母净利润（用于全景题标准答案）"""
import json
import os
import re

EX = 'data/extracted'
meta = json.load(open('data/sources.json', encoding='utf-8'))

NUM = re.compile(r'\d{1,3}(?:,\d{3})+(?:\.\d+)?')


def rows_containing(code, keys, max_page=40):
    doc = json.load(open(os.path.join(EX, f'{code}.json'), encoding='utf-8'))
    out = []
    for pg in doc['pages']:
        if pg['page'] > max_page:
            continue
        for ti, tb in enumerate(pg['tables'], 1):
            for row in tb['rows']:
                txt = ' | '.join(row)
                if any(k in txt for k in keys):
                    if NUM.search(txt.replace(' ', '')):
                        out.append((pg['page'], ti, txt))
    return out


print('公司 | 营业收入行 | 归母净利润行')
print('=' * 100)
for rep in meta['reports']:
    code, name = rep['code'], rep['name']
    rev = rows_containing(code, ['营业收入'])
    prof = rows_containing(code, ['归属于上市公司股东的净利润', '归属于母公司所有者的净利润',
                                  '归属于母公司股东的净利润'])
    # 只保留含数字、且在「主要会计数据/财务摘要」区间的行
    def pick(rs):
        best = None
        for p, ti, t in rs:
            nums = [float(x.replace(',', '')) for x in NUM.findall(t)]
            if not nums:
                continue
            mx = max(nums)
            if mx >= 1e4 and '分季度' not in t:
                if best is None or mx > best[2]:
                    best = (p, t, mx)
        return best
    r = pick(rev)
    pr = pick(prof)
    print(f'\n### {name} ({code})')
    print(f'  营收: [p{r[0]}] {r[1][:200]}' if r else '  营收: 未找到')
    print(f'  净利: [p{pr[0]}] {pr[1][:200]}' if pr else '  净利: 未找到')
