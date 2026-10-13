#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import os
import re

EX = 'data/extracted'
meta = json.load(open('data/sources.json', encoding='utf-8'))
TARGETS = ['000921', '000521', '002508', '002543', '300894', '002242']

for code in TARGETS:
    name = next(r['name'] for r in meta['reports'] if r['code'] == code)
    doc = json.load(open(os.path.join(EX, f'{code}.json'), encoding='utf-8'))
    print(f'\n=== {name} ({code}) ===')
    for pg in doc['pages']:
        if pg['page'] > 16:
            break
        for ti, tb in enumerate(pg['tables'], 1):
            for row in tb['rows']:
                txt = ' | '.join(row)
                if '净利润' in txt:
                    print(f'   [p{pg["page"]} 表{ti}] {txt[:230]}')
        for ln in pg['lines']:
            if '净利润' in ln['text'] and re.search(r'\d', ln['text']):
                print(f'   [p{pg["page"]} 文] {ln["text"][:230]}')
