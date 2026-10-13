#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
作业3（方向A）· 从巨潮资讯下载家电行业 12 家上市公司 2025 年年度报告全文

用法：
    python download_reports.py [--out data/pdf]

数据来源：巨潮资讯网（http://www.cninfo.com.cn/），交易所指定信息披露网站。
本脚本只下载「2025 年年度报告」正式版（排除摘要、英文版、更正版），
并把每份报告的元信息（公司、代码、标题、披露日期、官方 URL、SHA-256）写入
data/sources.json，便于复现与溯源。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import ssl
import sys
import time
import urllib.parse
import urllib.request

BASE = 'http://www.cninfo.com.cn'
UA = ('Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 '
      '(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36')

# 家用电器行业 12 家公司（覆盖白电、黑电、厨电、小家电）
COMPANIES = [
    ('000333', '美的集团'),
    ('000651', '格力电器'),
    ('600690', '海尔智家'),
    ('000921', '海信家电'),
    ('002668', 'TCL智家'),
    ('000521', '长虹美菱'),
    ('002508', '老板电器'),
    ('002543', '万和电气'),
    ('300894', '火星人'),
    ('002032', '苏泊尔'),
    ('002242', '九阳股份'),
    ('603355', '莱克电气'),
]

EXCLUDE = ('摘要', '英文', '取消', '更正', '补充', 'H股', '债券')

_ctx = ssl.create_default_context()
_ctx.check_hostname = False
_ctx.verify_mode = ssl.CERT_NONE


def _post(path: str, data: dict, timeout: int = 30):
    req = urllib.request.Request(
        BASE + path, data=urllib.parse.urlencode(data).encode(),
        headers={'User-Agent': UA,
                 'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8',
                 'Referer': BASE + '/new/commonUrl?url=disclosure/list/notice',
                 'X-Requested-With': 'XMLHttpRequest'})
    return json.loads(urllib.request.urlopen(req, timeout=timeout, context=_ctx).read())


def get_org_id(code: str):
    """查证券代码对应的巨潮 orgId（该接口要求 stock 参数为 code,orgId）"""
    for x in _post('/new/information/topSearch/query', {'keyWord': code, 'maxNum': 10}):
        if x.get('code') == code:
            return x.get('orgId')
    return None


def find_annual_report(code: str, org_id: str, year: int = 2025):
    """在指定年份区间内找该公司的年度报告正式版"""
    col = 'sse' if code.startswith('6') else 'szse'
    d = _post('/new/hisAnnouncement/query', {
        'pageNum': 1, 'pageSize': 30, 'column': col, 'tabName': 'fulltext',
        'stock': f'{code},{org_id}', 'category': 'category_ndbg_szsh',
        'seDate': f'{year + 1}-01-01~{year + 1}-12-31', 'isHLtitle': 'true'})
    for a in (d.get('announcements') or []):
        title = (a.get('announcementTitle') or '').replace('<em>', '').replace('</em>', '')
        if any(x in title for x in EXCLUDE):
            continue
        if f'{year}' not in title:
            continue
        return {
            'title': title,
            'disclosed': time.strftime('%Y-%m-%d',
                                       time.localtime((a.get('announcementTime') or 0) / 1000)),
            'url': 'http://static.cninfo.com.cn/' + (a.get('adjunctUrl') or ''),
        }
    return None


def download(url: str, dest: str) -> str:
    req = urllib.request.Request(url, headers={'User-Agent': UA,
                                              'Referer': BASE + '/'})
    h = hashlib.sha256()
    with urllib.request.urlopen(req, timeout=120, context=_ctx) as r, open(dest, 'wb') as f:
        while True:
            chunk = r.read(1 << 16)
            if not chunk:
                break
            h.update(chunk)
            f.write(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='data/pdf')
    ap.add_argument('--year', type=int, default=2025)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    sources = []
    for code, name in COMPANIES:
        oid = get_org_id(code)
        if not oid:
            print(f'[跳过] {code} 未取到 orgId')
            continue
        rep = find_annual_report(code, oid, args.year)
        if not rep:
            print(f'[跳过] {code} {name} 未找到 {args.year} 年年度报告')
            continue
        fn = os.path.join(args.out, f'{code}_{name}_{args.year}年报.pdf')
        if os.path.exists(fn) and os.path.getsize(fn) > 100_000:
            sha = hashlib.sha256(open(fn, 'rb').read()).hexdigest()
            print(f'[已存在] {name} {os.path.getsize(fn) / 1e6:.1f}MB')
        else:
            sha = download(rep['url'], fn)
            print(f'[下载完成] {name} {os.path.getsize(fn) / 1e6:.1f}MB  {rep["title"]}')
        sources.append({
            'code': code, 'name': name, 'title': rep['title'],
            'disclosed': rep['disclosed'], 'url': rep['url'],
            'file': fn, 'sha256': sha,
            'bytes': os.path.getsize(fn),
        })
        time.sleep(0.8)

    with open('data/sources.json', 'w', encoding='utf-8') as f:
        json.dump({'year': args.year, 'source': '巨潮资讯 http://www.cninfo.com.cn/',
                   'reports': sources}, f, ensure_ascii=False, indent=2)
    print(f'\n共 {len(sources)} 份，元信息写入 data/sources.json')


if __name__ == '__main__':
    sys.exit(main())
