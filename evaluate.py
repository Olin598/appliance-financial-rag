#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
作业3（方向A）· 逐题评测

读 evaluation/questions.json（运行前冻结），对每题：
1. 用混合检索（向量+BM25，RRF）取 top-6 证据，调模型生成带引用的答案；
2. 额外跑一次「只在向量召回里排第几 / 只在 BM25 召回里排第几」的对照，
   用于说明两类召回各自的短板；
3. 把结果原样写进 evaluation/runs/Qxx.json —— 答错也照实保留，不覆盖。

用法：
    python evaluate.py                 # 全部题目
    python evaluate.py --only Q01 Q08
    python evaluate.py --answer-only    # 只跑生成，不做召回对照
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

from dotenv import load_dotenv

load_dotenv()

import rag as ragmod  # noqa: E402

OUT = 'evaluation/runs'


def strip_text(h: dict, keep: int = 400) -> dict:
    o = {k: v for k, v in h.items() if k != 'text'}
    o['text_head'] = (h.get('text') or '')[:keep]
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--questions', default='evaluation/questions.json')
    ap.add_argument('--only', nargs='*')
    ap.add_argument('--topk', type=int, default=6)
    ap.add_argument('--mode', default='hybrid')
    ap.add_argument('--answer-only', action='store_true')
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    qs = json.load(open(args.questions, encoding='utf-8'))['questions']
    if args.only:
        qs = [q for q in qs if q['id'] in args.only]

    r = ragmod.Rag()
    print(f'语料 {len(r.chunks)} 块，编码器 {ragmod.encode.__module__}')

    for q in qs:
        print(f"\n===== {q['id']}  {q['type']} =====")
        print(f"问：{q['q']}")
        rec = {'question': q, 'mode': args.mode, 'topk': args.topk,
               'run_at': time.strftime('%Y-%m-%dT%H:%M:%S%z')}

        t0 = time.time()
        res = r.answer(q['q'], mode=args.mode, topk=args.topk)
        rec['retrieval_seconds'] = round(time.time() - t0 - (res.get('elapsed', 0) - 0), 2)
        rec['answer'] = res['answer']
        rec['citations'] = res['citations']
        rec['usage'] = res.get('usage', {})
        rec['elapsed'] = res.get('elapsed')
        rec['company_scope'] = res.get('company_scope')
        rec['query_rewritten'] = res.get('query_rewritten')
        rec['panorama'] = res.get('panorama')
        rec['retrieved'] = [strip_text(h) for h in res['retrieved']]
        print(f"答：{(res['answer'] or '')[:220]}")
        print(f"    引用 {len(res['citations'])} 条，耗时 {res.get('elapsed')}s"
              f"（检索查询：{(res.get('query_rewritten') or '')[:40]}）")

        if not args.answer_only:
            comp = {}
            rq = res.get('query_rewritten') or q['q']
            for m in ('vector', 'bm25'):
                hits = r.retrieve(rq, topk=args.topk, mode=m,
                                  company=res.get('company_scope'))
                comp[m] = [{'rank': h['rank'], 'chunk_id': h['chunk_id'], 'name': h['name'],
                            'section': h['section'], 'pdf_page': h['pdf_page'],
                            'kind': h['kind'], 'score': h.get('score')} for h in hits]
            rec['retrieval_comparison'] = comp
            # 混合检索里每块在两个单路召回中的排名
            by_id = {c['chunk_id']: c for c in comp['vector']}
            by_id2 = {c['chunk_id']: c for c in comp['bm25']}
            for h in rec['retrieved']:
                h['rank_in_vector_topk'] = by_id.get(h['chunk_id'], {}).get('rank')
                h['rank_in_bm25_topk'] = by_id2.get(h['chunk_id'], {}).get('rank')

        json.dump(rec, open(os.path.join(OUT, f"{q['id']}.json"), 'w', encoding='utf-8'),
                  ensure_ascii=False, indent=2)

    print(f'\n结果写入 {OUT}/')


if __name__ == '__main__':
    sys.exit(main())
