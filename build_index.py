#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
作业3（方向A）· 第 3 步：建索引

读 data/chunks.jsonl，产出 data/index/：
- vectors.npy   BGE-small-zh-v1.5 的 512 维归一化向量（float32，n×512）
- bm25.pkl      jieba 分词后的 BM25 倒排索引（k1=1.5, b=0.75）
- chunk_ids.json 与向量行号一一对应的 chunk_id 列表

BM25 自己实现（不引入额外依赖），公式：
    score(q,d) = Σ_t IDF(t) * f(t,d)*(k1+1) / (f(t,d) + k1*(1-b+b*|d|/avgdl))
    IDF(t) = ln(1 + (N - df(t) + 0.5)/(df(t) + 0.5))
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import sys

import jieba
import numpy as np
from dotenv import load_dotenv

load_dotenv()

from embedding import encode  # noqa: E402

K1, B = 1.5, 0.75
# 财报里的公司全称加入自定义词典，保证不被切碎
EXTRA_WORDS = [
    '美的集团', '格力电器', '海尔智家', '海信家电', 'TCL智家', '长虹美菱',
    '老板电器', '万和电气', '火星人', '苏泊尔', '九阳股份', '莱克电气',
    '营业收入', '归母净利润', '扣非净利润', '经营活动产生的现金流量净额',
    '毛利率', '加权平均净资产收益率', '基本每股收益', '研发投入',
    '销售费用', '管理费用', '财务费用', '应收账款', '存货', '合同负债',
    '归属于上市公司股东的净利润', '机器人及自动化', '暖通空调', '消费电器',
]
for w in EXTRA_WORDS:
    jieba.add_word(w)


def tokenize(text: str) -> list[str]:
    return [t for t in jieba.lcut(text) if t.strip()]


def build_bm25(tokens: list[list[str]]):
    n = len(tokens)
    doc_len = [len(t) for t in tokens]
    avgdl = sum(doc_len) / max(n, 1)
    tf: list[dict[str, int]] = []
    df: dict[str, int] = {}
    for toks in tokens:
        d: dict[str, int] = {}
        for t in toks:
            d[t] = d.get(t, 0) + 1
        for t in d:
            df[t] = df.get(t, 0) + 1
        tf.append(d)
    idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}
    return {'n': n, 'doc_len': doc_len, 'avgdl': avgdl, 'tf': tf, 'idf': idf}


def bm25_search(index, query_tokens: list[str], topk: int):
    scores = np.zeros(index['n'], dtype=np.float32)
    avgdl = index['avgdl']
    for t in set(query_tokens):
        idf = index['idf'].get(t)
        if idf is None:
            continue
        for i, d in enumerate(index['tf']):
            f = d.get(t)
            if not f:
                continue
            scores[i] += idf * f * (K1 + 1) / (f + K1 * (1 - B + B * index['doc_len'][i] / avgdl))
    idx = np.argsort(-scores)[:topk]
    return [(int(i), float(scores[i])) for i in idx if scores[i] > 0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--chunks', default='data/chunks.jsonl')
    ap.add_argument('--out', default='data/index')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    chunks = [json.loads(l) for l in open(args.chunks, encoding='utf-8') if l.strip()]
    print(f'载入 {len(chunks)} 块')

    def itext(c: dict) -> str:
        """建索引用 index_text（带公司/章节/页码前缀）；老数据回退到 text。"""
        return c.get('index_text') or c['text']

    print('jieba 分词 ...')
    tokens = [tokenize(itext(c)) for c in chunks]

    print('建 BM25 ...')
    bm = build_bm25(tokens)
    pickle.dump({'index': bm, 'tokens': tokens}, open(os.path.join(args.out, 'bm25.pkl'), 'wb'))

    print('BGE 向量化（CPU，可能需要几分钟）...')
    vecs = []
    step = 32
    for i in range(0, len(chunks), step):
        vecs.append(encode([itext(c) for c in chunks[i:i + step]]))
        if (i // step) % 10 == 0:
            print(f'  {i + len(vecs[-1])}/{len(chunks)}')
    vectors = np.vstack(vecs)
    np.save(os.path.join(args.out, 'vectors.npy'), vectors)

    json.dump([c['chunk_id'] for c in chunks],
              open(os.path.join(args.out, 'chunk_ids.json'), 'w', encoding='utf-8'),
              ensure_ascii=False)
    print(f'完成：vectors {vectors.shape}，写入 {args.out}/')


if __name__ == '__main__':
    sys.exit(main())
