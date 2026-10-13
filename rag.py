#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
作业3（方向A）· 检索与问答核心

- hybrid  向量召回 top-60 + BM25 召回 top-60 → RRF 融合 → 取 top-k
- vector  只用向量
- bm25    只用关键词
可选按公司过滤（问题里点明公司时）。

生成使用 DeepSeek（OpenAI 兼容接口），强制要求：答案里的每个事实都要带 [n] 引用，
引用可回溯到具体块（公司 / 章节 / PDF 页码 / 官方链接）。
"""
from __future__ import annotations

import json
import os
import pickle
import re
import time

import numpy as np
from dotenv import load_dotenv

load_dotenv()

from build_index import bm25_search, tokenize  # noqa: E402
from embedding import encode  # noqa: E402

RRF_K = 60
POOL = 60
COMPANIES = ['美的集团', '格力电器', '海尔智家', '海信家电', 'TCL智家', '长虹美菱',
             '老板电器', '万和电气', '火星人', '苏泊尔', '九阳股份', '莱克电气']

# 提问里可能出现简称，统一映射到语料里的公司名
COMPANY_ALIAS = {
    '美的': '美的集团', '美的集团': '美的集团',
    '格力': '格力电器', '格力电器': '格力电器',
    '海尔': '海尔智家', '海尔智家': '海尔智家',
    '海信家电': '海信家电', '海信': '海信家电',
    'tcl智家': 'TCL智家', 'tcl': 'TCL智家',
    '长虹美菱': '长虹美菱', '美菱': '长虹美菱',
    '老板电器': '老板电器',
    '万和电气': '万和电气', '万和': '万和电气',
    '火星人': '火星人',
    '苏泊尔': '苏泊尔',
    '九阳股份': '九阳股份', '九阳': '九阳股份',
    '莱克电气': '莱克电气', '莱克': '莱克电气',
}


def detect_company(query: str) -> str | None:
    """问题里点名的公司。命中多个（跨公司对比题）则返回 None，不做过滤。"""
    q = query.lower()
    hits = {v for k, v in COMPANY_ALIAS.items() if k.lower() in q}
    if len(hits) == 1:
        return hits.pop()
    # 「A 和 B 哪家…」这类对比题明确命中多家 → 不过滤
    return None


# 跨公司「全景题」的信号词：问最高/最低/排名/对比，且没有点名某一家
PANORAMA_HINTS = ['最高', '最低', '哪家', '哪一家', '排名', '12 家', '12家', '这批',
                  '所有公司', '全部公司', '这几家', '分别是谁', '谁最高', '谁最低',
                  '最赚钱', '最赚钱的', '对比']


def is_panorama(query: str) -> bool:
    if detect_company(query):
        return False
    return any(h in query for h in PANORAMA_HINTS)


# 「原因类」提问的信号词，以及年报里讲原因的叙述性章节
WHY_HINTS = ['为什么', '为何', '原因', '缘故', '导致', '因为什么']
NARRATIVE_SECTIONS = ['行业情况', '主营业务分析', '经营情况', '管理层讨论', '讨论与分析',
                      '核心竞争力', '经营概述', '公司业务概要', '主要业务分析']


def is_why_question(query: str) -> bool:
    return any(h in query for h in WHY_HINTS)


REWRITE_PROMPT = (
    "你是财报检索助手。把下面这句口语化的提问改写成一句**精确的检索查询**，要求：\n"
    "1) 补齐财务口径的标准名称（例如「赚了多少钱」→「归属于上市公司股东的净利润」，"
    "「营收」→「营业收入」，「花了多少研发」→「研发投入金额」）；\n"
    "2) 如果是相对时间（去年/前年/今年），结合下面的年报年份说明换成具体年份"
    "（本批年报均为 2025 年年度报告，提问里的「去年」= 2025 年，「前年」= 2024 年）；\n"
    "3) 保留问题里出现的公司名和指标名；\n"
    "4) 只输出改写后的那一句查询，不要任何解释、不要引号。\n\n"
    "提问："
)


def rewrite_query(query: str) -> str:
    """口语化提问 → 检索友好的查询（查询改写/扩展），检索失败时原样返回。"""
    try:
        out, _ = chat(REWRITE_PROMPT + query, temperature=0.0, timeout=30)
    except Exception:
        return query
    out = (out or '').strip().strip('"“”').splitlines()
    out = out[0].strip() if out else ''
    if not out or out.startswith('（生成失败') or len(out) < 4 or len(out) > 120:
        return query
    return out


class Rag:
    def __init__(self, index_dir: str = 'data/index', chunks_path: str = 'data/chunks.jsonl'):
        self.chunks = [json.loads(l) for l in open(chunks_path, encoding='utf-8') if l.strip()]
        self.by_id = {c['chunk_id']: c for c in self.chunks}
        self.chunk_ids = json.load(open(os.path.join(index_dir, 'chunk_ids.json'), encoding='utf-8'))
        self.vectors = np.load(os.path.join(index_dir, 'vectors.npy'))
        blob = pickle.load(open(os.path.join(index_dir, 'bm25.pkl'), 'rb'))
        self.bm25 = blob['index']
        self.row_of = {cid: i for i, cid in enumerate(self.chunk_ids)}

    # ---------- 召回 ----------
    def _vector_recall(self, query: str, pool: int, allow: np.ndarray | None):
        q = encode(query, is_query=True)[0]
        sims = self.vectors @ q
        if allow is not None:
            sims = np.where(allow, sims, -1e9)
        idx = np.argsort(-sims)[:pool]
        return [(int(i), float(sims[i])) for i in idx]

    def _bm25_recall(self, query: str, pool: int, allow: np.ndarray | None):
        hits = bm25_search(self.bm25, tokenize(query), topk=pool * 4)
        out = []
        for i, s in hits:
            if allow is not None and not allow[i]:
                continue
            out.append((i, s))
            if len(out) >= pool:
                break
        return out

    def retrieve(self, query: str, topk: int = 6, mode: str = 'hybrid',
                 company: str | None = None, pool: int = POOL,
                 sections: list[str] | None = None):
        allow = None
        if company:
            # chunk_ids 顺序与 chunks.jsonl 行序一致
            allow = np.array([self.by_id[cid]['name'] == company for cid in self.chunk_ids])
        if sections:
            # 只在指定的章节范围内检索（用于「为什么下降」这类原因题：答案在
            # 「行业情况 / 主营业务分析」等叙述性章节，而不是财务报表附注里）
            mask = np.array([any(s in self.by_id[cid]['section'] for s in sections)
                             for cid in self.chunk_ids])
            if mask.sum() >= 30:          # 命中太少就不限定，避免检索到空
                allow = mask if allow is None else (allow & mask)

        if mode == 'vector':
            hits = self._vector_recall(query, topk, allow)
            ranked = [(self.chunk_ids[i], s, i) for i, s in hits]
            return [{'chunk_id': c, 'score': s, 'rank': r + 1, 'row': row,
                     **{k: self.by_id[c][k] for k in ('name', 'section', 'pdf_page', 'kind', 'url', 'text')}}
                    for r, (c, s, row) in enumerate(ranked)]

        if mode == 'bm25':
            hits = self._bm25_recall(query, topk, allow)
            ranked = [(self.chunk_ids[i], s, i) for i, s in hits]
            return [{'chunk_id': c, 'score': s, 'rank': r + 1, 'row': row,
                     **{k: self.by_id[c][k] for k in ('name', 'section', 'pdf_page', 'kind', 'url', 'text')}}
                    for r, (c, s, row) in enumerate(ranked)]

        # hybrid: RRF
        vh = self._vector_recall(query, pool, allow)
        bh = self._bm25_recall(query, pool, allow)
        rrf: dict[int, float] = {}
        detail: dict[int, dict] = {}
        for r, (i, s) in enumerate(vh, 1):
            rrf[i] = rrf.get(i, 0.0) + 1.0 / (RRF_K + r)
            detail.setdefault(i, {})['vec_rank'] = r
            detail.setdefault(i, {})['cos'] = round(s, 4)
        for r, (i, s) in enumerate(bh, 1):
            rrf[i] = rrf.get(i, 0.0) + 1.0 / (RRF_K + r)
            detail.setdefault(i, {})['bm25_rank'] = r
            detail.setdefault(i, {})['bm25'] = round(s, 3)
        order = sorted(rrf.items(), key=lambda kv: -kv[1])[:topk]
        out = []
        for rank, (i, sc) in enumerate(order, 1):
            cid = self.chunk_ids[i]
            out.append({
                'chunk_id': cid, 'rank': rank, 'rrf': round(sc, 6), 'row': int(i),
                **detail.get(i, {}),
                **{k: self.by_id[cid][k] for k in ('name', 'section', 'pdf_page', 'kind', 'url', 'text')},
            })
        return out

    # ---------- 生成 ----------
    def retrieve_panorama(self, query: str, per_company: int = 1,
                          mode: str = 'hybrid') -> list[dict]:
        """跨公司「全景题」专用召回：每家公司各取最相关的若干块。

        纯 top-k 相似度召回在这类问题上会严重偏科——问「12 家里谁营收最高」，
        向量只把一两家公司的高分块捞上来，剩下的 10 家根本没进上下文，
        模型只能回「资料里没有」。这里改成按公司逐个召回，保证 12 家都在场。
        """
        out: list[dict] = []
        for name in COMPANIES:
            for h in self.retrieve(query, topk=per_company, mode=mode, company=name):
                h = dict(h)
                h['rank'] = len(out) + 1
                out.append(h)
        return out

    def answer(self, query: str, mode: str = 'hybrid', topk: int = 6,
               company: str | None = None, max_chars_per_evidence: int = 1200):
        t0 = time.time()
        # 问题里点名了公司 → 只在该公司的块里检索（表格块常不含公司名，不限定就会串公司）
        scope = company or detect_company(query)
        # 口语化提问先改写成检索友好的查询（「赚了多少钱」→「归属于上市公司股东的净利润」）
        rq = rewrite_query(query)
        pano = scope is None and is_panorama(query)
        why = is_why_question(query)
        if pano:
            # 全景题每家只取几块，必须保证「主要会计数据和财务指标」这张汇总表被召回到，
            # 否则会出现「拿某家的母公司口径去和别人家的合并口径比大小」
            hits = self.retrieve_panorama(rq + ' 主要会计数据和财务指标',
                                          per_company=3, mode=mode)
            max_chars_per_evidence = min(max_chars_per_evidence, 450)
        else:
            hits = self.retrieve(rq, topk=topk, mode=mode, company=scope,
                                 sections=NARRATIVE_SECTIONS if why else None)
        if not hits:
            return {'query': query, 'answer': '资料库中没有检索到相关内容，无法回答。',
                    'citations': [], 'elapsed': round(time.time() - t0, 2), 'mode': mode,
                    'company_scope': scope, 'query_rewritten': rq, 'panorama': pano}

        ev = []
        for n, h in enumerate(hits, 1):
            body = h['text'][:max_chars_per_evidence]
            ev.append(f"[{n}] 公司：{h['name']}｜章节：{h['section']}｜PDF 第 {h['pdf_page']} 页"
                      f"｜类型：{'表格' if h['kind'] == 'table' else '正文'}\n{body}")
        evidence = '\n\n'.join(ev)

        pano_rule = (
            "8) **这是一道跨公司全景题**：下面的资料已按公司逐家覆盖（每家几份最相关的片段），"
            "请把各家数值逐一对齐后比较，明确给出结论（是谁、多少），并点明你比较了哪几家。\n"
            "   比较时务必**口径一致**：优先取各公司「主要会计数据」表里的 2025 年数值；"
            "如果某家公司只找到母公司口径、分季度分项或其他非合并口径的数字，"
            "必须注明口径差异，**不要**把它和别家的合并口径直接比大小。\n"
            if pano else '')
        prompt = (
            f"你是一个严格基于给定资料回答问题的财报分析助手。\n"
            f"规则：\n"
            f"1) 只使用下面「资料」中的信息作答，绝对不要引入外部知识或自行估算。\n"
            f"2) 答案中每一个具体数字、事实后面都必须标注来源编号，如 [1] 或 [2][3]。\n"
            f"3) 资料里找不到答案时，直接回答「资料中未找到」，不要猜测。\n"
            f"4) 严格区分财务口径，不要混用：\n"
            f"   - 「营业总收入」≠「营业收入」；\n"
            f"   - 「净利润」≠「归属于上市公司股东的净利润」（归母净利润）≠「扣非净利润」。\n"
            f"   问题问哪个口径就答哪个口径；若资料里两个口径都出现，以问题所问的为准，\n"
            f"   并可用一句话注明另一口径的数值作对照。\n"
            f"5) 问题有几问就答几问，每一问都要给出结果，不要漏答。\n"
            f"6) 金额写清数值、单位和报告期（如「2025 年营业收入 4,564.52 亿元」）。\n"
            f"7) 回答用简体中文，简洁分段，不要复述规则。\n"
            f"{pano_rule}\n"
            f"资料：\n{evidence}\n\n"
            f"问题：{query}"
        )
        text, usage = chat(prompt)
        cites = sorted({int(x) for x in re.findall(r'\[(\d{1,2})\]', text or '')})
        return {
            'query': query, 'answer': text, 'mode': mode, 'company_scope': scope,
            'query_rewritten': rq, 'panorama': pano,
            'citations': [{'n': n, 'chunk_id': hits[n - 1]['chunk_id'],
                           'name': hits[n - 1]['name'], 'section': hits[n - 1]['section'],
                           'pdf_page': hits[n - 1]['pdf_page'], 'kind': hits[n - 1]['kind'],
                           'url': hits[n - 1]['url']}
                          for n in cites if 1 <= n <= len(hits)],
            'retrieved': [{k: h[k] for k in ('chunk_id', 'name', 'section', 'pdf_page', 'kind',
                                             'url', 'rank', 'rrf', 'vec_rank', 'bm25_rank',
                                             'cos', 'bm25', 'text') if k in h} for h in hits],
            'evidence_used': len(hits),
            'usage': usage,
            'elapsed': round(time.time() - t0, 2),
        }


# ---------------- LLM 通道 ----------------
def chat(prompt: str, system: str | None = None, temperature: float = 0.0,
         timeout: int = 120):
    import urllib.request
    key = os.environ.get('DEEPSEEK_API_KEY', '').strip()
    base = os.environ.get('DEEPSEEK_BASE_URL', 'https://api.deepseek.com/v1').rstrip('/')
    model = os.environ.get('DEEPSEEK_MODEL', 'deepseek-chat')
    if not key:
        return ('（未配置 DEEPSEEK_API_KEY，无法生成答案。检索结果仍然可用。）', {})
    msgs = ([{'role': 'system', 'content': system}] if system else []) + \
           [{'role': 'user', 'content': prompt}]
    payload = json.dumps({'model': model, 'messages': msgs,
                          'temperature': temperature, 'max_tokens': 1200}).encode()
    req = urllib.request.Request(base + '/chat/completions', data=payload, headers={
        'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'})
    try:
        r = json.loads(urllib.request.urlopen(req, timeout=timeout).read())
        return r['choices'][0]['message']['content'], r.get('usage', {})
    except Exception as e:
        return (f'（生成失败：{e}）', {})
