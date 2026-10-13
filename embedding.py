#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
作业3（方向A）· 向量编码器

两种后端，自动选择：
1. local  —— 若 models/bge-small-zh-v1.5/model.onnx 存在，用 onnxruntime 在本地 CPU 推理
              （BAAI/bge-small-zh-v1.5，512 维，CLS pooling + L2 归一化，完全离线）
2. dashscope —— 否则若配置了 DASHSCOPE_API_KEY，用阿里云百炼 text-embedding-v4（1024 维）

检索问题加 BGE 官方推荐指令前缀（仅 local 后端使用），文档不加。
带磁盘缓存，重复运行不会重复计费。
"""
from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.request

import numpy as np

MODEL_DIR = os.environ.get('BGE_MODEL_DIR', 'models/onnx-bge-small-zh')
ONNX_PATH = os.path.join(MODEL_DIR, 'model.onnx')
CACHE_PATH = os.environ.get('EMB_CACHE', 'data/index/emb_cache_onnx.jsonl')
QUERY_INSTRUCTION = '为这个句子生成表示以用于检索相关文章：'
MAX_LEN = 512
DS_BATCH = 10

_session = None
_tokenizer = None
_backend = None
_cache: dict[str, list[float]] = {}
_cache_dirty = 0


# ---------------- 本地 ONNX ----------------
def _load_onnx():
    global _session, _tokenizer
    import onnxruntime as ort
    from tokenizers import Tokenizer
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = max(1, (os.cpu_count() or 4) - 1)
    opts.inter_op_num_threads = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    _session = ort.InferenceSession(ONNX_PATH, sess_options=opts,
                                    providers=['CPUExecutionProvider'])
    _tokenizer = Tokenizer.from_file(os.path.join(MODEL_DIR, 'tokenizer.json'))
    _tokenizer.enable_truncation(max_length=MAX_LEN)
    _tokenizer.enable_padding()


def _encode_onnx(texts: list[str], batch_size: int = 32) -> np.ndarray:
    _load_onnx()
    names = {i.name for i in _session.get_inputs()}
    out = []
    for i in range(0, len(texts), batch_size):
        encs = _tokenizer.encode_batch(texts[i:i + batch_size])
        ids = np.array([e.ids for e in encs], dtype=np.int64)
        mask = np.array([e.attention_mask for e in encs], dtype=np.int64)
        feed = {'input_ids': ids, 'attention_mask': mask}
        if 'token_type_ids' in names:
            feed['token_type_ids'] = np.zeros_like(ids)
        v = _session.run(None, feed)[0][:, 0, :]      # CLS pooling
        n = np.linalg.norm(v, axis=1, keepdims=True)
        n[n == 0] = 1.0
        out.append(v / n)                              # L2 归一化
    return np.vstack(out).astype(np.float32)


# ---------------- 阿里云百炼 ----------------
DS_MAX_CHARS = 2400     # 单条文本截断长度，避免超出模型的 token 上限
_ds_err_printed = 0


def _ds_one_call(chunk: list[str], key: str, url: str, model: str) -> list[list[float]]:
    payload = json.dumps({'model': model, 'input': chunk,
                          'dimensions': 1024, 'encoding_format': 'float'}).encode()
    req = urllib.request.Request(url, data=payload, headers={
        'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'})
    r = json.loads(urllib.request.urlopen(req, timeout=120).read())
    data = sorted(r['data'], key=lambda x: x['index'])
    return [d['embedding'] for d in data]


def _ds_embed(texts: list[str]) -> np.ndarray:
    global _ds_err_printed
    key = os.environ.get('DASHSCOPE_API_KEY', '').strip()
    model = os.environ.get('DASHSCOPE_EMB_MODEL', 'text-embedding-v4')
    url = 'https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings'
    out: list[list[float]] = []
    dim = int(os.environ.get('DASHSCOPE_EMB_DIM', '1024'))

    for i in range(0, len(texts), DS_BATCH):
        chunk = [(t or ' ')[:DS_MAX_CHARS] for t in texts[i:i + DS_BATCH]]
        vecs: list[list[float]] | None = None
        for attempt in range(6):
            try:
                vecs = _ds_one_call(chunk, key, url, model)
                break
            except urllib.error.HTTPError as e:
                body = ''
                try:
                    body = e.read().decode('utf-8', 'replace')[:300]
                except Exception:
                    pass
                if _ds_err_printed < 3:
                    print(f'  [embed] HTTP {e.code} {body}', flush=True)
                    _ds_err_printed += 1
                time.sleep(1.5 * (attempt + 1))
            except Exception as e:
                if _ds_err_printed < 3:
                    print(f'  [embed] {type(e).__name__}: {e}', flush=True)
                    _ds_err_printed += 1
                time.sleep(1.5 * (attempt + 1))

        if vecs is None:
            # 整批失败 → 逐条重试；仍失败的用零向量占位，保证流程能跑完
            vecs = []
            for t in chunk:
                got = None
                for attempt in range(3):
                    try:
                        got = _ds_one_call([t], key, url, model)[0]
                        break
                    except Exception:
                        time.sleep(1.0 * (attempt + 1))
                if got is None:
                    print(f'  [embed] 单条失败，用零向量占位：{t[:40]!r}', flush=True)
                    got = [0.0] * dim
                vecs.append(got)
        out.extend(vecs)

    v = np.array(out, dtype=np.float32)
    n = np.linalg.norm(v, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return v / n


def _pick_backend() -> str:
    global _backend
    if _backend:
        return _backend
    if os.path.exists(ONNX_PATH) and os.path.getsize(ONNX_PATH) > 50_000_000:
        _backend = 'local'
    elif os.environ.get('DASHSCOPE_API_KEY', '').strip():
        _backend = 'dashscope'
    else:
        raise RuntimeError('既没有可用的本地 ONNX 模型，也没有配置 DASHSCOPE_API_KEY')
    return _backend


def _load_cache():
    if _cache or not os.path.exists(CACHE_PATH):
        return
    for line in open(CACHE_PATH, encoding='utf-8'):
        try:
            o = json.loads(line)
            _cache[o['k']] = o['v']
        except Exception:
            pass


def _key(text: str) -> str:
    return hashlib.sha1(text.encode('utf-8')).hexdigest()


def encode(texts, is_query: bool = False) -> np.ndarray:
    global _cache_dirty
    if isinstance(texts, str):
        texts = [texts]
    backend = _pick_backend()
    if backend == 'local' and is_query:
        texts = [QUERY_INSTRUCTION + t for t in texts]

    _load_cache()
    result: list[list[float] | None] = [_cache.get(_key(t)) for t in texts]
    todo = [i for i, v in enumerate(result) if v is None]
    if todo:
        batch = [texts[i] for i in todo]
        vecs = _encode_onnx(batch) if backend == 'local' else _ds_embed(batch)
        os.makedirs(os.path.dirname(CACHE_PATH) or '.', exist_ok=True)
        with open(CACHE_PATH, 'a', encoding='utf-8') as f:
            for k, i in enumerate(todo):
                v = [round(float(x), 6) for x in vecs[k]]
                result[i] = v
                f.write(json.dumps({'k': _key(texts[i]), 'v': v}) + '\n')
        _cache_dirty += len(todo)
    return np.array(result, dtype=np.float32)


def backend_name() -> str:
    b = _pick_backend()
    if b == 'local':
        return 'bge-small-zh-v1.5（本地 ONNX，512 维）'
    return 'text-embedding-v4（阿里云百炼，1024 维）'


def dim() -> int:
    return 512 if _pick_backend() == 'local' else 1024
