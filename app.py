#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
作业3（方向A）· 问答页面（FastAPI）

启动：  python app.py            （默认 http://127.0.0.1:8765）
只监听本机回环地址；页面把模型输出与原文都做 HTML 转义后渲染。

接口：
  GET  /                问答页面
  POST /api/ask         {"query": "...", "mode": "hybrid|vector|bm25",
                         "company": "美的集团" 或 null, "topk": 6}
  GET  /api/stats       语料与索引统计
"""
from __future__ import annotations

import json
import os

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

load_dotenv()

import rag as ragmod  # noqa: E402

app = FastAPI(title='家电财报有据 · 问答')
RAG: ragmod.Rag | None = None


def get_rag() -> ragmod.Rag:
    global RAG
    if RAG is None:
        RAG = ragmod.Rag()
    return RAG


class AskIn(BaseModel):
    query: str
    mode: str = 'hybrid'
    company: str | None = None
    topk: int = 6


@app.post('/api/ask')
def api_ask(body: AskIn):
    r = get_rag()
    if not body.query.strip():
        return JSONResponse({'error': '问题为空'}, status_code=400)
    out = r.answer(body.query, mode=body.mode, topk=max(1, min(body.topk, 12)),
                   company=body.company or None)
    return JSONResponse(out)


@app.get('/api/stats')
def api_stats():
    p = 'data/corpus_stats.json'
    stats = json.load(open(p, encoding='utf-8')) if os.path.exists(p) else {}
    meta = json.load(open('data/sources.json', encoding='utf-8'))
    return JSONResponse({
        'corpus': stats,
        'reports': [{'code': r['code'], 'name': r['name'], 'disclosed': r['disclosed'],
                     'url': r['url']} for r in meta['reports']],
        'companies': [r['name'] for r in meta['reports']],
        'llm_configured': bool(os.environ.get('DEEPSEEK_API_KEY')),
    })


@app.get('/', response_class=HTMLResponse)
def index():
    return HTML_PAGE


HTML_PAGE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>家电财报有据 · 带出处的年报问答</title>
<style>
  :root{
    --bg:#f5f6f8; --card:#ffffff; --line:#e3e6ea; --ink:#1b1f24; --sub:#6b7280;
    --accent:#0f5fa8; --accent-soft:#eaf2fb; --ok:#0a7a4a;
  }
  *{box-sizing:border-box;margin:0;padding:0}
  body{background:var(--bg);color:var(--ink);
       font-family:-apple-system,BlinkMacSystemFont,"PingFang SC","Hiragino Sans GB","Microsoft YaHei",sans-serif;
       line-height:1.65;padding:28px 16px 64px}
  .wrap{max-width:960px;margin:0 auto}
  header h1{font-size:1.5rem;letter-spacing:-.2px}
  header p{color:var(--sub);font-size:.9rem;margin-top:6px}
  .card{background:var(--card);border:1px solid var(--line);border-radius:12px;
        padding:18px 20px;margin-top:18px}
  .row{display:flex;gap:10px;flex-wrap:wrap;align-items:center}
  textarea{width:100%;min-height:74px;resize:vertical;border:1px solid var(--line);
           border-radius:8px;padding:11px 12px;font-size:.95rem;font-family:inherit;background:#fcfcfd}
  textarea:focus{outline:2px solid var(--accent-soft);border-color:var(--accent)}
  select,button{font-family:inherit;font-size:.9rem;border-radius:8px;
                border:1px solid var(--line);padding:8px 12px;background:#fff;cursor:pointer}
  button.go{background:var(--accent);color:#fff;border-color:var(--accent);
            padding:9px 22px;font-weight:600}
  button.go:disabled{opacity:.55;cursor:progress}
  .chips{margin-top:10px;display:flex;gap:8px;flex-wrap:wrap}
  .chip{font-size:.82rem;color:var(--accent);background:var(--accent-soft);
        border:1px solid #d3e4f6;border-radius:999px;padding:4px 11px;cursor:pointer}
  .chip:hover{background:#dceaf9}
  h2{font-size:1.02rem;margin-bottom:10px}
  .ans{white-space:pre-wrap;font-size:.96rem}
  .cite{display:inline-block;min-width:19px;text-align:center;background:var(--accent-soft);
        color:var(--accent);border-radius:5px;font-size:.76rem;padding:0 5px;
        margin:0 2px;vertical-align:2px;text-decoration:none;font-weight:600}
  .ev{border-top:1px dashed var(--line);padding:11px 0;font-size:.86rem}
  .ev:first-of-type{border-top:0}
  .ev .meta{color:var(--sub);font-size:.8rem;margin-bottom:4px}
  .ev .meta b{color:var(--accent);font-weight:600}
  .ev .txt{white-space:pre-wrap;color:#39414a;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;
           font-size:.79rem;max-height:170px;overflow:auto;background:#fafbfc;
           border:1px solid var(--line);border-radius:6px;padding:8px}
  .ev a{color:var(--accent);text-decoration:none;font-size:.78rem}
  .muted{color:var(--sub);font-size:.85rem}
  .tag{display:inline-block;font-size:.74rem;border:1px solid var(--line);border-radius:5px;
       padding:1px 7px;color:var(--sub);margin-left:6px}
  .metric{display:flex;gap:18px;flex-wrap:wrap;color:var(--sub);font-size:.82rem;margin-top:10px}
  footer{color:var(--sub);font-size:.8rem;margin-top:26px;text-align:center}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>家电财报有据 · 带出处的年报问答</h1>
    <p>12 家家用电器上市公司 2025 年年度报告全文 · 混合检索（BGE 向量 + BM25 / RRF）· 答案逐句标注来源</p>
  </header>

  <div class="card">
    <textarea id="q" placeholder="例如：美的集团 2025 年营业收入是多少？／三家白电龙头谁的毛利率最高？"></textarea>
    <div class="row" style="margin-top:10px">
      <select id="mode">
        <option value="hybrid">混合检索（向量 + BM25）</option>
        <option value="vector">纯向量</option>
        <option value="bm25">纯关键词 BM25</option>
      </select>
      <select id="company"><option value="">不限公司</option></select>
      <select id="topk">
        <option value="6">取 6 块证据</option>
        <option value="4">取 4 块证据</option>
        <option value="10">取 10 块证据</option>
      </select>
      <button class="go" id="go">提问</button>
      <span class="muted" id="hint"></span>
    </div>
    <div class="chips" id="chips"></div>
  </div>

  <div class="card" id="ansCard" style="display:none">
    <h2>回答</h2>
    <div class="ans" id="ans"></div>
    <div class="metric" id="metric"></div>
  </div>

  <div class="card" id="evCard" style="display:none">
    <h2>召回证据<span class="tag">编号与回答里的 [n] 对应</span></h2>
    <div id="ev"></div>
  </div>

  <footer>数据来源：巨潮资讯网（cninfo.com.cn）官方披露的年报 PDF · 仅供课程作业使用</footer>
</div>

<script>
const $ = s => document.querySelector(s);
const DEMO = [
  '美的集团 2025 年营业收入是多少？',
  '格力电器 2025 年归母净利润同比变化如何？',
  '对比美的集团、格力电器、海尔智家 2025 年的毛利率，谁最高？',
  '12 家公司 2025 年研发投入金额最高的是哪家？'
];

function esc(s){return String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));}

function renderAnswer(text){
  let h = esc(text || '');
  h = h.replace(/\[(\d{1,2})\]/g, (m,n)=>`<a class="cite" href="#ev-${n}">${n}</a>`);
  return h;
}

async function boot(){
  const r = await fetch('/api/stats'); const d = await r.json();
  const sel = $('#company');
  (d.companies||[]).forEach(c=>{const o=document.createElement('option');o.value=c;o.textContent=c;sel.appendChild(o);});
  const c = d.corpus || {};
  $('#hint').textContent = c.total_chunks ? `语料 ${c.total_chunks} 块 · ${(d.companies||[]).length} 家公司` : '';
  if(!d.llm_configured) $('#hint').textContent += '（未配置模型 key，仅返回检索结果）';
  const chips = $('#chips');
  DEMO.forEach(q=>{const b=document.createElement('span');b.className='chip';b.textContent=q;
    b.onclick=()=>{$('#q').value=q;ask();};chips.appendChild(b);});
}

async function ask(){
  const q = $('#q').value.trim(); if(!q) return;
  const btn = $('#go'); btn.disabled = true; btn.textContent = '检索中…';
  $('#ansCard').style.display='block'; $('#evCard').style.display='block';
  $('#ans').innerHTML = '<span class="muted">正在检索并生成…</span>'; $('#ev').innerHTML='';
  try{
    const r = await fetch('/api/ask',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({query:q,mode:$('#mode').value,company:$('#company').value||null,topk:+$('#topk').value})});
    const d = await r.json();
    $('#ans').innerHTML = renderAnswer(d.answer);
    $('#metric').innerHTML = `检索模式：${esc(d.mode||'')}　耗时：${d.elapsed??'-'}s　证据块：${d.evidence_used??0}　引用：${(d.citations||[]).length}`;
    const ev = $('#ev');
    (d.retrieved||[]).forEach((h,i)=>{
      const n = i+1;
      const el = document.createElement('div'); el.className='ev'; el.id='ev-'+n;
      el.innerHTML = `<div class="meta"><b>[${n}] ${esc(h.name)}</b> · ${esc(h.section||'')} · PDF 第 ${h.pdf_page} 页 · ${h.kind==='table'?'表格':'正文'}`
        + (h.vec_rank?` · 向量第${h.vec_rank}名`:'') + (h.bm25_rank?` · BM25第${h.bm25_rank}名`:'')
        + ` · 块 ${esc(h.chunk_id)}</div>`
        + `<div class="txt">${esc(h.text)}</div>`
        + `<a href="${esc(h.url)}" target="_blank" rel="noopener">在官方年报 PDF 中打开第 ${h.pdf_page} 页 ↗</a>`;
      ev.appendChild(el);
    });
  }catch(e){
    $('#ans').innerHTML = '<span class="muted">请求失败：'+esc(e)+'</span>';
  }finally{ btn.disabled=false; btn.textContent='提问'; }
}

$('#go').onclick = ask;
$('#q').addEventListener('keydown', e=>{ if((e.metaKey||e.ctrlKey)&&e.key==='Enter') ask(); });
boot();
</script>
</body>
</html>
"""

if __name__ == '__main__':
    host = os.environ.get('HOST', '127.0.0.1')
    port = int(os.environ.get('PORT', '8765'))
    uvicorn.run(app, host=host, port=port, log_level='info')
