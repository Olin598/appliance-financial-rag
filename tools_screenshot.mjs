/**
 * 作业3 · 生成问答页面截图
 *
 * 用本机已安装的 Microsoft Edge（channel: msedge）无头驱动，
 * 真实打开页面 → 输入问题 → 等答案和证据渲染完 → 整页截图。
 * 不是拼图，也不是手写 HTML 再截图。
 *
 * 用法：
 *   NODE_PATH=<workspace>/node_modules node tools_screenshot.mjs
 * 前置：问答页面已在 http://127.0.0.1:8765 运行
 */
import { chromium } from 'playwright-core';
import fs from 'fs';

const BASE = process.env.BASE || 'http://127.0.0.1:8765';
const OUT = process.env.OUT_DIR || 'deliverables';

const SHOTS = [
  {
    file: '问答页面.png',
    q: '格力电器 2025 年归属于上市公司股东的净利润是多少？同比变化如何？',
    waitMs: 45000,
  },
  {
    file: '全景问答.png',
    q: '这批 12 家家电公司里，2025 年营业收入最高和最低的分别是谁？各是多少？',
    waitMs: 90000,
  },
];

fs.mkdirSync(OUT, { recursive: true });

const browser = await chromium.launch({ channel: 'msedge', headless: true });
const ctx = await browser.newContext({
  viewport: { width: 1280, height: 900 },
  deviceScaleFactor: 2,
  locale: 'zh-CN',
});
const page = await ctx.newPage();

for (const shot of SHOTS) {
  console.log(`[shot] ${shot.file} ← ${shot.q}`);
  await page.goto(BASE, { waitUntil: 'networkidle' });
  await page.waitForSelector('#chips .chip', { timeout: 20000 });
  await page.fill('#q', shot.q);
  await page.click('#go');
  // 等回答真的出来（不再显示“正在检索并生成…”）
  await page.waitForFunction(
    () => {
      const el = document.querySelector('#ans');
      const txt = el ? el.innerText : '';
      return txt && !txt.includes('正在检索') && txt.length > 20;
    },
    { timeout: shot.waitMs }
  ).catch(() => console.log('  (等待超时，按当前状态截图)'));
  await page.waitForTimeout(1500);
  await page.screenshot({ path: `${OUT}/${shot.file}`, fullPage: true });
  const first = await page.locator('#ans').innerText().catch(() => '');
  console.log(`  答案开头：${first.slice(0, 80).replace(/\n/g, ' ')}`);
}

await browser.close();
console.log('截图输出到', OUT);
