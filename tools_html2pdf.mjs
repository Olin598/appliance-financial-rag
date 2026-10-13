import { chromium } from 'playwright-core';
import { resolve } from 'node:path';
const src = resolve(process.argv[2]);
const out = resolve(process.argv[3]);
const b = await chromium.launch({ channel: 'msedge', headless: true });
const p = await b.newPage();
await p.goto('file://' + src, { waitUntil: 'load' });
await p.waitForTimeout(600);
await p.pdf({ path: out, format: 'A4', printBackground: true,
  margin: { top: '12mm', bottom: '12mm', left: '13mm', right: '13mm' } });
await b.close();
console.log('PDF →', out);
