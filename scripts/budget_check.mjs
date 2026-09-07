#!/usr/bin/env node
// 性能バジェットの検証（docs/07 §7.5）。
//
// CI が強制するのはサイズと品質の静的指標のみ。時間は測らない——GitHub ランナーに
// GPU が無く、SwiftShader で測った数字は実機について何も語らないため。
import { gzipSync } from 'node:zlib';
import { readFileSync, readdirSync, statSync, existsSync } from 'node:fs';
import { join, extname } from 'node:path';

const args = process.argv.slice(2);
const argOf = (name, dflt) => {
  const i = args.indexOf(name);
  return i >= 0 && args[i + 1] ? args[i + 1] : dflt;
};
const DIST = argOf('--dist', 'dist');
const METRICS = argOf('--metrics', 'tests/results/metrics.json');
const FAIL_MODE = args.includes('--fail');
const budget = JSON.parse(readFileSync(argOf('--budget', 'budget.json'), 'utf8'));

/** dist を再帰的に走査してファイル一覧を返す。 */
function walk(dir, out = []) {
  if (!existsSync(dir)) return out;
  for (const name of readdirSync(dir)) {
    const p = join(dir, name);
    const st = statSync(p);
    if (st.isDirectory()) walk(p, out);
    else out.push({ path: p, rel: p.slice(DIST.length + 1), bytes: st.size });
  }
  return out;
}

const gz = (p) => gzipSync(readFileSync(p), { level: 9 }).length;
const mb = (n) => `${(n / 1e6).toFixed(2)} MB`;
const kb = (n) => `${(n / 1e3).toFixed(1)} KB`;

const files = walk(DIST);
if (files.length === 0) {
  console.error(`[budget] ${DIST}/ が見つかりません。先に vite build を実行してください。`);
  process.exit(2);
}

// --- 実測値を集める ---------------------------------------------------------
const isModel = (f) => f.rel.startsWith('models/') && f.rel.endsWith('.onnx');
const isOrtWasm = (f) => f.rel.startsWith('ort/') && f.rel.endsWith('.wasm');

const jsGzip = files.filter((f) => extname(f.path) === '.js').reduce((a, f) => a + gz(f.path), 0);
const cssGzip = files.filter((f) => extname(f.path) === '.css').reduce((a, f) => a + gz(f.path), 0);

// WebGPU 経路で実際に読まれるのは jsep 版だけ。素の wasm はフォールバック時のみ。
const jsepWasm = files.find((f) => isOrtWasm(f) && f.rel.includes('jsep'));
const ortWasmGzip = jsepWasm ? gz(jsepWasm.path) : 0;

const modelBytes = Object.fromEntries(
  files.filter(isModel).map((f) => [f.rel.replace(/^models\//, '').replace(/\.onnx$/, ''), f.bytes]),
);
const pick = (re) =>
  Object.entries(modelBytes).filter(([k]) => re.test(k)).reduce((a, [, v]) => a + v, 0);

/**
 * 端末が実際に落とす量を、バックエンドごとに数える。
 *
 * D22 でモデルはバックエンドぶん置かれるようになった（深度モデルなら
 * WebGPU 用 q4f16 と WASM 用 uint8 の2つ）。サーバ上のファイルを単純に
 * 足すと、1台の端末が落とす量の2倍近くになってしまう。バジェットが
 * 見たいのは「利用者が待つ量」なので、マニフェストの byBackend を見て
 * バックエンドごとに数え、**重いほう**で判定する。
 *
 * マニフェストが無い場合（モデル未焼き込み）は、従来どおりファイル名で拾う。
 */
const manifestPath = join(DIST, 'models', 'manifest.json');
const manifest = existsSync(manifestPath) ? JSON.parse(readFileSync(manifestPath, 'utf8')) : null;

const bytesOfFile = (file) => modelBytes[file.replace(/\.onnx$/, '')] ?? 0;

/** マニフェストから、そのバックエンドで id 群が要する実バイト数。 */
const backendBytes = (backend, ids) => {
  if (!manifest) return null;
  const seen = new Map();
  for (const id of ids) {
    const entry = manifest.models?.find((m) => m.id === id);
    const file = entry?.byBackend?.[backend] ?? entry?.file;
    if (file) seen.set(file, bytesOfFile(file));
  }
  return [...seen.values()].reduce((a, v) => a + v, 0);
};

const backends = manifest?.backends ?? ['webgpu', 'wasm'];
const personIds = ['depth-anything-v3-small', 'modnet', 'mi-gan'];
const objectIds = ['isnet-general'];

const depth = pick(/^depth-anything-v3-small\./) || pick(/^depth-anything-v2-small\./);
const person = pick(/^modnet\./);
const inpaint = pick(/^mi-gan\./);

// 初回必須は、バックエンドごとに数えて重いほうを採る。
const firstVisitPerBackend = manifest
  ? backends.map((b) => ({ backend: b, bytes: backendBytes(b, personIds) ?? 0 }))
  : [];
const firstVisit = firstVisitPerBackend.length
  ? Math.max(...firstVisitPerBackend.map((x) => x.bytes))
  : depth + person + inpaint;

const objectExtra = manifest
  ? Math.max(...backends.map((b) => backendBytes(b, objectIds) ?? 0))
  : pick(/^isnet-general\.|^rmbg/);

// サーバ側の総量。1台が落とす量ではないので、別の指標として持つ。
const modelTotal = Object.values(modelBytes).reduce((a, v) => a + v, 0);
const pagesTotal = files.reduce((a, f) => a + f.bytes, 0);

const metrics = existsSync(METRICS) ? JSON.parse(readFileSync(METRICS, 'utf8')) : null;

// --- 判定 -------------------------------------------------------------------
const results = [];
const skipped = [];
const add = (label, value, fmt, rule) => {
  if (!rule) return;
  // 未計測（undefined/null）は 0 とみなさずスキップする。
  // 0 として判定すると「まだ測っていない」が「上限違反」に化ける。
  if (value === undefined || value === null || Number.isNaN(value)) {
    skipped.push(label);
    return;
  }
  let level = 'ok';
  let limit = '';
  if (rule.failRange || rule.warnRange) {
    const [wlo, whi] = rule.warnRange ?? [];
    const [flo, fhi] = rule.failRange ?? [];
    if (flo !== undefined && (value < flo || value > fhi)) level = 'fail';
    else if (wlo !== undefined && (value < wlo || value > whi)) level = 'warn';
    limit = `${wlo}〜${whi}`;
  } else if (rule.failMin !== undefined || rule.warnMin !== undefined) {
    if (rule.failMin !== undefined && value < rule.failMin) level = 'fail';
    else if (rule.warnMin !== undefined && value < rule.warnMin) level = 'warn';
    limit = `≥ ${rule.warnMin ?? rule.failMin}`;
  } else {
    if (rule.fail !== undefined && value > rule.fail) level = 'fail';
    else if (rule.warn !== undefined && value > rule.warn) level = 'warn';
    limit = `≤ ${fmt(rule.warn ?? rule.fail)}`;
  }
  results.push({ label, value, shown: fmt(value), limit, level });
};

add('JS バンドル (gzip)', jsGzip, kb, budget.bundle?.['js.gzip']);
add('CSS バンドル (gzip)', cssGzip, kb, budget.bundle?.['css.gzip']);
add('ORT wasm jsep (gzip)', ortWasmGzip, mb, budget.runtime?.['ort.wasm.gzip']);
add('初回必須モデル (人物)', firstVisit, mb, budget.models?.['firstVisit.person']);
add('物体モード追加分', objectExtra, mb, budget.models?.['objectMode.extra']);
add('モデル合計 (配信側)', modelTotal, mb, budget.models?.total);
add('Pages 全体 (非圧縮)', pagesTotal, mb, budget.pages?.totalUncompressed);

if (metrics) {
  add('.pgs サイズ', metrics.pgsBytes, kb, budget.output?.['pgs.bytes']);
  add('ガウシアン数', metrics.gaussianCount, (n) => n.toLocaleString('ja-JP'), budget.output?.gaussianCount);
  add('SSIM (入力視点)', metrics.ssimInputView, (n) => n.toFixed(3), budget.quality?.['ssim.inputView']);
  add('穴の割合 (±45°)', metrics.holeRatio45, (n) => `${(n * 100).toFixed(2)}%`, budget.quality?.['holeRatio.45deg']);
  add('縞指標の改善 (±45°)', metrics.stripeGain45, (n) => `${n.toFixed(2)} nat`, budget.quality?.['stripeGain.45deg']);
}

// --- 出力 -------------------------------------------------------------------
const icon = { ok: '  ', warn: '⚠ ', fail: '✗ ' };
// 全角文字は2列を占めるので、文字数ではなく表示幅で揃える
const width = (s) => [...s].reduce((n, c) => n + (/[\u3000-\u9fff\uff00-\uff60]/.test(c) ? 2 : 1), 0);
const labelWidth = Math.max(...results.map((r) => width(r.label))) + 2;
console.log('\n性能バジェット\n');
for (const r of results) {
  const gap = ' '.repeat(Math.max(1, labelWidth - width(r.label)));
  console.log(`${icon[r.level]}${r.label}${gap}${r.shown.padStart(11)}   (${r.limit})`);
}

const fails = results.filter((r) => r.level === 'fail');
const warns = results.filter((r) => r.level === 'warn');
console.log(`\n合格 ${results.length - fails.length - warns.length} / 警告 ${warns.length} / 失敗 ${fails.length}`);
if (!metrics) console.log(`（${METRICS} が無いため、出力サイズと品質の判定はスキップしました）`);
else if (skipped.length) console.log(`（未計測のためスキップ: ${skipped.join('、')}）`);

if (process.env.GITHUB_STEP_SUMMARY) {
  const rows = results.map((r) => `| ${icon[r.level].trim() || '✓'} | ${r.label} | ${r.shown} | ${r.limit} |`);
  const md = ['## 性能バジェット', '', '| | 項目 | 実測 | 上限 |', '|---|---|---|---|', ...rows, ''].join('\n');
  const { appendFileSync } = await import('node:fs');
  appendFileSync(process.env.GITHUB_STEP_SUMMARY, md);
}

if (FAIL_MODE && fails.length > 0) {
  console.error(`\n[budget] ${fails.length} 項目が上限を超えました。`);
  process.exit(1);
}
