/**
 * PoC-1 のページ制御（docs/08 §8.11）。
 *
 * 実機で開いて実行し、結果をコピーして共有してもらう診断ツール。
 * 本体アプリとは独立に配信する（poc.html）。
 */
import { CANDIDATES, benchRender, testModel, type Candidate, type ModelResult, type RenderBenchResult } from './bench';
import { detectCapability, describeCapability, type Capability } from '../runtime/capability';
import { cachedBytes, clearModelCache, configureOrt, type Backend } from '../runtime/OrtSession';

const $ = <T extends HTMLElement>(id: string): T => {
  const el = document.getElementById(id);
  if (!el) throw new Error(`要素がありません: ${id}`);
  return el as T;
};

const state: {
  capability?: Capability;
  models: ModelResult[];
  render: RenderBenchResult[];
} = { models: [], render: [] };

const mb = (n: number) => `${(n / 1e6).toFixed(1)} MB`;

// --- ① 端末の能力 -----------------------------------------------------------

async function showCapability(): Promise<Capability> {
  const cap = await detectCapability();
  state.capability = cap;

  const lines = describeCapability(cap);
  const cached = await cachedBytes();
  const ok = cap.webgpu.supported;
  $('cap').innerHTML =
    `<div class="row" style="margin-bottom:8px">
       <span class="pill ${ok ? 'ok' : 'ng'}">${ok ? 'WebGPU 利用可' : 'WebGPU 利用不可'}</span>
       <span class="k">キャッシュ済み ${mb(cached)}</span>
     </div>` +
    `<pre>${lines.map(esc).join('\n')}\n\nUA: ${esc(navigator.userAgent)}</pre>`;

  if (ok) {
    // 自前のアダプタを ORT に共有させる（docs/02 §2.5）
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    configureOrt({ ...(adapter ? { adapter } : {}), wasmThreads: cap.wasmThreads });
  } else {
    configureOrt({ wasmThreads: cap.wasmThreads });
  }
  return cap;
}

// --- ② モデル ---------------------------------------------------------------

function renderModelList(): void {
  const host = $('models');
  host.innerHTML = CANDIDATES.map(
    (c) => `
    <div class="item" data-id="${c.id}">
      <div class="row">
        <span class="name grow">${esc(c.label)}</span>
        <span class="k mono">${c.approxMB} MB</span>
        <span class="pill idle" data-status>未実行</span>
      </div>
      ${c.note ? `<div class="note">${esc(c.note)}</div>` : ''}
      <div class="bar"><i></i></div>
      <div class="res" data-res></div>
    </div>`,
  ).join('');
}

function setStatus(id: string, cls: string, text: string): void {
  const el = document.querySelector(`.item[data-id="${id}"] [data-status]`);
  if (el) {
    el.className = `pill ${cls}`;
    el.textContent = text;
  }
}

function setProgress(id: string, ratio: number): void {
  const el = document.querySelector<HTMLElement>(`.item[data-id="${id}"] .bar i`);
  if (el) el.style.width = `${Math.round(ratio * 100)}%`;
}

function appendResult(id: string, r: ModelResult): void {
  const el = document.querySelector(`.item[data-id="${id}"] [data-res]`);
  if (!el) return;
  const parts: string[] = [];
  if (r.ok) {
    parts.push(
      `<b>${r.backend}</b> 取得 ${r.fetchMs}ms / 生成 ${r.createMs}ms / 初回推論 ${r.warmupMs}ms / <b>推論 ${r.inferMs}ms</b>`,
    );
    parts.push(`出力: ${(r.outputs ?? []).map((o) => `${esc(o)}${fmtShape(r.outputShapes?.[o])}`).join(', ')}`);
  } else {
    parts.push(`<b>${r.backend}</b> 失敗: ${esc(r.error ?? '')}`);
  }
  el.innerHTML += `<div>${parts.join('<br>')}</div>`;
}

const fmtShape = (s: readonly number[] | undefined) => (s ? ` [${s.join('×')}]` : '');

async function runModels(filter: (c: Candidate) => boolean): Promise<void> {
  const cap = state.capability ?? (await showCapability());
  const backends: Backend[] = [];
  if (cap.webgpu.supported) backends.push('webgpu');
  if ($<HTMLInputElement>('useWasm').checked || !cap.webgpu.supported) backends.push('wasm');

  for (const c of CANDIDATES.filter(filter)) {
    const el = document.querySelector(`.item[data-id="${c.id}"] [data-res]`);
    if (el) el.innerHTML = '';
    for (const backend of backends) {
      setStatus(c.id, 'run', `${backend} 実行中`);
      setProgress(c.id, 0);
      const r = await testModel(c, backend, (loaded, total) => setProgress(c.id, loaded / total));
      setProgress(c.id, 1);
      state.models.push(r);
      appendResult(c.id, r);
      setStatus(c.id, r.ok ? 'ok' : 'ng', r.ok ? `${backend} OK` : `${backend} NG`);
      dumpResults();
      // UI を描き直す隙を作る（長い同期処理でスマホが固まらないように）
      await new Promise((res) => setTimeout(res, 50));
    }
  }
}

// --- ③ 描画ベンチ -----------------------------------------------------------

/**
 * ベンチの設定。既定は設計の4段階（軽量／カリング後／標準／高品質）。
 *
 * URL パラメータで上書きできる。ソフトウェア実装の CI では実機の20〜50倍遅いので、
 * `?frames=8&counts=424000` のように減らして「壊れていないこと」だけを見る。
 */
const params = new URLSearchParams(location.search);
const BENCH_COUNTS = (params.get('counts') ?? '99000,240000,424000,549000')
  .split(',')
  .map((s) => Number.parseInt(s, 10))
  .filter((n) => Number.isFinite(n) && n > 0);
const BENCH_FRAMES = Math.max(4, Number.parseInt(params.get('frames') ?? '90', 10) || 90);

async function runRenderBench(): Promise<void> {
  const cap = state.capability ?? (await showCapability());
  const out = $('renderOut');
  if (!cap.webgpu.supported) {
    out.innerHTML = '<span class="pill ng">WebGPU が無いので実行できません</span>';
    return;
  }
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) {
    out.innerHTML = '<span class="pill ng">アダプタを取得できませんでした</span>';
    return;
  }
  const device = await adapter.requestDevice();
  device.lost.then((info) => {
    out.innerHTML += `<div><span class="pill ng">デバイス消失</span> ${esc(info.message)}</div>`;
  });

  const canvas = $<HTMLCanvasElement>('cv');
  const bar = $('renderBar').firstElementChild as HTMLElement;
  state.render = [];
  out.innerHTML = '';

  for (let i = 0; i < BENCH_COUNTS.length; i++) {
    const n = BENCH_COUNTS[i] as number;
    bar.style.width = `${((i + 0.5) / BENCH_COUNTS.length) * 100}%`;
    out.innerHTML += `<div class="k">${n.toLocaleString('ja-JP')} 個 … 実行中</div>`;
    let r: RenderBenchResult;
    try {
      r = await benchRender(device, canvas, n, BENCH_FRAMES);
    } catch (e) {
      out.innerHTML += `<div><span class="pill ng">失敗</span> ${esc(String(e))}</div>`;
      break;
    }
    state.render.push(r);
    const good = r.frameMs <= 16.6;
    out.lastElementChild?.remove();
    out.innerHTML +=
      `<div class="row" style="gap:8px">
         <span class="pill ${good ? 'ok' : 'ng'}">${good ? '60fps 可' : '60fps 不可'}</span>
         <span class="mono">${n.toLocaleString('ja-JP')} 個 — ${r.frameMs} ms / ${r.fps} fps
         （描画 ${r.drawnSplats.toLocaleString('ja-JP')}、カリング ${(r.cullRatio * 100).toFixed(0)}%）</span>
       </div>`;
    dumpResults();
    await new Promise((res) => setTimeout(res, 60));
  }
  bar.style.width = '100%';
  device.destroy();
}

// --- ④ 結果 -----------------------------------------------------------------

function summary(): Record<string, unknown> {
  const cap = state.capability;
  return {
    生成日時: new Date().toISOString(),
    UA: navigator.userAgent,
    画面: `${screen.width}×${screen.height} @${devicePixelRatio}x`,
    WebGPU: cap?.webgpu.supported
      ? {
          vendor: cap.webgpu.vendor ?? null,
          architecture: cap.webgpu.architecture ?? null,
          limits: cap.webgpu.limits ?? null,
          作業グリッド: cap.workingGrid,
        }
      : { 利用可否: false, 理由: cap?.webgpu.reason ?? '未判定' },
    crossOriginIsolated: cap?.crossOriginIsolated ?? null,
    モデル: state.models,
    描画ベンチ: state.render,
    判定: verdicts(),
  };
}

/** 3つのリスクに対する自動判定。人が読んで確かめられるよう根拠も添える。 */
function verdicts(): Record<string, string> {
  const v: Record<string, string> = {};
  const webgpuDepth = state.models.filter((m) => m.backend === 'webgpu' && m.id.startsWith('da'));
  if (webgpuDepth.length === 0) v['R1'] = '未検証';
  else v['R1'] = webgpuDepth.some((m) => m.ok)
    ? `OK — WebGPU で動いた深度モデル: ${webgpuDepth.filter((m) => m.ok).map((m) => m.id).join(', ')}`
    : `NG — WebGPU ではどの深度モデルも動かなかった（WASM 縮退の検討が必要）`;

  const da3 = state.models.find((m) => m.id === 'da3-small-fp32' && m.ok);
  if (!da3) v['R10'] = state.models.some((m) => m.id === 'da3-small-fp32') ? 'NG — DA3 が動かない。V2 へ切り替え' : '未検証';
  else {
    const outs = da3.outputs ?? [];
    const hasIntrinsics = outs.length > 1;
    v['R10'] = `OK — DA3 は動く。出力 ${outs.length} 個（${outs.join(', ')}）。` +
      (hasIntrinsics ? '内部パラメータの出力がある可能性が高い' : '深度のみ。焦点距離は EXIF → 画角55°仮定へ');
  }

  const std = state.render.find((r) => r.splatCount === 424_000);
  if (!std) v['R12'] = '未検証';
  else v['R12'] = std.frameMs <= 16.6
    ? `OK — 42万で ${std.frameMs}ms（${std.fps}fps）。1024² を既定にできる`
    : `NG — 42万で ${std.frameMs}ms（${std.fps}fps）。既定を 768² に落とすか LOD 前提にする`;
  return v;
}

function dumpResults(): void {
  $('out').textContent = JSON.stringify(summary(), null, 2);
}

function esc(s: string): string {
  return s.replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c] ?? c);
}

// --- 起動 -------------------------------------------------------------------

renderModelList();
void showCapability().then(dumpResults);

$('runDepth').addEventListener('click', () => void guard(() => runModels((c) => c.role === 'depth')));
$('runAll').addEventListener('click', () => void guard(() => runModels(() => true)));
$('runRender').addEventListener('click', () => void guard(runRenderBench));

$('copy').addEventListener('click', async () => {
  const text = JSON.stringify(summary(), null, 2);
  try {
    await navigator.clipboard.writeText(text);
    $('copy').textContent = 'コピーしました';
  } catch {
    // クリップボードが使えない環境では選択して手動コピーしてもらう
    const pre = $('out');
    const range = document.createRange();
    range.selectNodeContents(pre);
    getSelection()?.removeAllRanges();
    getSelection()?.addRange(range);
    $('copy').textContent = '選択しました。手動でコピーしてください';
  }
  setTimeout(() => ($('copy').textContent = '結果をコピー'), 2500);
});

$('clearCache').addEventListener('click', async () => {
  await clearModelCache();
  $('clearCache').textContent = '消しました';
  setTimeout(() => ($('clearCache').textContent = 'モデルキャッシュを消す'), 2000);
  void showCapability();
});

/** ボタンの二重押しを防ぎつつ、例外を画面に出す。 */
let busy = false;
async function guard(fn: () => Promise<void>): Promise<void> {
  if (busy) return;
  busy = true;
  const buttons = [...document.querySelectorAll('button')];
  buttons.forEach((b) => ((b as HTMLButtonElement).disabled = true));
  try {
    await fn();
  } catch (e) {
    $('out').textContent = `実行中にエラーが起きました:\n${String(e)}\n\n${JSON.stringify(summary(), null, 2)}`;
  } finally {
    buttons.forEach((b) => ((b as HTMLButtonElement).disabled = false));
    busy = false;
    dumpResults();
  }
}
