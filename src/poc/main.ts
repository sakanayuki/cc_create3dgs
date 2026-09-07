/**
 * PoC-1 のページ制御（docs/08 §8.11）。
 *
 * 実機で開いて実行し、結果をコピーして共有してもらう診断ツール。
 * 本体アプリとは独立に配信する（poc.html）。
 */
import {
  CANDIDATES,
  benchRender,
  benchThreads,
  syntheticThreadTarget,
  testModel,
  type Candidate,
  type ModelResult,
  type RenderBenchResult,
  type ThreadTarget,
} from './bench';
import type { ThreadTestResponse } from './threadWorker';
import { createRenderer } from '../render/createRenderer';
import { detectCapability, describeCapability, type Capability } from '../runtime/capability';
import {
  disableCrossOriginIsolation,
  ensureCrossOriginIsolation,
  isolationSummary,
} from '../runtime/crossOriginIsolation';
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
  threads: ThreadTestResponse[];
  threadTarget?: string;
} = { models: [], render: [], threads: [] };

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

// --- ③ スレッド数の効果（PoC-2） ---------------------------------------------

/** 実モデルでのスレッド計測。PoC-1 で WASM 最速だった uint8 版を使う。 */
const THREAD_MODEL_ID = 'da2-small-uint8';

function showIsolation(): void {
  const s = isolationSummary();
  const pill = $('coiState');
  const detail = $('coiDetail');
  if (s.crossOriginIsolated) {
    pill.className = 'pill ok';
    pill.textContent = 'マルチスレッド可';
    detail.textContent = `最大 ${s.maxThreads} スレッド（論理コア ${s.hardwareConcurrency}）`;
  } else {
    pill.className = 'pill ng';
    pill.textContent = '単スレッド';
    detail.textContent = `crossOriginIsolated = false（論理コア ${s.hardwareConcurrency}）`;
  }
}

async function enableCoi(): Promise<void> {
  const r = await ensureCrossOriginIsolation(false);
  $('coiDetail').textContent = r.detail ?? '';
  if (r.state === 'reloading') {
    $('coiState').className = 'pill run';
    $('coiState').textContent = '再読み込みします…';
    // ensureCrossOriginIsolation が「制御を取るまで」待ってから返しているので、
    // ここは表示を見せるためだけの短い間を置く
    setTimeout(() => location.reload(), 200);
  } else {
    showIsolation();
  }
}

async function runThreadBench(useSynthetic: boolean): Promise<void> {
  const out = $('threadOut');
  let target: ThreadTarget | undefined;
  if (useSynthetic) {
    target = syntheticThreadTarget();
  } else {
    const c = CANDIDATES.find((x) => x.id === THREAD_MODEL_ID);
    if (c) target = { label: c.label, source: c.source, inputSize: c.inputSize, approxMB: c.approxMB };
  }
  if (!target) {
    out.textContent = 'スレッド計測の対象が見つかりません';
    return;
  }
  const candidate = target;
  const bar = $('threadBar').firstElementChild as HTMLElement;
  const iso = isolationSummary();
  const counts = iso.crossOriginIsolated ? [1, 2, 4] : [1];

  out.innerHTML = `<div class="k">対象: ${esc(candidate.label)}</div>`;
  if (!iso.crossOriginIsolated) {
    out.innerHTML +=
      '<div><span class="pill ng">単スレッドのみ</span> ' +
      'まず「マルチスレッドを有効にする」を押してください。比較のため1スレッドだけ測ります。</div>';
  }

  state.threads = [];
  state.threadTarget = candidate.label;
  for (let i = 0; i < counts.length; i++) {
    const n = counts[i] as number;
    bar.style.width = `${((i + 0.5) / counts.length) * 100}%`;
    out.innerHTML += `<div class="k" data-pending="${n}">${n} スレッド … 実行中</div>`;
    const [res] = await benchThreads(candidate, [n]);
    document.querySelector(`[data-pending="${n}"]`)?.remove();
    if (!res) continue;
    state.threads.push(res);

    const base = state.threads.find((r) => r.numThreads === 1 && r.ok)?.inferMs;
    const speedup = base && res.ok && res.inferMs ? ` — ${(base / res.inferMs).toFixed(2)}× 速い` : '';
    out.innerHTML += res.ok
      ? `<div class="row" style="gap:8px">
           <span class="pill ok">${n} スレッド</span>
           <span class="mono">推論 ${res.inferMs} ms（生成 ${res.createMs} ms）${speedup}</span>
         </div>`
      : `<div><span class="pill ng">${n} スレッド 失敗</span> <span class="mono">${esc(res.error ?? '')}</span></div>`;
    dumpResults();
    await new Promise((r) => setTimeout(r, 60));
  }
  bar.style.width = '100%';

  const ok = state.threads.filter((r) => r.ok && r.inferMs);
  if (ok.length >= 2) {
    const one = ok.find((r) => r.numThreads === 1)?.inferMs;
    const best = ok.reduce((a, b) => ((b.inferMs ?? 0) < (a.inferMs ?? Infinity) ? b : a));
    if (one && best.inferMs) {
      const gain = one / best.inferMs;
      out.innerHTML +=
        `<div class="row" style="gap:8px;margin-top:8px">
           <span class="pill ${gain >= 2 ? 'ok' : 'run'}">結論</span>
           <span class="mono">最良 ${best.numThreads} スレッドで <b>${gain.toFixed(2)}×</b>` +
        `（${one} → ${best.inferMs} ms）</span>
         </div>`;
    }
  }
}

// --- ④ 描画ベンチ -----------------------------------------------------------

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
  const out = $('renderOut');
  const canvas = $<HTMLCanvasElement>('cv');
  const bar = $('renderBar').firstElementChild as HTMLElement;
  state.render = [];
  out.innerHTML = '';

  // WebGPU が無くても WebGL2 で測れる（決定 D19）。どちらで測ったかは結果に残る。
  let backendLabel: string;
  try {
    const probe = await createRenderer({ canvas });
    backendLabel = probe.backend;
    if (probe.fallbackReason) {
      out.innerHTML =
        `<div class="k">WebGPU が使えないため WebGL2 で測ります（${esc(probe.fallbackReason)}）</div>`;
    } else {
      out.innerHTML = `<div class="k">${probe.backend} で測ります</div>`;
    }
    probe.renderer.dispose();
  } catch (e) {
    out.innerHTML = `<span class="pill ng">描画できません</span> <span class="mono">${esc(String(e))}</span>`;
    return;
  }

  // WebGL2 は WebGPU より遅いので、60fps ではなく 30fps を目標にする（docs/01 §1.4）
  const targetMs = backendLabel === 'webgpu' ? 16.6 : 33.3;
  const targetLabel = backendLabel === 'webgpu' ? '60fps' : '30fps';

  for (let i = 0; i < BENCH_COUNTS.length; i++) {
    const n = BENCH_COUNTS[i] as number;
    bar.style.width = `${((i + 0.5) / BENCH_COUNTS.length) * 100}%`;
    out.innerHTML += `<div class="k" data-running="${n}">${n.toLocaleString('ja-JP')} 個 … 実行中</div>`;
    let r: RenderBenchResult;
    try {
      r = await benchRender(canvas, n, BENCH_FRAMES);
    } catch (e) {
      out.innerHTML += `<div><span class="pill ng">失敗</span> ${esc(String(e))}</div>`;
      break;
    }
    state.render.push(r);
    const good = r.frameMs <= targetMs;
    document.querySelector(`[data-running="${n}"]`)?.remove();
    out.innerHTML +=
      `<div class="row" style="gap:8px">
         <span class="pill ${good ? 'ok' : 'ng'}">${good ? `${targetLabel} 可` : `${targetLabel} 不可`}</span>
         <span class="mono">${n.toLocaleString('ja-JP')} 個 — ${r.frameMs} ms / ${r.fps} fps
         （${r.backend}、描画 ${r.drawnSplats.toLocaleString('ja-JP')}、カリング ${(r.cullRatio * 100).toFixed(0)}%）</span>
       </div>`;
    dumpResults();
    await new Promise((res) => setTimeout(res, 60));
  }
  bar.style.width = '100%';
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
    スレッド計測: {
      crossOriginIsolated: isolationSummary().crossOriginIsolated,
      論理コア: isolationSummary().hardwareConcurrency,
      対象: state.threadTarget ?? null,
      結果: state.threads,
    },
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

  // D11（マルチスレッド化）の判定。
  //
  // 「何倍速いか」ではなく **SLO に届くかどうか** で判断する。
  // 倍率に閾値を置くのは恣意的で、実際に誤解を招く判定を出した。
  // 人物・軽量プリセットの内訳（PoC-1 実測）:
  //   マット 4.1s + 深度 11.6s + 幾何 2.0s（幾何は JS なのでスレッド化の対象外）
  const MATTE_S = 4.1;
  const DEPTH_S = 11.6;
  const GEOMETRY_S = 2.0; // ORT のスレッドでは速くならない
  const LIGHT_SLO_S = 10.0;

  const ok = state.threads.filter((t) => t.ok && t.inferMs);
  const one = ok.find((t) => t.numThreads === 1)?.inferMs;
  const best = ok.length ? ok.reduce((a, b) => ((b.inferMs ?? 0) < (a.inferMs ?? Infinity) ? b : a)) : null;

  if (!one || !best?.inferMs || ok.length < 2) {
    v['D11'] = isolationSummary().crossOriginIsolated ? '未検証' : '未検証（マルチスレッド未有効）';
  } else {
    const gain = one / best.inferMs;
    const single = MATTE_S + DEPTH_S + GEOMETRY_S;
    const multi = (MATTE_S + DEPTH_S) / gain + GEOMETRY_S;
    const fmt = (n: number) => `${n.toFixed(1)}s`;
    const detail =
      `${best.numThreads} スレッドで ${gain.toFixed(2)}×（${one} → ${best.inferMs} ms）。` +
      `人物・軽量プリセットの見込み: 単スレッド ${fmt(single)} → ${fmt(multi)}`;

    if (multi <= LIGHT_SLO_S) {
      v['D11'] = `導入する — これで軽量プリセットが SLO ${LIGHT_SLO_S}秒に収まる。${detail}`;
    } else if (single > LIGHT_SLO_S) {
      // 単スレッドでは論外なので、届かなくても導入は必要。足りない分は別の手当てが要る。
      v['D11'] =
        `導入は必要だが、それだけでは足りない — SLO ${LIGHT_SLO_S}秒に対し ${fmt(multi)}。` +
        `入力解像度を下げるなど別のレバーが要る。${detail}`;
    } else {
      v['D11'] = `導入しなくても SLO に収まる — Service Worker のコストに見合わない。${detail}`;
    }
  }

  const std = state.render.find((r) => r.splatCount === 424_000);
  if (!std) {
    v['R12'] = '未検証';
  } else {
    // WebGPU は 60fps、WebGL2 は 30fps を目標にする（docs/01 §1.4）
    const target = std.backend === 'webgpu' ? 16.6 : 33.3;
    const label = std.backend === 'webgpu' ? '60fps' : '30fps';
    v['R12'] = std.frameMs <= target
      ? `OK（${std.backend}）— 42万で ${std.frameMs}ms / ${std.fps}fps。${label} 目標を満たし 1024² を既定にできる`
      : `NG（${std.backend}）— 42万で ${std.frameMs}ms / ${std.fps}fps。${label} 目標に届かないので既定を 768² に落とすか LOD 前提にする`;
  }
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
showIsolation();
void showCapability().then(dumpResults);

$('runDepth').addEventListener('click', () => void guard(() => runModels((c) => c.role === 'depth')));
$('runAll').addEventListener('click', () => void guard(() => runModels(() => true)));
$('runRender').addEventListener('click', () => void guard(runRenderBench));
$('enableCoi').addEventListener('click', () => void guard(enableCoi));
$('runThreadsQuick').addEventListener('click', () => void guard(() => runThreadBench(true)));
$('runThreads').addEventListener('click', () => void guard(() => runThreadBench(false)));
$('disableCoi').addEventListener('click', async () => {
  await disableCrossOriginIsolation();
  $('disableCoi').textContent = '解除しました。再読み込みしてください';
});

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
