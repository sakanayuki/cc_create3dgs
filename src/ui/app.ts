/**
 * 本体アプリの画面制御（docs/02 §2, docs/03 §3.10）。
 *
 * 画面は3つの状態しか持たない。
 *   選ぶ → つくっている → 見る
 * 状態を増やすほど、途中で失敗したときの戻り先が分からなくなる。
 */
import { generate, type GenerateResult, type SubjectMode } from '../pipeline/generate';
import { LIGHT_GRID, WORKING_GRID } from '../pipeline/0-preprocess';
import { detectCapability, type Capability } from '../runtime/capability';
import { configureOrt, ortDevice, type Backend } from '../runtime/OrtSession';
import { isolationSummary } from '../runtime/crossOriginIsolation';
import { toSpzFile } from './export';
import { Viewer } from './viewer';

export type Preset = 'light' | 'standard' | 'high';

/** docs/04 §4.7 の品質プリセット。 */
const PRESETS: Record<Preset, { grid: number; reduction: number; inpaint: boolean; label: string }> = {
  // 軽量はインペイントを行わない（docs/04 §4.7）。奥側の色を伸ばすだけになる。
  light: { grid: LIGHT_GRID, reduction: 0.45, inpaint: false, label: '軽量' },
  standard: { grid: WORKING_GRID, reduction: 0.3, inpaint: true, label: '標準' },
  high: { grid: WORKING_GRID, reduction: 0, inpaint: true, label: '高品質' },
};

const $ = <T extends HTMLElement>(id: string): T => {
  const el = document.getElementById(id);
  if (!el) throw new Error(`要素がありません: ${id}`);
  return el as T;
};

const esc = (s: string): string =>
  s.replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c] ?? c);

interface State {
  capability?: Capability;
  viewer?: Viewer;
  result?: GenerateResult;
  busy: boolean;
}

const state: State = { busy: false };

function show(screen: 'pick' | 'work' | 'view'): void {
  for (const id of ['pick', 'work', 'view']) {
    $(id).hidden = id !== screen;
  }
}

/** 推論の実行プロバイダ。WebGPU が使えるならそちら（決定 D19）。 */
function inferenceBackend(cap: Capability): Backend {
  return cap.webgpu.supported ? 'webgpu' : 'wasm';
}

async function boot(): Promise<void> {
  const cap = await detectCapability();
  state.capability = cap;

  const iso = isolationSummary();
  configureOrt({ wasmThreads: iso.maxThreads });

  const backend = inferenceBackend(cap);
  $('env').textContent =
    backend === 'webgpu'
      ? `この端末: WebGPU で推論・描画（作業グリッド ${cap.workingGrid}²）`
      : `この端末: WASM ${iso.maxThreads} スレッドで推論、WebGL2 で描画`;

  if (!cap.webgpu.supported && !('WebGL2RenderingContext' in window)) {
    $('unsupported').hidden = false;
    $('photo').setAttribute('disabled', 'true');
  }
}

function currentPreset(): Preset {
  const el = document.querySelector<HTMLInputElement>('input[name="preset"]:checked');
  return (el?.value as Preset) ?? 'standard';
}

function currentMode(): SubjectMode {
  const el = document.querySelector<HTMLInputElement>('input[name="mode"]:checked');
  return (el?.value as SubjectMode) ?? 'person';
}

function setProgress(fraction: number, label: string): void {
  ($('bar').firstElementChild as HTMLElement).style.width = `${Math.round(fraction * 100)}%`;
  $('stage').textContent = label;
}

/** ビューアは1つだけ作って使い回す。作り直すと GPU の資源を無駄に取り直す。 */
async function ensureViewer(): Promise<Viewer> {
  if (state.viewer) return state.viewer;
  const viewer = new Viewer({
    canvas: $<HTMLCanvasElement>('cv'),
    onBackend: (b, why) => {
      $('renderInfo').textContent = why ? `${b}（WebGPU 不可: ${why}）` : b;
    },
  });
  await viewer.init();
  state.viewer = viewer;
  return viewer;
}

async function run(file: File): Promise<void> {
  if (state.busy) return;
  state.busy = true;

  try {
    // 画面の切り替えも進捗の表示も、失敗しうる DOM 操作である。
    // try の外に置くと、そこで落ちたときに握るものが無くなり、
    // 「押したのに何も起きない」になる（実際に #bar の id 抜けでそうなった）。
    show('work');
    setProgress(0, '準備しています');
    $('workError').hidden = true;

    const cap = state.capability ?? (await detectCapability());
    const preset = PRESETS[currentPreset()];

    // ビューアは生成を始める前に用意する。プレビューが出来た瞬間に
    // 表示したいので、ここで作っておかないと最初の1枚を捨てることになる。
    const viewer = await ensureViewer();

    const result = await generate(file, {
      mode: currentMode(),
      grid: preset.grid,
      reduction: preset.reduction,
      inpaint: preset.inpaint,
      backend: inferenceBackend(cap),
      onProgress: setProgress,
      // プレビューができた時点で先に見せる（docs/03 §3.1）。
      // インペイントを待たずに立体が出る。
      onPreview: (b) => {
        show('view');
        viewer.resize();
        viewer.setSplats(b.data, b.count, b.nearZ, b.farZ);
      },
    });
    state.result = result;

    show('view');
    viewer.resize();
    viewer.setSplats(result.build.data, result.build.count, result.build.nearZ, result.build.farZ);

    const t = result.stats.timings;
    const total = Object.values(t).reduce((a, b) => a + b, 0);
    $('stats').innerHTML =
      `<div class="grid">
         <div class="stat"><span class="k">ガウシアン</span><span class="v">${result.build.count.toLocaleString('ja-JP')}</span></div>
         <div class="stat"><span class="k">前面 / 背面 / スカート</span><span class="v">${result.build.frontCount.toLocaleString('ja-JP')} / ${result.build.backCount.toLocaleString('ja-JP')} / ${result.build.skirtCount.toLocaleString('ja-JP')}</span></div>
         <div class="stat"><span class="k">統合率</span><span class="v">${(result.stats.reduction * 100).toFixed(0)}%</span></div>
         <div class="stat"><span class="k">生成時間</span><span class="v">${(total / 1000).toFixed(1)} s</span></div>
       </div>
       <p class="note">焦点距離 ${result.stats.focalPx.toFixed(0)} px（${
         result.stats.intrinsicsFromModel ? 'モデルの推定値' : '画角 55° の仮定'
       }） / 遮蔽部の補完: ${
         { 'mi-gan': 'MI-GAN', stretch: '引き伸ばし（縮退）', skipped: '不要' }[result.stats.inpaint]
       }</p>
       <details><summary>工程ごとの時間</summary><pre class="mono">${esc(
         Object.entries(t)
           .map(([k, v]) => `${k.padEnd(16, '　')} ${String(v).padStart(6)} ms`)
           .join('\n'),
       )}</pre></details>`;
  } catch (e) {
    show('work');
    $('workError').hidden = false;
    $('workError').innerHTML =
      `<strong>生成できませんでした。</strong><br><span class="mono">${esc(String(e))}</span>`;
  } finally {
    state.busy = false;
  }
}

async function download(): Promise<void> {
  const r = state.result;
  if (!r) return;
  const blob = await toSpzFile(r.build.data, r.build.count);
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `photosplat-${Date.now()}.spz`;
  a.click();
  // revoke は次のタスクで。同期で消すと Safari でダウンロードが始まらない。
  setTimeout(() => URL.revokeObjectURL(url), 10_000);
}

export function mountApp(): void {
  $('photo').addEventListener('change', (e) => {
    const file = (e.target as HTMLInputElement).files?.[0];
    if (file) void run(file);
  });
  $('again').addEventListener('click', () => {
    ($('photo') as HTMLInputElement).value = '';
    show('pick');
  });
  $('retry').addEventListener('click', () => show('pick'));
  $('save').addEventListener('click', () => void download());
  void boot();
}
