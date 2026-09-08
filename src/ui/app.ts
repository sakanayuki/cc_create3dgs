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
import { EXPORT_FORMATS, toSplatFile, type ExportFormat } from './export';
import { Viewer } from './viewer';

export type Preset = 'light' | 'standard' | 'high';

/** docs/04 §4.7 の品質プリセット。 */
interface PresetSpec {
  grid: number;
  reduction: number;
  inpaint: boolean;
  /** 深度のタイルパス。顔の立体感はほぼこれで決まる（docs/03 §3.4）。 */
  depthTiles: boolean;
}

// 表示名は index.html のラジオが持つ。ここは挙動だけ。
const PRESETS: Record<Preset, PresetSpec> = {
  // 軽量はタイルパスもインペイントも行わない（docs/04 §4.7）。速いが顔は平坦になる。
  light: { grid: LIGHT_GRID, reduction: 0.45, inpaint: false, depthTiles: false },
  standard: { grid: WORKING_GRID, reduction: 0.3, inpaint: true, depthTiles: true },
  high: { grid: WORKING_GRID, reduction: 0, inpaint: true, depthTiles: true },
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
      ? `この端末: WebGPU で推論・描画（作業グリッド ${cap.workingGrid}²` +
        `${cap.shaderF16 ? '' : '、shader-f16 が無いため uint8 モデルを使用'}）`
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
  // E2E から角度を変えて確かめるための入口。
  (window as unknown as Record<string, unknown>)['__viewer'] = viewer;
  return viewer;
}

/**
 * 推論バックエンドで一度失敗したら、WASM でやり直す（v2.6、実機で判明）。
 *
 * WebGPU の実装差で、あるモデルのシェーダが弾かれることがある。ORT の
 * WebGPU バックエンドはセッションをまたいで 1 つなので、そこで壊れると
 * **次のモデルが無関係な例外で落ちる**。実機では u2netp の fp16 が
 * 「'f16' type used without 'f16' extension enabled」で無効なシェーダを
 * 作り、その後の DA3 が
 *
 *     kernel "[Resize] /backbone/Resize" is not allowed to be called recursively
 *
 * で落ちた。落ちる場所と原因が離れるので、利用者には何も分からない。
 *
 * 原因側（f16 を要求しない変種を配る）は直したが、端末ごとの WebGPU の
 * 差を全部見切ることはできない。**遅くても結果が出る**ほうを選ぶ。
 * WASM 経路は同じモデルを uint8 で回すので、絵は出る。
 */
async function withInferenceFallback<T>(
  first: Backend,
  attempt: (backend: Backend) => Promise<T>,
): Promise<T> {
  try {
    return await attempt(first);
  } catch (e) {
    if (first === 'wasm') throw e;
    // 何が起きたかは残す。黙って遅くなると、原因を追う手がかりが消える。
    console.warn('[photosplat] WebGPU での推論に失敗したので WASM でやり直します', e);
    setProgress(0, 'WebGPU で失敗したため、互換モード（WASM）でやり直しています');
    return await attempt('wasm');
  }
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

    const result = await withInferenceFallback(inferenceBackend(cap), (backend) =>
      generate(file, {
      mode: currentMode(),
      grid: preset.grid,
      reduction: preset.reduction,
      inpaint: preset.inpaint,
      depthTiles: preset.depthTiles,
      backend,
      // shader-f16 が無い WebGPU では q4f16 のモデルが黙って壊れる。
      // その場合は uint8 側を落とす（modelCatalog の manifestBackendKey）。
      shaderF16: cap.shaderF16,
      onProgress: setProgress,
      // プレビューができた時点で先に見せる（docs/03 §3.1）。
      // インペイントを待たずに立体が出る。
      onPreview: (b) => {
        // プレビューは「まだ仕上げ中」の下書き。黙って出すと、統計が空のまま
        // 止まって見え、終わったのか壊れたのか分からない。
        show('view');
        $('refining').hidden = false;
        $('stats').innerHTML = '';
        viewer.resize();
        viewer.setSplats(b.data, b.count, b.nearZ, b.farZ);
      },
      }),
    );
    state.result = result;

    show('view');
    $('refining').hidden = true;
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
       }） / 奥行き÷幅 ${result.stats.depthToWidth.toFixed(2)}（人物は 0.9 前後）/ 奥行き÷高さ ${result.stats.depthToHeight.toFixed(2)}${
         result.stats.metricDepth ? '（実寸）' : '（比率指定）'
       } / 深度タイル ${result.stats.depthTiles} 枚 / 遮蔽部の補完: ${
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

function currentFormat(): ExportFormat {
  const el = document.querySelector<HTMLInputElement>('input[name="format"]:checked');
  return (el?.value as ExportFormat) ?? 'spz';
}

/** 選ばれている形式の説明を出す。何が違うのか分からないまま選ばせない。 */
function showFormatNote(): void {
  const info = EXPORT_FORMATS[currentFormat()];
  $('formatNote').textContent = info.note;
  $('save').textContent = `${info.label} で保存`;
}

async function download(): Promise<void> {
  const r = state.result;
  if (!r) return;
  const format = currentFormat();
  const info = EXPORT_FORMATS[format];
  const button = $<HTMLButtonElement>('save');
  const label = button.textContent;
  button.disabled = true;
  button.textContent = '書き出しています…';
  try {
    const blob = await toSplatFile(r.build.data, r.build.count, format);
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `photosplat-${Date.now()}.${info.extension}`;
    a.click();
    // revoke は次のタスクで。同期で消すと Safari でダウンロードが始まらない。
    setTimeout(() => URL.revokeObjectURL(url), 10_000);
  } finally {
    button.disabled = false;
    button.textContent = label;
  }
}

export function mountApp(): void {
  $('photo').addEventListener('change', (e) => {
    const file = (e.target as HTMLInputElement).files?.[0];
    if (file) void run(file);
  });
  $('again').addEventListener('click', () => {
    ($('photo') as HTMLInputElement).value = '';
    $('refining').hidden = true;
    show('pick');
  });
  $('retry').addEventListener('click', () => show('pick'));
  $('save').addEventListener('click', () => void download());
  for (const el of document.querySelectorAll('input[name="format"]')) {
    el.addEventListener('change', showFormatNote);
  }
  showFormatNote();
  void boot();
}
