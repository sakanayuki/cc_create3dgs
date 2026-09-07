/**
 * 本体アプリのエントリ。
 *
 * 生成パイプラインは実装中なので、いまは次の3つを行う。
 *   1. マルチスレッド化の下ごしらえ（決定 D20）
 *   2. 端末の能力判定と、どちらの経路で動くかの表示
 *   3. PoC への導線
 */
import { detectCapability, describeCapability, type Capability } from './runtime/capability';
import {
  PREFERRED_THREADS,
  ensureCrossOriginIsolation,
  isolationSummary,
} from './runtime/crossOriginIsolation';
import { Webgl2SplatRenderer } from './render/backends/webgl2';

/** どの経路で動くか。UI と、この先のパイプラインの分岐に使う。 */
export type Backend = 'webgpu' | 'webgl2' | 'unsupported';

function chooseBackend(cap: Capability): Backend {
  if (cap.webgpu.supported) return 'webgpu';
  if (Webgl2SplatRenderer.isSupported()) return 'webgl2';
  return 'unsupported';
}

function describeBackend(backend: Backend, cap: Capability): string {
  const iso = isolationSummary();
  switch (backend) {
    case 'webgpu':
      return (
        `この端末: WebGPU 経路 / 作業グリッド ${cap.workingGrid}²` +
        (cap.webgpu.vendor ? ` / ${cap.webgpu.vendor}` : '')
      );
    case 'webgl2':
      return (
        `この端末: WebGL2 経路（WebGPU 非対応）/ 推論は WASM ` +
        `${iso.maxThreads} スレッド / 論理コア ${iso.hardwareConcurrency}`
      );
    default:
      return 'この端末: WebGL2 も WebGPU も使えないため生成できません';
  }
}

async function main(): Promise<void> {
  const card = document.querySelector('.card.accent');
  const note = document.createElement('p');
  note.style.marginTop = '10px';
  note.style.fontSize = '12.5px';
  note.className = 'mono';
  card?.appendChild(note);

  // 先にマルチスレッド化を成立させる（決定 D20）。
  // 未成立なら Service Worker を登録して1度だけリロードする。重い処理の前に済ませる。
  const before = isolationSummary();
  if (!before.crossOriginIsolated) {
    note.textContent = 'マルチスレッドの準備中…';
    const result = await ensureCrossOriginIsolation();
    if (result.state === 'reloading') return; // リロードされるのでここで終わり
    if (result.state !== 'isolated') {
      // 成立しなくても単スレッドで動く。遅くなるだけなので止めない。
      console.warn('[PhotoSplat] マルチスレッド化できませんでした:', result.detail);
    }
  }

  const cap = await detectCapability();
  const backend = chooseBackend(cap);
  note.textContent = describeBackend(backend, cap);

  const iso = isolationSummary();
  if (backend !== 'webgpu' && iso.maxThreads < PREFERRED_THREADS) {
    // 実機で 1 スレッドだと人物でも 17.7 秒かかる。黙って遅くせず理由を出す。
    const warn = document.createElement('p');
    warn.style.marginTop = '6px';
    warn.style.fontSize = '12.5px';
    warn.textContent =
      'マルチスレッドが使えないため、生成に時間がかかります（人物で 18 秒前後）。';
    card?.appendChild(warn);
  }

  // 詳細はコンソールに出す。実機でのデバッグ用。
  console.info(
    '[PhotoSplat] 端末の能力\n' +
      describeCapability(cap).join('\n') +
      `\n選択した経路: ${backend}` +
      `\nWASM スレッド数: ${iso.maxThreads}（既定 ${PREFERRED_THREADS}）`,
  );
}

void main();
