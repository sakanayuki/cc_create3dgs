/**
 * `crossOriginIsolated` を Service Worker で成立させる（決定 D11 の再検討）。
 *
 * GitHub Pages は COOP/COEP を返せないので、そのままでは SharedArrayBuffer が
 * 使えず ONNX Runtime の WASM が単スレッドに制限される。PoC-1 の実測では
 * それが人物 30 秒・物体 63 秒という所要時間の主因だった。
 *
 * 設計 v2 は「WebGPU があれば WASM マルチスレッドは不要」としてこれを却下していたが、
 * WebGPU が使えない端末（Android 11 以下など）が現に存在するため前提が崩れた。
 */

export type IsolationState =
  | 'isolated' // すでに crossOriginIsolated
  | 'reloading' // Service Worker を登録したのでページを再読み込みする
  | 'unsupported' // Service Worker が使えない（プライベートウィンドウ等）
  | 'failed'; // 登録はできたが isolation が成立しなかった

export interface IsolationResult {
  readonly state: IsolationState;
  readonly crossOriginIsolated: boolean;
  /** 実際に使えるスレッド数の上限。 */
  readonly maxThreads: number;
  readonly detail?: string;
}

const SW_PATH = 'coi-sw.js';
/** 一度リロードしたことを覚えておく。無限リロードを防ぐ。 */
const RELOAD_FLAG = 'photosplat.coi.reloaded';

function threadsAvailable(): number {
  if (typeof crossOriginIsolated === 'undefined' || !crossOriginIsolated) return 1;
  if (typeof SharedArrayBuffer === 'undefined') return 1;
  // 物理コア数を超えても速くならない。4 で頭打ちにする（モバイルの大コアは概ね4）。
  return Math.max(1, Math.min(4, navigator.hardwareConcurrency || 1));
}

/**
 * isolation を試みる。成立していなければ Service Worker を登録して1度だけリロードする。
 *
 * @param autoReload false にすると登録だけ行い、リロードは呼び出し側に任せる。
 */
export async function ensureCrossOriginIsolation(autoReload = true): Promise<IsolationResult> {
  if (typeof crossOriginIsolated !== 'undefined' && crossOriginIsolated) {
    return { state: 'isolated', crossOriginIsolated: true, maxThreads: threadsAvailable() };
  }

  if (!('serviceWorker' in navigator)) {
    return {
      state: 'unsupported',
      crossOriginIsolated: false,
      maxThreads: 1,
      detail: 'この環境では Service Worker が使えません（プライベートウィンドウなど）',
    };
  }

  // 2回目以降は諦める。リロードループになるより単スレッドで動くほうがよい。
  if (sessionStorage.getItem(RELOAD_FLAG) === '1') {
    return {
      state: 'failed',
      crossOriginIsolated: false,
      maxThreads: 1,
      detail: 'Service Worker を登録してもリロード後に isolation が成立しませんでした',
    };
  }

  try {
    const base = import.meta.env.BASE_URL ?? '/';
    const registration = await navigator.serviceWorker.register(`${base}${SW_PATH}`, { scope: base });
    // 既に制御下にあるなら、リロードしても状況は変わらない
    if (registration.active && navigator.serviceWorker.controller) {
      return {
        state: 'failed',
        crossOriginIsolated: false,
        maxThreads: 1,
        detail: 'Service Worker は動作していますが isolation が成立していません',
      };
    }
    sessionStorage.setItem(RELOAD_FLAG, '1');
    if (autoReload) location.reload();
    return {
      state: 'reloading',
      crossOriginIsolated: false,
      maxThreads: 1,
      detail: 'Service Worker を登録しました。ページを再読み込みすると有効になります',
    };
  } catch (e) {
    return {
      state: 'unsupported',
      crossOriginIsolated: false,
      maxThreads: 1,
      detail: `Service Worker の登録に失敗しました: ${String(e)}`,
    };
  }
}

/** 登録済みの Service Worker を解除する。デバッグ用。 */
export async function disableCrossOriginIsolation(): Promise<void> {
  sessionStorage.removeItem(RELOAD_FLAG);
  if (!('serviceWorker' in navigator)) return;
  const regs = await navigator.serviceWorker.getRegistrations();
  await Promise.all(regs.map((r) => r.unregister()));
}

export function isolationSummary(): { crossOriginIsolated: boolean; maxThreads: number; hardwareConcurrency: number } {
  return {
    crossOriginIsolated: typeof crossOriginIsolated !== 'undefined' && crossOriginIsolated,
    maxThreads: threadsAvailable(),
    hardwareConcurrency: navigator.hardwareConcurrency || 0,
  };
}
