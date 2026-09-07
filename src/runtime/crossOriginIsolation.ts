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

/**
 * 既定のスレッド数（決定 D20）。
 *
 * PoC-2 の実機計測（Android 10、論理コア 8）で決めた値。
 *
 *   1 スレッド … 11,556 ms （基準）
 *   2 スレッド …  6,655 ms （1.74×）
 *   4 スレッド …  6,192 ms （1.87×）
 *
 * 論理コアは 8 あるがスケーリングは **2 スレッドで頭打ち**だった。
 * 大コアが 2 つの big.LITTLE 構成と見られる。2 → 4 で得られるのは 7% だけで、
 * その一方で 4 スレッドの計測では初回推論（5,714 ms）が定常推論（6,192 ms）より
 * 速いという逆転が起きており、発熱による性能低下を示唆していた。
 *
 * 連続生成での安定性と電池を考えて 2 を既定にする。
 * 開発コンテナ（4 コア均質）では 4 スレッドで 3〜4× 出たので、
 * 据え置き機では上げる余地がある。将来は起動時ベンチで決めてもよい。
 */
export const PREFERRED_THREADS = 2;

function threadsAvailable(): number {
  if (typeof crossOriginIsolated === 'undefined' || !crossOriginIsolated) return 1;
  if (typeof SharedArrayBuffer === 'undefined') return 1;
  return Math.max(1, Math.min(PREFERRED_THREADS, navigator.hardwareConcurrency || 1));
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

    // 既に制御下にあるのに isolation が無いなら、リロードしても状況は変わらない
    if (registration.active && navigator.serviceWorker.controller) {
      return {
        state: 'failed',
        crossOriginIsolated: false,
        maxThreads: 1,
        detail: 'Service Worker は動作していますが isolation が成立していません',
      };
    }

    // register() は install の完了前に解決する。ここで即リロードすると、
    // まだ Service Worker がページを制御しておらず fetch が横取りされないため、
    // COOP/COEP が付かず isolation が成立しない。
    // 「有効化」と「このページの制御を取る」の両方を待ってからリロードする。
    await waitUntilControlling();

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

/**
 * Service Worker が有効化され、このページを制御するようになるまで待つ。
 *
 * `clients.claim()` の効果が届くのを待つ。届かない環境もありうるので、
 * 一定時間で諦めてリロードに進む（リロード後に制御が付くこともある）。
 */
async function waitUntilControlling(timeoutMs = 3000): Promise<void> {
  await navigator.serviceWorker.ready;
  if (navigator.serviceWorker.controller) return;
  await new Promise<void>((resolve) => {
    const done = () => {
      clearTimeout(timer);
      navigator.serviceWorker.removeEventListener('controllerchange', done);
      resolve();
    };
    const timer = setTimeout(done, timeoutMs);
    navigator.serviceWorker.addEventListener('controllerchange', done);
  });
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
