/**
 * ONNX Runtime Web のセッション管理（docs/02 §2.5, docs/08 §8.2）。
 *
 * ・実行プロバイダは WebGPU を優先し、失敗したら WASM に落とす3段の縮退。
 * ・モデルは Cache API に貯めて2回目以降のダウンロードをゼロにする。
 *   GitHub Pages のキャッシュヘッダは制御できないので、明示的に自分で貯める。
 * ・WebGPU デバイスは自前のものを ORT に共有させ、推論出力を GPU に置いたまま
 *   WGSL カーネルへ渡せるようにする。
 */
import * as ort from 'onnxruntime-web';

export type Backend = 'webgpu' | 'wasm';

/** 縮退の段。docs/08 §8.2 の3段に対応する。 */
export interface Attempt {
  readonly backend: Backend;
  /** レジストリ上の量子化モード。ファイル名の一部になる。 */
  readonly quant: string;
}

export const DEFAULT_LADDER: readonly Attempt[] = [
  { backend: 'webgpu', quant: 'q4f16' },
  { backend: 'webgpu', quant: 'fp16' },
  { backend: 'wasm', quant: 'uint8' },
];

export interface ModelSource {
  /** 表示用の識別子。 */
  readonly id: string;
  /** モデル本体の URL。 */
  readonly url: string;
  /** 外部データ（model.onnx_data）がある場合の URL とファイル名。 */
  readonly externalData?: { readonly url: string; readonly path: string };
}

export interface LoadResult {
  readonly session: ort.InferenceSession;
  readonly backend: Backend;
  readonly quant: string;
  /** 取得〜セッション生成の内訳（ミリ秒）。ベンチで使う。 */
  readonly timing: { readonly fetchMs: number; readonly createMs: number };
  readonly bytes: number;
  readonly fromCache: boolean;
}

const CACHE_NAME = 'photosplat-models-v1';
let configured = false;

/** ORT のグローバル設定。一度だけ行う。 */
export function configureOrt(opts: { adapter?: GPUAdapter; wasmThreads?: number } = {}): void {
  if (configured) return;
  const base = import.meta.env.BASE_URL ?? '/';
  // 同一オリジンで配る（決定 D7）。scripts/copy_ort.mjs が public/ort/ に置く。
  ort.env.wasm.wasmPaths = `${base}ort/`;
  // GitHub Pages では SharedArrayBuffer が無いので単スレッド。
  ort.env.wasm.numThreads = opts.wasmThreads ?? 1;
  ort.env.wasm.simd = true;
  ort.env.logLevel = 'error';
  if (opts.adapter) {
    // 自前のアダプタを ORT に使わせる。これをしないと ORT が別デバイスを確保し、
    // 推論結果を取り出すたびに CPU 経由の往復が発生する（docs/02 §2.5）。
    (ort.env.webgpu as unknown as { adapter?: GPUAdapter }).adapter = opts.adapter;
  }
  configured = true;
}

/** ORT が確保／共有している WebGPU デバイス。WGSL カーネルと共有するために読む。 */
export function ortDevice(): GPUDevice | undefined {
  return (ort.env.webgpu as unknown as { device?: GPUDevice }).device;
}

async function cacheOpen(): Promise<Cache | null> {
  try {
    return await caches.open(CACHE_NAME);
  } catch {
    // プライベートウィンドウ等では caches が使えないことがある
    return null;
  }
}

/** Cache API を挟んでバイト列を取得する。 */
export async function fetchCached(
  url: string,
  onProgress?: (loaded: number, total: number) => void,
): Promise<{ buffer: ArrayBuffer; fromCache: boolean }> {
  const cache = await cacheOpen();
  if (cache) {
    const hit = await cache.match(url);
    if (hit) return { buffer: await hit.arrayBuffer(), fromCache: true };
  }

  const res = await fetch(url);
  if (!res.ok) throw new Error(`取得に失敗しました (${res.status}) ${url}`);

  let buffer: ArrayBuffer;
  const total = Number(res.headers.get('content-length') ?? 0);
  if (onProgress && res.body && total > 0) {
    const reader = res.body.getReader();
    const chunks: Uint8Array[] = [];
    let loaded = 0;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      loaded += value.byteLength;
      onProgress(loaded, total);
    }
    const merged = new Uint8Array(loaded);
    let off = 0;
    for (const c of chunks) {
      merged.set(c, off);
      off += c.byteLength;
    }
    buffer = merged.buffer;
    // Response は一度読むと再利用できないので、貯めるために作り直す
    if (cache) await cache.put(url, new Response(merged.slice(0))).catch(() => {});
  } else {
    const clone = cache ? res.clone() : null;
    buffer = await res.arrayBuffer();
    if (cache && clone) await cache.put(url, clone).catch(() => {});
  }
  return { buffer, fromCache: false };
}

/** 1つの構成でセッション生成を試みる。失敗したら例外を投げる。 */
export async function createSession(
  src: ModelSource,
  attempt: Attempt,
  onProgress?: (loaded: number, total: number) => void,
): Promise<LoadResult> {
  const t0 = performance.now();
  const main = await fetchCached(src.url, onProgress);
  let bytes = main.buffer.byteLength;

  const options: ort.InferenceSession.SessionOptions = {
    executionProviders: [attempt.backend],
    graphOptimizationLevel: 'all',
  };

  if (src.externalData) {
    const ext = await fetchCached(src.externalData.url);
    bytes += ext.buffer.byteLength;
    // 外部データを伴うモデル（DA3-S の model.onnx_data など）
    (options as unknown as Record<string, unknown>)['externalData'] = [
      { path: src.externalData.path, data: new Uint8Array(ext.buffer) },
    ];
  }

  const t1 = performance.now();
  const session = await ort.InferenceSession.create(new Uint8Array(main.buffer), options);
  const t2 = performance.now();

  return {
    session,
    backend: attempt.backend,
    quant: attempt.quant,
    timing: { fetchMs: t1 - t0, createMs: t2 - t1 },
    bytes,
    fromCache: main.fromCache,
  };
}

/** 縮退の段を順に試し、最初に成功したものを返す。 */
export async function loadWithLadder(
  resolve: (attempt: Attempt) => ModelSource | null,
  ladder: readonly Attempt[] = DEFAULT_LADDER,
  onStep?: (attempt: Attempt, error?: unknown) => void,
): Promise<LoadResult> {
  const errors: string[] = [];
  for (const attempt of ladder) {
    const src = resolve(attempt);
    if (!src) continue;
    try {
      onStep?.(attempt);
      return await createSession(src, attempt);
    } catch (e) {
      errors.push(`${attempt.backend}/${attempt.quant}: ${String(e)}`);
      onStep?.(attempt, e);
    }
  }
  throw new Error(`どの構成でもセッションを作れませんでした:\n${errors.join('\n')}`);
}

/** キャッシュ済みモデルの合計サイズ。UI で「準備済み」を示すのに使う。 */
export async function cachedBytes(): Promise<number> {
  const cache = await cacheOpen();
  if (!cache) return 0;
  let total = 0;
  for (const req of await cache.keys()) {
    const res = await cache.match(req);
    if (res) total += (await res.arrayBuffer()).byteLength;
  }
  return total;
}

export async function clearModelCache(): Promise<void> {
  try {
    await caches.delete(CACHE_NAME);
  } catch {
    /* 消せなくても致命的ではない */
  }
}
