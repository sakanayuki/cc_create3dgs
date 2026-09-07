/**
 * スレッド数を変えて推論時間を測るワーカー（PoC-2）。
 *
 * ONNX Runtime は `env.wasm.numThreads` を **wasm の初期化時に一度だけ**読む。
 * 一度セッションを作ると以後の変更は効かないので、スレッド数ごとに
 * 新しいワーカー（＝新しいモジュールインスタンス）で測る必要がある。
 */
import * as ort from 'onnxruntime-web';

export interface ThreadTestRequest {
  readonly modelUrl: string;
  readonly externalDataUrl?: string;
  readonly externalDataPath?: string;
  readonly numThreads: number;
  readonly inputSize: readonly [number, number];
  readonly runs: number;
}

export interface ThreadTestResponse {
  readonly numThreads: number;
  readonly ok: boolean;
  readonly error?: string;
  /** 実際に isolation が成立していたか。false なら要求しても単スレッドで動く。 */
  readonly crossOriginIsolated: boolean;
  readonly createMs?: number;
  readonly warmupMs?: number;
  /** ウォームアップ後の推論時間の中央値。 */
  readonly inferMs?: number;
}

const CACHE_NAME = 'photosplat-models-v1';

async function fetchCached(url: string): Promise<ArrayBuffer> {
  try {
    const cache = await caches.open(CACHE_NAME);
    const hit = await cache.match(url);
    if (hit) return await hit.arrayBuffer();
    const res = await fetch(url);
    if (!res.ok) throw new Error(`取得に失敗しました (${res.status})`);
    const clone = res.clone();
    const buf = await res.arrayBuffer();
    await cache.put(url, clone).catch(() => {});
    return buf;
  } catch {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`取得に失敗しました (${res.status})`);
    return await res.arrayBuffer();
  }
}

const median = (a: number[]): number => {
  const s = [...a].sort((x, y) => x - y);
  return s[Math.floor(s.length / 2)] ?? 0;
};

self.onmessage = async (event: MessageEvent<ThreadTestRequest>) => {
  const req = event.data;
  const isolated = typeof crossOriginIsolated !== 'undefined' && crossOriginIsolated;

  const reply = (r: ThreadTestResponse) => self.postMessage(r);

  try {
    const base = import.meta.env.BASE_URL ?? '/';
    ort.env.wasm.wasmPaths = `${base}ort/`;
    // isolation が無いときに >1 を要求すると ORT が警告を出しつつ 1 に落ちる
    ort.env.wasm.numThreads = isolated ? req.numThreads : 1;
    ort.env.wasm.simd = true;
    ort.env.logLevel = 'error';

    const modelBuf = await fetchCached(req.modelUrl);
    const options: ort.InferenceSession.SessionOptions = {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all',
    };
    if (req.externalDataUrl && req.externalDataPath) {
      const ext = await fetchCached(req.externalDataUrl);
      (options as unknown as Record<string, unknown>)['externalData'] = [
        { path: req.externalDataPath, data: new Uint8Array(ext) },
      ];
    }

    const t0 = performance.now();
    const session = await ort.InferenceSession.create(new Uint8Array(modelBuf), options);
    const createMs = performance.now() - t0;

    // 入力はメタデータから組み立てる。動的次元は inputSize で埋める。
    const feeds: Record<string, ort.Tensor> = {};
    const meta = (session as unknown as { inputMetadata?: { name?: string; type?: string; shape?: (number | string)[] }[] })
      .inputMetadata;
    session.inputNames.forEach((name, i) => {
      const m = meta?.[i];
      const declared = m?.shape ?? [1, 3, req.inputSize[1], req.inputSize[0]];
      const dims = declared.map((d, k) => {
        if (typeof d === 'number' && d > 0) return d;
        if (k === 0) return 1;
        if (k === declared.length - 1) return req.inputSize[0];
        if (k === declared.length - 2) return req.inputSize[1];
        return 1;
      });
      const n = dims.reduce((a, b) => a * b, 1);
      const type = (m?.type ?? 'tensor(float)').toLowerCase();
      if (type.includes('uint8')) {
        const a = new Uint8Array(n);
        for (let j = 0; j < n; j++) a[j] = (j * 37) % 251;
        feeds[name] = new ort.Tensor('uint8', a, dims);
      } else {
        const a = new Float32Array(n);
        for (let j = 0; j < n; j++) a[j] = ((j * 0.0137) % 1) * 0.8 + 0.1;
        feeds[name] = new ort.Tensor('float32', a, dims);
      }
    });

    const w0 = performance.now();
    await session.run(feeds);
    const warmupMs = performance.now() - w0;

    const times: number[] = [];
    for (let i = 0; i < Math.max(1, req.runs); i++) {
      const t = performance.now();
      await session.run(feeds);
      times.push(performance.now() - t);
    }
    await session.release();

    reply({
      numThreads: req.numThreads,
      ok: true,
      crossOriginIsolated: isolated,
      createMs: Math.round(createMs),
      warmupMs: Math.round(warmupMs),
      inferMs: Math.round(median(times)),
    });
  } catch (e) {
    reply({
      numThreads: req.numThreads,
      ok: false,
      crossOriginIsolated: isolated,
      error: e instanceof Error ? `${e.name}: ${e.message}` : String(e),
    });
  }
};
