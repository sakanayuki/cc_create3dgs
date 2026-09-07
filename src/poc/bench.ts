/**
 * PoC-1 の各検証項目（docs/08 §8.11）。
 *
 * 目的は3つ:
 *   R1  ORT の WebGPU EP が実機（特に iOS Safari）で動くか
 *   R10 DA3-Small が動くか、ONNX にカメラ内部パラメータの出力があるか
 *   R12 42万サーフェルで 60fps が出るか
 */
import * as ort from 'onnxruntime-web';
import { SPLAT_BYTES } from '../render/SplatRenderer';
import { WgslSplatRenderer } from '../render/backends/wgsl';
import { createSession, type Backend, type ModelSource } from '../runtime/OrtSession';
import { encodeOct, packHalf2, packRgba8 } from '../codec/pack';

const HF = 'https://huggingface.co';
const hf = (repo: string, path: string) => `${HF}/${repo}/resolve/main/${path}`;

export interface Candidate {
  readonly id: string;
  readonly label: string;
  readonly role: 'depth' | 'matte' | 'inpaint';
  readonly quant: string;
  readonly source: ModelSource;
  readonly inputSize: readonly [number, number];
  /** 想定ダウンロード量（MB）。実機で待たせる前に示すため。 */
  readonly approxMB: number;
  readonly note?: string;
}

/**
 * PoC では CI の量子化成果物を待たず、HuggingFace から直接取得する。
 * DA3-S は量子化版が公開されていないので fp32 で「グラフが通るか」だけを見る。
 * 速度の判断には、量子化版が公開されている V2-S の実測を使う。
 */
export const CANDIDATES: readonly Candidate[] = [
  {
    id: 'da3-small-fp32',
    label: 'Depth Anything 3 Small (fp32)',
    role: 'depth',
    quant: 'fp32',
    inputSize: [518, 518],
    approxMB: 105,
    source: {
      id: 'da3-small-fp32',
      url: hf('onnx-community/depth-anything-v3-small', 'onnx/model.onnx'),
      externalData: {
        url: hf('onnx-community/depth-anything-v3-small', 'onnx/model.onnx_data'),
        path: 'model.onnx_data',
      },
    },
    note: 'R10 の本題。まず「グラフが通るか」と「出力が深度だけか、内部パラメータもあるか」を見る。',
  },
  {
    id: 'da2-small-q4f16',
    label: 'Depth Anything V2 Small (q4f16)',
    role: 'depth',
    quant: 'q4f16',
    inputSize: [518, 518],
    approxMB: 19,
    source: { id: 'da2-small-q4f16', url: hf('onnx-community/depth-anything-v2-small', 'onnx/model_q4f16.onnx') },
    note: 'R1 の本題。MatMulNBits が Safari の WebGPU で動くか。',
  },
  {
    id: 'da2-small-fp16',
    label: 'Depth Anything V2 Small (fp16)',
    role: 'depth',
    quant: 'fp16',
    inputSize: [518, 518],
    approxMB: 50,
    source: { id: 'da2-small-fp16', url: hf('onnx-community/depth-anything-v2-small', 'onnx/model_fp16.onnx') },
    note: '縮退2段目。',
  },
  {
    id: 'da2-small-uint8',
    label: 'Depth Anything V2 Small (uint8)',
    role: 'depth',
    quant: 'uint8',
    inputSize: [518, 518],
    approxMB: 27,
    source: { id: 'da2-small-uint8', url: hf('onnx-community/depth-anything-v2-small', 'onnx/model_uint8.onnx') },
    note: '縮退3段目（WASM 用）。',
  },
  {
    id: 'modnet-uint8',
    label: 'MODNet (uint8)',
    role: 'matte',
    quant: 'uint8',
    inputSize: [512, 512],
    approxMB: 7,
    source: { id: 'modnet-uint8', url: hf('Xenova/modnet', 'onnx/model_uint8.onnx') },
  },
  {
    id: 'isnet-fp16',
    label: 'ISNet General (fp16)',
    role: 'matte',
    quant: 'fp16',
    inputSize: [1024, 1024],
    approxMB: 88,
    source: { id: 'isnet-fp16', url: hf('imgly/isnet-general-onnx', 'onnx/model_fp16.onnx') },
    note: '物体モード。uint8 版は未公開なので CI で作る。ここでは fp16 で速度の当たりを取る。',
  },
  {
    id: 'migan-fp32',
    label: 'MI-GAN pipeline v2 (fp32)',
    role: 'inpaint',
    quant: 'fp32',
    inputSize: [512, 512],
    approxMB: 28,
    source: { id: 'migan-fp32', url: hf('andraniksargsyan/migan', 'migan_pipeline_v2.onnx') },
    note: 'R11。GAN が WebGPU で動くか、512² が何秒か。',
  },
];

export interface ModelResult {
  id: string;
  backend: Backend;
  ok: boolean;
  error?: string;
  fetchMs?: number;
  createMs?: number;
  warmupMs?: number;
  /** ウォームアップ後の推論時間の中央値。 */
  inferMs?: number;
  bytes?: number;
  fromCache?: boolean;
  inputs?: string[];
  outputs?: string[];
  /** 出力の形。DA3 のカメラ内部パラメータ出力の有無を見るのに使う。 */
  outputShapes?: Record<string, readonly number[]>;
  outputStats?: Record<string, { min: number; max: number; mean: number }>;
}

type Meta = { name: string; type?: string; shape?: readonly (number | string)[] };

function metaOf(session: ort.InferenceSession, kind: 'input' | 'output'): Meta[] {
  const names = kind === 'input' ? session.inputNames : session.outputNames;
  const raw = (session as unknown as Record<string, unknown>)[`${kind}Metadata`];
  if (Array.isArray(raw) && raw.length === names.length) {
    return raw.map((m, i) => {
      const o = m as { name?: string; type?: string; shape?: readonly (number | string)[] };
      return { name: o.name ?? names[i] ?? `${kind}${i}`, ...(o.type ? { type: o.type } : {}), ...(o.shape ? { shape: o.shape } : {}) };
    });
  }
  return names.map((n) => ({ name: n }));
}

/** メタデータから妥当なダミー入力を作る。動的次元は候補の inputSize で埋める。 */
function makeInputs(session: ort.InferenceSession, size: readonly [number, number]): Record<string, ort.Tensor> {
  const feeds: Record<string, ort.Tensor> = {};
  for (const m of metaOf(session, 'input')) {
    const declared = m.shape ?? [1, 3, size[1], size[0]];
    const dims = declared.map((d, i) => {
      if (typeof d === 'number' && d > 0) return d;
      // 動的次元: 先頭はバッチ、末尾2つは空間次元とみなす
      if (i === 0) return 1;
      if (i === declared.length - 1) return size[0];
      if (i === declared.length - 2) return size[1];
      return 1;
    });
    const n = dims.reduce((a, b) => a * b, 1);
    const t = (m.type ?? 'tensor(float)').toLowerCase();

    if (t.includes('uint8')) {
      // MI-GAN のように uint8 の画像入力とマスクを取るモデル向け。
      // チャンネル数が 1 ならマスクとみなし、半分を穴（0）にする。
      const ch = dims.length >= 3 ? dims[1] ?? 1 : 1;
      const a = new Uint8Array(n);
      if (ch === 1) a.fill(0, 0, Math.floor(n / 2)), a.fill(255, Math.floor(n / 2));
      else for (let i = 0; i < n; i++) a[i] = (i * 37) % 251;
      feeds[m.name] = new ort.Tensor('uint8', a, dims);
    } else if (t.includes('float16')) {
      feeds[m.name] = new ort.Tensor('float16', new Uint16Array(n).fill(0x3400), dims);
    } else if (t.includes('int64')) {
      feeds[m.name] = new ort.Tensor('int64', new BigInt64Array(n), dims);
    } else {
      const a = new Float32Array(n);
      for (let i = 0; i < n; i++) a[i] = ((i * 0.0137) % 1) * 0.8 + 0.1;
      feeds[m.name] = new ort.Tensor('float32', a, dims);
    }
  }
  return feeds;
}

function statsOf(t: ort.Tensor): { min: number; max: number; mean: number } {
  const d = t.data as unknown as ArrayLike<number>;
  const n = Math.min(d.length, 100_000);
  let min = Infinity;
  let max = -Infinity;
  let sum = 0;
  for (let i = 0; i < n; i++) {
    const v = Number(d[i]);
    if (v < min) min = v;
    if (v > max) max = v;
    sum += v;
  }
  return { min, max, mean: sum / Math.max(n, 1) };
}

const median = (a: number[]) => {
  const s = [...a].sort((x, y) => x - y);
  return s[Math.floor(s.length / 2)] ?? 0;
};

export async function testModel(
  c: Candidate,
  backend: Backend,
  onProgress?: (loaded: number, total: number) => void,
  runs = 3,
): Promise<ModelResult> {
  const result: ModelResult = { id: c.id, backend, ok: false };
  let loaded: Awaited<ReturnType<typeof createSession>> | null = null;
  try {
    loaded = await createSession(c.source, { backend, quant: c.quant }, onProgress);
    result.fetchMs = Math.round(loaded.timing.fetchMs);
    result.createMs = Math.round(loaded.timing.createMs);
    result.bytes = loaded.bytes;
    result.fromCache = loaded.fromCache;

    const session = loaded.session;
    result.inputs = [...session.inputNames];
    result.outputs = [...session.outputNames];

    const feeds = makeInputs(session, c.inputSize);

    const w0 = performance.now();
    const first = await session.run(feeds);
    result.warmupMs = Math.round(performance.now() - w0);

    result.outputShapes = {};
    result.outputStats = {};
    for (const [name, tensor] of Object.entries(first)) {
      result.outputShapes[name] = [...tensor.dims];
      result.outputStats[name] = statsOf(tensor);
    }

    const times: number[] = [];
    for (let i = 0; i < runs; i++) {
      const t = performance.now();
      await session.run(feeds);
      times.push(performance.now() - t);
    }
    result.inferMs = Math.round(median(times));
    result.ok = true;
  } catch (e) {
    result.error = e instanceof Error ? `${e.name}: ${e.message}` : String(e);
  } finally {
    try {
      await loaded?.session.release();
    } catch {
      /* 解放できなくても続行する */
    }
  }
  return result;
}

// --- スレッド計測（PoC-2） ---------------------------------------------------

import type { ThreadTestRequest, ThreadTestResponse } from './threadWorker';

/** スレッド計測の対象。実モデルでも同梱の合成モデルでもよい。 */
export interface ThreadTarget {
  readonly label: string;
  readonly source: ModelSource;
  readonly inputSize: readonly [number, number];
  readonly approxMB: number;
}

/**
 * 同一オリジンに置いた合成モデル（10層の畳み込み @512²、0.8MB）。
 *
 * 実モデル（27MB）を落とさずにスレッドのスケーリングだけ即座に測れる。
 * 測っているのは畳み込みのスループットで、Transformer とは傾向が違いうるが、
 * 「マルチスレッドが効くか否か」の判断には十分で、待ち時間が桁違いに短い。
 */
export function syntheticThreadTarget(): ThreadTarget {
  const base = import.meta.env.BASE_URL ?? '/';
  return {
    label: '同梱の合成モデル（畳み込み10層 @512²）',
    source: { id: 'threadbench', url: `${base}threadbench.onnx` },
    inputSize: [512, 512],
    approxMB: 0.8,
  };
}

/**
 * スレッド数を変えて同じモデルを推論し、スケーリングを測る。
 *
 * ONNX Runtime は wasm 初期化時に一度だけ numThreads を読むので、
 * スレッド数ごとに新しいワーカーを立てる。
 */
export async function benchThreads(
  candidate: ThreadTarget,
  counts: readonly number[],
  runs = 2,
): Promise<ThreadTestResponse[]> {
  const out: ThreadTestResponse[] = [];
  for (const numThreads of counts) {
    const worker = new Worker(new URL('./threadWorker.ts', import.meta.url), { type: 'module' });
    try {
      const req: ThreadTestRequest = {
        modelUrl: candidate.source.url,
        ...(candidate.source.externalData
          ? {
              externalDataUrl: candidate.source.externalData.url,
              externalDataPath: candidate.source.externalData.path,
            }
          : {}),
        numThreads,
        inputSize: candidate.inputSize,
        runs,
      };
      const res = await new Promise<ThreadTestResponse>((resolve) => {
        const timer = setTimeout(
          () =>
            resolve({
              numThreads,
              ok: false,
              crossOriginIsolated: false,
              error: '3分以内に応答がありませんでした',
            }),
          180_000,
        );
        worker.onmessage = (e: MessageEvent<ThreadTestResponse>) => {
          clearTimeout(timer);
          resolve(e.data);
        };
        worker.onerror = (e) => {
          clearTimeout(timer);
          resolve({
            numThreads,
            ok: false,
            crossOriginIsolated: false,
            error: `ワーカーがエラーになりました: ${e.message}`,
          });
        };
        worker.postMessage(req);
      });
      out.push(res);
    } finally {
      worker.terminate();
    }
  }
  return out;
}

// --- 描画ベンチ（R12） -------------------------------------------------------

export interface RenderBenchResult {
  splatCount: number;
  /** 中央値フレーム時間（ミリ秒）。 */
  frameMs: number;
  fps: number;
  /** 背面カリング後に実際に描かれた数。 */
  drawnSplats: number;
  cullRatio: number;
}

/** 球殻状にサーフェルを撒く。前面／背面シェルの向きの分布を模す。 */
export function makeProceduralSplats(count: number): Uint8Array {
  const buf = new ArrayBuffer(count * SPLAT_BYTES);
  const f32 = new Float32Array(buf);
  const u32 = new Uint32Array(buf);
  const stride = SPLAT_BYTES / 4;

  for (let i = 0; i < count; i++) {
    // 黄金角で球面に一様分布させる
    const t = (i + 0.5) / count;
    const phi = Math.acos(1 - 2 * t);
    const theta = Math.PI * (1 + Math.sqrt(5)) * i;
    const r = 0.35 + 0.05 * Math.sin(i * 0.017);
    const x = r * Math.sin(phi) * Math.cos(theta);
    const y = r * Math.sin(phi) * Math.sin(theta) * 1.25;
    const z = r * Math.cos(phi);

    const o = i * stride;
    f32[o] = x;
    f32[o + 1] = y;
    f32[o + 2] = z;
    u32[o + 3] = encodeOct(x / r, y / r, z / r);
    const s = 1.6 / Math.sqrt(count);
    u32[o + 4] = packHalf2(s, s);
    const cr = Math.floor(140 + 90 * Math.sin(i * 0.001));
    const cg = Math.floor(140 + 90 * Math.sin(i * 0.0013 + 2));
    const cb = Math.floor(150 + 80 * Math.cos(i * 0.0007));
    u32[o + 5] = packRgba8(cr, cg, cb, 220);
  }
  return new Uint8Array(buf);
}

export async function benchRender(
  device: GPUDevice,
  canvas: HTMLCanvasElement,
  splatCount: number,
  frames = 90,
): Promise<RenderBenchResult> {
  const renderer = new WgslSplatRenderer({ device, canvas });
  try {
    renderer.resize(canvas.clientWidth * devicePixelRatio || 720, canvas.clientHeight * devicePixelRatio || 720);
    renderer.setSplats(makeProceduralSplats(splatCount), splatCount);
    renderer.setDepthRange(0.5, 1.6);

    // 最初の数フレームはシェーダ準備で遅いので捨てる。frames が小さいときは割合で決める。
    const warmup = Math.min(20, Math.max(2, Math.floor(frames * 0.25)));
    const times: number[] = [];
    for (let i = 0; i < frames; i++) {
      const yaw = (i / frames) * Math.PI * 0.7 - Math.PI * 0.35;
      renderer.setCamera({ yaw, pitch: 0.1, distance: 1.0, target: [0, 0, 0] });
      const t = performance.now();
      renderer.render();
      // GPU の完了を待つ。onSubmittedWorkDone が実測に最も近い。
      await device.queue.onSubmittedWorkDone();
      const dt = performance.now() - t;
      if (i >= warmup) times.push(dt);
    }
    const frameMs = median(times);
    const drawn = renderer.stats.drawnSplats;
    return {
      splatCount,
      frameMs: Math.round(frameMs * 100) / 100,
      fps: Math.round((1000 / Math.max(frameMs, 0.01)) * 10) / 10,
      drawnSplats: drawn,
      cullRatio: drawn > 0 ? Math.round((1 - drawn / splatCount) * 1000) / 1000 : 0,
    };
  } finally {
    renderer.dispose();
  }
}
