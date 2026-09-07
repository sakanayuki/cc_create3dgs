/**
 * 生成パイプラインの司令塔（docs/03 §3.1）。
 *
 * 写真1枚を受け取り、レンダラに渡せるスプラットまで持っていく。
 * 個々の工程は他のモジュールが持っているので、ここがするのは
 * 「順番」「モデルの呼び出し」「進捗の報告」の3つだけ。
 *
 * 途中でプレビューを1回返す（docs/03 §3.1 のフロー図）。
 * インペイントと継ぎ目補正を待たずに見せるほうが体感が速いため。
 */
import * as ort from 'onnxruntime-web';
import { estimateNormals, calibrate } from './3-calibrate';
import { thicknessMap } from './4-shell';
import { buildSplats, DEFAULT_BUILD_PARAMS, type BuildParams, type SplatBuild } from './6-splats';
import { adaptiveSample, solveSamplingParams, samplingReduction } from './7-sample';
import { refineMatte } from './1-matte';
import { prepareImage, WORKING_GRID, type Letterbox } from './0-preprocess';
import { IMAGENET_MEAN, IMAGENET_STD, minMax, resizePlane, resizeRgba, toTensorNCHW } from './imageOps';
import { createSession, type Backend } from '../runtime/OrtSession';
import { resolveModel, type HfFallback } from '../runtime/modelCatalog';

/** 被写体のモード。docs/03 §3.3。 */
export type SubjectMode = 'person' | 'object';

export interface GenerateOptions {
  readonly mode: SubjectMode;
  /** 作業グリッド。docs/04 §4.7 のプリセット。 */
  readonly grid: number;
  /** 適応サンプリングの目標統合率。 */
  readonly reduction: number;
  /** 推論の実行プロバイダ。 */
  readonly backend: Backend;
  /** 進捗の報告。0〜1 と、いま何をしているか。 */
  readonly onProgress?: (fraction: number, label: string) => void;
  /** プレビューができた時点で1回呼ばれる。 */
  readonly onPreview?: (build: SplatBuild) => void;
  readonly params?: Partial<BuildParams>;
}

export const DEFAULT_GENERATE_OPTIONS: Omit<GenerateOptions, 'backend'> = {
  mode: 'person',
  grid: WORKING_GRID,
  reduction: 0.3,
};

export interface GenerateResult {
  readonly build: SplatBuild;
  readonly box: Letterbox;
  /** 作業グリッド上の各プレーン。書き出し（.pgs）で使う。 */
  readonly planes: {
    readonly color: Uint8ClampedArray;
    readonly alpha: Uint8ClampedArray;
    readonly depth: Float32Array;
  };
  readonly stats: {
    readonly focalPx: number;
    readonly cells: number;
    readonly reduction: number;
    readonly timings: Record<string, number>;
    /** 深度モデルが内部パラメータを返したか。返さなければ画角を仮定している。 */
    readonly intrinsicsFromModel: boolean;
  };
}

/**
 * 同一オリジン（CI が焼いたもの）→ HuggingFace の順に試す。
 *
 * どの量子化方式を使うかは `modelCatalog` がバックエンドに応じて選ぶ
 * （決定 D22）。ここはその結果を順に試すだけ。
 */
async function loadModel(
  id: string,
  backend: Backend,
  fallback: HfFallback,
): Promise<ort.InferenceSession> {
  const sources = await resolveModel(id, backend, fallback);
  const errors: string[] = [];
  for (const src of sources) {
    try {
      const r = await createSession(src, { backend, quant: 'auto' });
      return r.session;
    } catch (e) {
      errors.push(`${src.url}: ${String(e)}`);
    }
  }
  throw new Error(`${id} を読み込めませんでした:\n${errors.join('\n')}`);
}

/** セッションの入力名は1つとは限らないので、最初の1つに入れる。 */
function feedOf(session: ort.InferenceSession, tensor: ort.Tensor): Record<string, ort.Tensor> {
  const name = session.inputNames[0];
  if (!name) throw new Error('モデルに入力がありません');
  return { [name]: tensor };
}

/** 名前に部分一致する出力を探す。無ければ最初の出力。 */
function pickOutput(
  outputs: ort.InferenceSession.OnnxValueMapType,
  session: ort.InferenceSession,
  ...names: string[]
): ort.Tensor {
  for (const want of names) {
    for (const key of session.outputNames) {
      if (key.toLowerCase().includes(want)) {
        const t = outputs[key];
        if (t) return t as ort.Tensor;
      }
    }
  }
  const first = session.outputNames[0];
  const t = first ? outputs[first] : undefined;
  if (!t) throw new Error('モデルの出力を取り出せませんでした');
  return t as ort.Tensor;
}

/**
 * ① 被写体抽出。
 *
 * 人物は MODNet、物体は ISNet。どちらも入力は正方の RGB で、出力は
 * 1チャンネルのマット。値域はモデルによって [0,1] だったり広かったりするので、
 * 最大値で正規化してから 0..255 にする。
 */
async function runMatte(
  rgba: Uint8ClampedArray,
  grid: number,
  mode: SubjectMode,
  backend: Backend,
): Promise<Uint8ClampedArray> {
  const size = mode === 'person' ? 512 : 1024;
  const session =
    mode === 'person'
      ? await loadModel('modnet', backend, { repo: 'Xenova/modnet', file: 'onnx/model_uint8.onnx' })
      : await loadModel('isnet-general', backend, {
          repo: 'imgly/isnet-general-onnx',
          file: 'onnx/model.onnx',
        });

  const small = resizeRgba(rgba, grid, grid, size, size);
  const input = toTensorNCHW(small, size, size, { mean: [0.5, 0.5, 0.5], std: [0.5, 0.5, 0.5] });
  const out = await session.run(feedOf(session, new ort.Tensor('float32', input, [1, 3, size, size])));
  const t = pickOutput(out, session, 'matte', 'alpha', 'output');
  const raw = t.data as unknown as Float32Array;

  // 出力の空間サイズはモデル次第。要素数から一辺を割り出す。
  const side = Math.round(Math.sqrt(raw.length));
  const [, hi] = minMax(raw);
  const norm = new Float32Array(raw.length);
  const k = hi > 1.5 ? 1 / hi : 1; // 0..255 で返すモデルもある
  for (let i = 0; i < raw.length; i++) norm[i] = Math.max(0, Math.min(1, (raw[i] as number) * k));

  const full = resizePlane(norm, side, side, grid, grid);
  const alpha = new Uint8ClampedArray(grid * grid);
  for (let i = 0; i < alpha.length; i++) alpha[i] = Math.round((full[i] as number) * 255);
  return alpha;
}

export interface DepthOutput {
  /** モデルの生出力（正規化前）。 */
  readonly raw: Float32Array;
  readonly kind: 'depth' | 'inverse-depth';
  /** モデルが返した焦点距離（画素）。返さなければ null。 */
  readonly focalPx: number | null;
}

/**
 * ② 深度推定。
 *
 * DA3 は `predicted_depth` と `intrinsics` を返す（PoC-1 で実機確認済み）。
 * `intrinsics` があれば焦点距離が直接得られるので、画角の仮定が要らない。
 * 返さないモデル（V2）では呼び出し側が既定値を使う。
 */
async function runDepth(
  rgba: Uint8ClampedArray,
  grid: number,
  backend: Backend,
): Promise<DepthOutput> {
  const size = 518;
  const session = await loadModel('depth-anything-v3-small', backend, {
    repo: 'onnx-community/depth-anything-v3-small',
    file: 'onnx/model.onnx',
    extra: 'onnx/model.onnx_data',
  });

  const small = resizeRgba(rgba, grid, grid, size, size);
  const input = toTensorNCHW(small, size, size, { mean: IMAGENET_MEAN, std: IMAGENET_STD });
  const out = await session.run(feedOf(session, new ort.Tensor('float32', input, [1, 3, size, size])));

  const depthTensor = pickOutput(out, session, 'predicted_depth', 'depth', 'output');
  const raw = depthTensor.data as unknown as Float32Array;
  const side = Math.round(Math.sqrt(raw.length));
  const full = resizePlane(raw, side, side, grid, grid);

  // intrinsics は [fx 0 cx; 0 fy cy; 0 0 1]。推論解像度での画素単位なので、
  // 作業グリッドの大きさに直す。
  let focalPx: number | null = null;
  for (const name of session.outputNames) {
    if (!name.toLowerCase().includes('intrinsic')) continue;
    const t = out[name] as ort.Tensor | undefined;
    const d = t?.data as unknown as Float32Array | undefined;
    if (d && d.length >= 5 && Number.isFinite(d[0]) && (d[0] as number) > 0) {
      focalPx = (d[0] as number) * (grid / size);
    }
    break;
  }

  // DA3 は深度そのもの、V2 は逆深度。出力名で見分ける。
  const isInverse = !session.outputNames.some((n) => n.toLowerCase().includes('predicted_depth'));
  return { raw: full, kind: isInverse ? 'inverse-depth' : 'depth', focalPx };
}

/** 画角 55° を仮定したときの焦点距離。モデルが内部パラメータを返さないときの予備。 */
function assumedFocal(grid: number): number {
  return grid / (2 * Math.tan((55 * Math.PI) / 180 / 2));
}

/**
 * 写真1枚からスプラットまで。
 *
 * 進捗の刻みは「待っている人が見て意味の分かる単位」にする。
 * 内部の関数呼び出し単位で刻んでも、見ている側には何も伝わらない。
 */
export async function generate(photo: Blob, options: GenerateOptions): Promise<GenerateResult> {
  const opts = { ...DEFAULT_GENERATE_OPTIONS, ...options };
  const grid = opts.grid;
  const report = (f: number, label: string): void => opts.onProgress?.(f, label);
  const timings: Record<string, number> = {};
  const mark = async <T>(key: string, fn: () => Promise<T> | T): Promise<T> => {
    const t0 = performance.now();
    const v = await fn();
    timings[key] = Math.round(performance.now() - t0);
    return v;
  };

  report(0.02, '写真を読み込んでいます');
  const prepared = await mark('前処理', () => prepareImage(photo, grid));
  const rgba = prepared.rgba;

  report(0.1, '被写体を切り抜いています');
  const rawAlpha = await mark('マット推定', () => runMatte(rgba, grid, opts.mode, opts.backend));
  const alpha = await mark('マット後処理', () => refineMatte(rawAlpha, rgba, grid, grid));

  report(0.35, '奥行きを推定しています');
  const depthOut = await mark('深度推定', () => runDepth(rgba, grid, opts.backend));
  const focalPx = depthOut.focalPx ?? assumedFocal(grid);

  report(0.62, '奥行きを整えています');
  const calibrated = await mark('深度較正', () =>
    calibrate({
      raw: depthOut.raw,
      width: grid,
      height: grid,
      alpha,
      kind: depthOut.kind,
      focalPx,
    }),
  );

  // 0..1 に直した深度。以降の工程はこの形で受け取る。
  const depth01 = new Float32Array(grid * grid);
  for (let i = 0; i < depth01.length; i++) depth01[i] = (calibrated.depth[i] as number) / 65535;
  // 法線は実距離で推定する。正規化した値のままだと焦点距離と単位が合わない。
  const metric = new Float32Array(grid * grid);
  const span = calibrated.farZ - calibrated.nearZ;
  for (let i = 0; i < metric.length; i++) metric[i] = calibrated.nearZ + (depth01[i] as number) * span;

  report(0.72, '面の向きを求めています');
  const normals = await mark('法線推定', () =>
    estimateNormals(metric, grid, grid, focalPx, span * 0.05),
  );

  report(0.82, 'ガウシアンを配置しています');
  const samplingParams = await mark('サンプリング閾値', () =>
    solveSamplingParams(depth01, rgba, alpha, grid, grid, opts.reduction),
  );
  const cells = await mark('適応サンプリング', () =>
    adaptiveSample(depth01, rgba, alpha, grid, grid, samplingParams),
  );

  report(0.9, '厚みをつけています');
  const thickness = await mark('厚みマップ', () =>
    thicknessMap(alpha, grid, grid, { maxThickness: 0.35, profile: 'ellipsoid' }),
  );

  const buildParams: BuildParams = { ...DEFAULT_BUILD_PARAMS, ...opts.params };
  const build = await mark('スプラット組み立て', () =>
    buildSplats(
      {
        cells,
        normals,
        width: grid,
        height: grid,
        focalPx,
        nearZ: calibrated.nearZ,
        farZ: calibrated.farZ,
      },
      rgba,
      alpha,
      thickness,
      null,
      buildParams,
    ),
  );
  opts.onPreview?.(build);

  report(1, '完成しました');
  return {
    build,
    box: prepared.box,
    planes: { color: rgba, alpha, depth: depth01 },
    stats: {
      focalPx,
      cells: cells.cellCount,
      reduction: samplingReduction(cells),
      timings,
      intrinsicsFromModel: depthOut.focalPx !== null,
    },
  };
}
