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
import { estimateNormals, calibrate, pullBoundaryDepthInward } from './3-calibrate';
import { thicknessMap } from './4-shell';
import { buildSplats, DEFAULT_BUILD_PARAMS, type BuildParams, type SplatBuild } from './6-splats';
import { adaptiveSample, solveSamplingParams, samplingReduction } from './7-sample';
import { applyMatteGate, refineMatte } from './1-matte';
import {
  applyFaceRelief,
  boxFromLandmarks,
  faceDepthSurface,
  headBoxFromMatte,
  headDepthTile,
  type FaceLandmark,
} from './geometry/faceSurface';
import {
  depthTiles,
  expandToSquare,
  prepareImage,
  prepareImageScaled,
  subjectBBox,
  WORKING_GRID,
  type Letterbox,
  gridToSource,
  type Rect,
} from './0-preprocess';
import { fuseDepth, type DepthTile } from './2-depth';
import { applyLimbRoundness, DEFAULT_LIMB_PARAMS } from './geometry/limbRoundness';
import { clampReliefAmplitude, DEFAULT_RELIEF_CLAMP } from './geometry/reliefClamp';
import { limitCrossSectionDent, DEFAULT_CROSS_SECTION } from './geometry/crossSection';
import {
  buildInpaintMask,
  compositeInpaint,
  stretchFallback,
  type InpaintMask,
} from './8-inpaint';
import {
  cropRgba,
  IMAGENET_MEAN,
  IMAGENET_STD,
  minMax,
  resizePlane,
  resizeRgba,
  toTensorNCHW,
} from './imageOps';
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
  /**
   * WebGPU に `shader-f16` があるか。無ければ q4f16 のモデルを避ける
   * （src/runtime/modelCatalog.ts の manifestBackendKey）。既定は「ある」。
   */
  readonly shaderF16?: boolean;
  /** 進捗の報告。0〜1 と、いま何をしているか。 */
  readonly onProgress?: (fraction: number, label: string) => void;
  /** プレビューができた時点で1回呼ばれる。 */
  readonly onPreview?: (build: SplatBuild) => void;
  readonly params?: Partial<BuildParams>;
  /** ⑧ のインペイントを行うか。軽量プリセットは false（docs/04 §4.7）。 */
  readonly inpaint?: boolean;
  /** ② のタイルパスを行うか。軽量プリセットは false（docs/04 §4.7）。 */
  readonly depthTiles?: boolean;
}

/**
 * 背面シェルの厚み（正規化深度に対する割合）。
 *
 * 0.35 だと前面と背面の隔たりが中央値 0.127 になり、点群の 奥行き ÷ 幅 を
 * 0.12 も押し上げていた。理想的な出力（別実装）は隔たり 0.012 で、ほぼ
 * 一枚の面である。半球カバー（決定 D2）のために厚みは要るが、この量は
 * 過大だった。0.12 で隔たり 0.051 になり、紙のようには見えない。
 */
const BACK_SHELL_THICKNESS = 0.12;

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
    /** 実寸をそのまま使ったか。false なら比率を指定して引き伸ばしている。 */
    readonly metricDepth: boolean;
    /** 被写体の 奥行き ÷ 高さ。構図で変わるので参考値。 */
    readonly depthToHeight: number;
    /** 被写体の 奥行き ÷ 幅。人物なら 0.9 前後が自然。妥当性はこれで見る。 */
    readonly depthToWidth: number;
    /** 深度タイルパスで実際に使えたタイル数（0 なら全体パスのみ）。 */
    readonly depthTiles: number;
    /** 顔の起伏を入れられたか（顔が写っていなければ false）。 */
    readonly faceRelief: boolean;
    /** ⑧ で何を使ったか。mi-gan が本命、stretch は縮退、skipped は段差なし。 */
    readonly inpaint: 'mi-gan' | 'stretch' | 'skipped';
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
  shaderF16 = true,
): Promise<ort.InferenceSession> {
  const sources = await resolveModel(id, backend, fallback, shaderF16);
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

/**
 * 補助モデル（u2netp / face-mesh）は必ず WASM で回す（v2.6、実機で判明）。
 *
 * ORT の WebGPU バックエンドはセッションをまたいで 1 つしかない。そこで
 * 推論が例外で終わると内部の `currentKernelId` が立ったまま残り、**次に走る
 * 別のモデル**が
 *
 *     kernel "[Resize] /backbone/Resize" is not allowed to be called recursively
 *
 * で落ちる。原因と現象が別のモデルに分かれるので、実機のログからは追えない。
 * 実際、この 2 つを外すと WebGPU は完走した。
 *
 * どちらも小さい（320² と 256²）ので、WASM で回しても待ち時間に響かない。
 * 重い DA3・MODNet・MI-GAN は WebGPU のままにする。
 */
/**
 * 境界の深度を引き込む帯の幅（作業グリッドに対する割合）。
 *
 * 深度モデルは輪郭で前景と背景を混ぜた値を返し、そのにじみは α が 1 の
 * 内側にも続く。帯の広さはモデルの推論解像度で決まるので、被写体の
 * 大きさではなくグリッドに対する割合で置く。docs/09 §V1・§V19。
 */
const BOUNDARY_BAND_RATIO = 0.012;

/**
 * 顔の外での局所強調の倍率（docs/11 §11.6 S1）。
 *
 * 強調（`reliefBoost`、既定 3）は顔の凹凸を出すための処理で、体では
 * **あり得ない起伏を作る側に働く**。実測（立ち姿）で、体を 1 倍にすると
 * 横断面の飛び出しの 99 パーセンタイルが **13.83% → 9.11%**、首肩の
 * 奥行き÷幅が 0.43 → 0.54（参照実装 0.55）になり、顔は 0.45 → 0.51
 * （参照実装 0.47）と保たれた。至近距離の描画の勾配も 2.50 → 2.69 で
 * 落ちない。
 */
const BODY_RELIEF_BOOST = 1;

const AUX_BACKEND: Backend = 'wasm';

/** セッションの入力名は1つとは限らないので、最初の1つに入れる。 */
function feedOf(session: ort.InferenceSession, tensor: ort.Tensor): Record<string, ort.Tensor> {
  const name = session.inputNames[0];
  if (!name) throw new Error('モデルに入力がありません');
  return { [name]: tensor };
}

/**
 * モデルが求める階数に合わせて入力の形を決める。
 *
 * NCHW を決め打ちにしてはいけない。**Depth Anything 3 は多視点モデル**で、
 * 入力が `[batch, views, 3, H, W]` の5階になる（1枚だけ渡すときは views=1）。
 * 4階で渡すと onnxruntime が弾く。
 *
 *     Invalid rank for input: pixel_values Got: 4 Expected: 5
 *
 * 宣言された階数が読めなければ NCHW を仮定する。
 */
export function inputDims(
  declaredRank: number | undefined,
  channels: number,
  height: number,
  width: number,
): number[] {
  const nchw = [1, channels, height, width];
  if (!declaredRank || declaredRank === 4) return nchw;
  if (declaredRank === 3) return [channels, height, width];
  // 5階以上は、バッチと空間次元の間に視点などの軸が挟まる。1 で埋める。
  return [1, ...Array(declaredRank - 4).fill(1), channels, height, width];
}

/** セッションの入力宣言から階数を読む。取れなければ undefined。 */
function declaredRank(session: ort.InferenceSession): number | undefined {
  const meta = (session as unknown as Record<string, unknown>)['inputMetadata'];
  if (!Array.isArray(meta) || meta.length === 0) return undefined;
  const first = meta[0] as { shape?: readonly unknown[] } | undefined;
  return Array.isArray(first?.shape) ? first.shape.length : undefined;
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

/** マットモデルを1つ動かして、作業グリッドの 0..255 マットにする。 */
async function inferMatte(
  session: ort.InferenceSession,
  rgba: Uint8ClampedArray,
  grid: number,
  size: number,
  norm: { mean: readonly [number, number, number]; std: readonly [number, number, number] },
): Promise<Uint8ClampedArray> {
  const small = resizeRgba(rgba, grid, grid, size, size);
  const input = toTensorNCHW(small, size, size, norm);
  const dims = inputDims(declaredRank(session), 3, size, size);
  const out = await session.run(feedOf(session, new ort.Tensor('float32', input, dims)));
  const t = pickOutput(out, session, 'matte', 'alpha', 'output');
  const raw = t.data as unknown as Float32Array;

  // 出力の空間サイズはモデル次第。要素数から一辺を割り出す。
  const side = Math.round(Math.sqrt(raw.length));
  // u2netp は符号なしの生スコアを返すので、最小最大で 0..1 に伸ばす。
  // MODNet は既に [0,1] なので、この正規化を通しても値は変わらない。
  const [lo, hi] = minMax(raw);
  const span = Math.max(hi - lo, 1e-9);
  const unit = new Float32Array(raw.length);
  for (let i = 0; i < raw.length; i++) {
    unit[i] = Math.max(0, Math.min(1, ((raw[i] as number) - lo) / span));
  }

  const full = resizePlane(unit, side, side, grid, grid);
  const alpha = new Uint8ClampedArray(grid * grid);
  for (let i = 0; i < alpha.length; i++) alpha[i] = Math.round((full[i] as number) * 255);
  return alpha;
}

/**
 * ① 被写体抽出。
 *
 * 人物は「u2netp が領域を決め、MODNet が輪郭を出す」二段（v2.5、実写で判明）。
 *
 * MODNet だけだと、床に座る人物で敷物を被写体に含めてしまう。実写では敷物を
 * α=254（確信度いっぱい）で被写体と判定し、モデル全体の 28% が敷物になった。
 * 閾値の問題ではない。MODNet は肖像マッティング用で、この構図が分布外である。
 * u2netp は同じ写真で敷物を α=0 と正しく除く。
 *
 * 逆に u2netp は 320² なので輪郭が粗い（頭部の輪郭長 54,461 対 MODNet 67,226）。
 * そこで領域は u2netp、輪郭は MODNet と役割を分ける。u2netp を少しだけ膨らませて
 * 門にするのは、MODNet のほうがわずかに外側までシルエットを取るため。
 *
 * 物体モードは ISNet 単体（もともと領域を正しく取る）。
 */
async function runMatte(
  rgba: Uint8ClampedArray,
  grid: number,
  mode: SubjectMode,
  backend: Backend,
  shaderF16: boolean,
): Promise<Uint8ClampedArray> {
  if (mode !== 'person') {
    const session = await loadModel(
      'isnet-general',
      backend,
      { repo: 'imgly/isnet-general-onnx', file: 'onnx/model.onnx' },
      shaderF16,
    );
    return inferMatte(session, rgba, grid, 1024, { mean: [0.5, 0.5, 0.5], std: [0.5, 0.5, 0.5] });
  }

  const modnet = await loadModel(
    'modnet',
    backend,
    { repo: 'Xenova/modnet', file: 'onnx/model_uint8.onnx' },
    shaderF16,
  );
  const fine = await inferMatte(modnet, rgba, grid, 512, {
    mean: [0.5, 0.5, 0.5],
    std: [0.5, 0.5, 0.5],
  });

  let gate: Uint8ClampedArray;
  try {
    const u2 = await loadModel(
      'u2netp',
      AUX_BACKEND,
      { repo: 'tomjackson2023/rembg', file: 'u2netp.onnx' },
      shaderF16,
    );
    // u2netp は ImageNet 正規化で学習されている。
    gate = await inferMatte(u2, rgba, grid, 320, { mean: IMAGENET_MEAN, std: IMAGENET_STD });
  } catch {
    // 門が取れなければ MODNet 単体に落ちる。敷物は残るが、生成は止めない。
    return fine;
  }
  return applyMatteGate(fine, gate, grid, grid);
}

/**
 * 顔モデルの自己申告スコアの下限。
 *
 * 実測では、マットから当てた初期位置で 6.9、そこから切り直すと 17.6 だった。
 * 顔が写っていない・横向きすぎる写真では大きく負に振れる。0 を境にすると
 * 「たまたま顔に見えた何か」を拾うので、少し上に置く。
 */
const FACE_MIN_SCORE = 3;

/** 顔の面がこの画素数に満たなければ使わない。小さすぎる顔は効果がない。 */
const FACE_MIN_PIXELS = 1024;

/**
 * 切り直しの最大回数。
 *
 * 実測で 3 回目に乗った写真がある（床に座った人物: −7.5 → 2.5 → 14.6 → 19.2）。
 * 1 回は 256² の 0.6M パラメータなので、数回まわしても無視できる。
 */
const FACE_MAX_PASSES = 4;

/**
 * 顔の切り出しを 256² の RGBA で返す。取れなければ null。
 *
 * 作業グリッドは長辺 1024 なので、大きい写真では顔が縮んでいる。実写
 * （1116×2000）では元写真で 273px の顔が、グリッドでは 140px しかない。
 * landmark モデルは 256² に伸ばして受け取るので、グリッドから切ると
 * **一度縮めたものを引き伸ばす**ことになる。元写真から切れば本来の
 * 解像度で渡せる。
 */
type FacePatchSource = (box: Rect, size: number) => Promise<Uint8ClampedArray | null>;

/**
 * 顔の 3D landmark を 1 回推論する（478 点、画像座標へ戻して返す）。
 *
 * モデルは 256² の正方入力で、x/y/z を同じ尺度（入力画素）で返す。
 * z は頭の中心を原点にした相対値で、小さいほど手前。深度モデルと
 * 向きが同じなので符号はそのまま使える。
 */
async function inferFaceLandmarks(
  session: ort.InferenceSession,
  rgba: Uint8ClampedArray,
  grid: number,
  box: Rect,
  /** 元写真から切り出す手立て。無ければ作業グリッドから切る。 */
  nativePatch: FacePatchSource | null = null,
): Promise<{ points: FaceLandmark[]; score: number }> {
  const size = 256;
  let small: Uint8ClampedArray | null = nativePatch ? await nativePatch(box, size) : null;
  if (!small) {
    const patch = cropRgba(rgba, grid, box.x, box.y, box.width, box.height);
    small = resizeRgba(patch, box.width, box.height, size, size);
  }

  // このモデルは NHWC で 0..1 の RGB を取る（正規化はしない）。
  const input = new Float32Array(size * size * 3);
  for (let i = 0, j = 0; i < small.length; i += 4) {
    input[j++] = (small[i] as number) / 255;
    input[j++] = (small[i + 1] as number) / 255;
    input[j++] = (small[i + 2] as number) / 255;
  }
  const feeds = feedOf(session, new ort.Tensor('float32', input, [1, size, size, 3]));
  const out = await session.run(feeds);
  const names = session.outputNames;
  const lmT = out[names[0] as string] as ort.Tensor;
  const scoreT = names.length > 1 ? (out[names[1] as string] as ort.Tensor) : null;
  const raw = lmT.data as Float32Array;
  const k = box.width / size;

  const points: FaceLandmark[] = [];
  for (let i = 0; i + 2 < raw.length; i += 3) {
    points.push({
      x: box.x + (raw[i] as number) * k,
      y: box.y + (raw[i + 1] as number) * k,
      z: (raw[i + 2] as number) * k,
    });
  }
  const score = scoreT ? ((scoreT.data as Float32Array)[0] as number) : 0;
  return { points, score };
}

/**
 * 元写真から顔を切り出す手立てを作る（v2.6.4、docs/09 §V18）。
 *
 * 縮小して載せた写真でだけ意味がある。拡大して載せた写真（§3.2.2）では
 * グリッドのほうが細かいので null を返し、呼び出し側はグリッドから切る。
 *
 * EXIF の回転を自分で解かないよう、**画像全体を `from-image` で一度
 * 復号してから切る**。`createImageBitmap` の切り出し矩形と回転を同時に
 * 使うと、どちらが先に効くかがブラウザ間で揃わない。
 *
 * 失敗しても生成は止めない。null を返せばグリッド経由に落ちる。
 */
function nativeFacePatchSource(photo: Blob, lb: Letterbox): FacePatchSource | null {
  if (lb.scale >= 1) return null;
  if (typeof createImageBitmap !== 'function' || typeof OffscreenCanvas === 'undefined') return null;

  let bitmap: ImageBitmap | null = null;
  let failed = false;
  return async (box: Rect, size: number): Promise<Uint8ClampedArray | null> => {
    if (failed) return null;
    try {
      if (!bitmap) bitmap = await createImageBitmap(photo, { imageOrientation: 'from-image' });
      const [sx, sy] = gridToSource(lb, box.x, box.y);
      const side = box.width / lb.scale;
      // 元写真より粗くなるなら、わざわざ切り直す意味が無い。
      if (side <= size) return null;
      const canvas = new OffscreenCanvas(size, size);
      const ctx = canvas.getContext('2d', { willReadFrequently: true });
      if (!ctx) return null;
      ctx.imageSmoothingQuality = 'high';
      ctx.drawImage(bitmap, sx, sy, side, side, 0, 0, size, size);
      return ctx.getImageData(0, 0, size, size).data;
    } catch {
      failed = true;
      return null;
    }
  };
}

/**
 * 顔の起伏を深度へ入れる（docs/03 §3.4.5、v2.6）。
 *
 * 一般の深度モデルは顔をほぼ平らな楕円として返す。DA3 は Small でも
 * Base でも鼻・眼窩・唇が出ず、入力解像度を上げても変わらない
 * （faceSurface.ts の表）。顔専用の landmark モデルで面を起こし、
 * 深度の細部の帯だけを差し替える。
 *
 * 顔検出モデルは載せない。マットから頭の位置を当てて 1 回目を回し、
 * その landmark で切り直して 2 回目を回す。実測でモデルの自己申告
 * スコアが 6.9 → 17.6 に上がった。
 *
 * 失敗しても生成は止めない。顔が写っていない写真のほうが普通である。
 */
async function applyFaceDepth(
  depth: Float32Array,
  rgba: Uint8ClampedArray,
  alpha: Uint8ClampedArray,
  grid: number,
  focalPx: number,
  backend: Backend,
  shaderF16: boolean,
  nativePatch: FacePatchSource | null = null,
): Promise<{ depth: Float32Array; applied: boolean }> {
  const seed = headBoxFromMatte(alpha, grid, grid);
  if (!seed) return { depth, applied: false };

  let session: ort.InferenceSession;
  try {
    session = await loadModel(
      'face-mesh',
      AUX_BACKEND,
      { repo: 'astaileyyoung/FaceMeshONNX', file: 'mesh.onnx' },
      shaderF16,
    );
  } catch {
    return { depth, applied: false };
  }

  try {
    // 切り直しながら数回まわす。マットから当てた初期位置は粗いので、
    // 1 往復では足りないことがある。実測（床に座った人物）で
    // −7.5 → 2.5 → 14.6 → 19.2 と、3 回目でようやく乗った。
    let box = seed;
    let best: { points: FaceLandmark[]; score: number; box: Rect } | null = null;
    for (let i = 0; i < FACE_MAX_PASSES; i++) {
      const r = await inferFaceLandmarks(session, rgba, grid, box, nativePatch);
      if (!best || r.score > best.score) best = { ...r, box };
      const next = boxFromLandmarks(r.points, grid, grid);
      if (!next) break;
      // 切り出しが動かなくなったら、それ以上まわしても変わらない。
      if (Math.abs(next.x - box.x) + Math.abs(next.y - box.y) + Math.abs(next.width - box.width) < 4) {
        box = next;
        break;
      }
      box = next;
    }
    if (!best || best.score < FACE_MIN_SCORE) return { depth, applied: false };

    const surface = faceDepthSurface(best.points, best.box);
    if (surface.covered < FACE_MIN_PIXELS) return { depth, applied: false };
    return {
      depth: applyFaceRelief(depth, surface, best.box, grid, grid, focalPx),
      applied: true,
    };
  } catch {
    return { depth, applied: false };
  }
}

export interface DepthOutput {
  /** モデルの生出力（正規化前）。 */
  readonly raw: Float32Array;
  readonly kind: 'depth' | 'inverse-depth';
  /** モデルが返した焦点距離（画素）。返さなければ null。 */
  readonly focalPx: number | null;
  /** タイルパスで使い回すためのセッション。 */
  readonly session: ort.InferenceSession;
}

/**
 * ② 深度推定。
 *
 * DA3 は `predicted_depth` と `intrinsics` を返す（PoC-1 で実機確認済み）。
 * `intrinsics` があれば焦点距離が直接得られるので、画角の仮定が要らない。
 * 返さないモデル（V2）では呼び出し側が既定値を使う。
 */
async function depthSession(backend: Backend, shaderF16: boolean): Promise<ort.InferenceSession> {
  return loadModel(
    'depth-anything-v3-small',
    backend,
    {
      repo: 'onnx-community/depth-anything-v3-small',
      file: 'onnx/model.onnx',
      extra: 'onnx/model.onnx_data',
    },
    shaderF16,
  );
}

/** 切り出した領域を 518² に伸ばして推論し、元の大きさに戻す。 */
async function inferDepthPatch(
  session: ort.InferenceSession,
  rgba: ArrayLike<number>,
  srcWidth: number,
  srcHeight: number,
  rect: Rect,
): Promise<Float32Array> {
  const size = 518;
  // 正方でない領域をそのまま 518² へ伸ばすと、被写体が縦横で違う倍率に
  // なる。実写では縦長タイル（330×550）で顔が横に 1.67 倍伸び、そこから
  // 返る深度が使い物にならなかった。まず画像の内側で正方に広げて推論し、
  // 返ってきた深度から元の領域を切り出す。黒でパディングしないのは、
  // 縁に偽の深度段差ができるため。
  const square = expandToSquare(rect, srcWidth, srcHeight);
  const patch = cropRgba(rgba, srcWidth, square.x, square.y, square.width, square.height);
  const small = resizeRgba(patch, square.width, square.height, size, size);
  const input = toTensorNCHW(small, size, size, { mean: IMAGENET_MEAN, std: IMAGENET_STD });
  const dims = inputDims(declaredRank(session), 3, size, size);
  const out = await session.run(feedOf(session, new ort.Tensor('float32', input, dims)));
  const raw = pickOutput(out, session, 'predicted_depth', 'depth', 'output').data as unknown as
    Float32Array;
  const side = Math.round(Math.sqrt(raw.length));
  const full = resizePlane(raw, side, side, square.width, square.height);
  if (square.width === rect.width && square.height === rect.height) return full;

  // 画像が細長くて正方形が短辺で頭打ちになると、rect が square に収まり
  // きらないことがある。そのときは端の値で埋める（読み出しははみ出さない）。
  const cut = new Float32Array(rect.width * rect.height);
  const dx = rect.x - square.x;
  const dy = rect.y - square.y;
  for (let y = 0; y < rect.height; y++) {
    const sy = Math.max(0, Math.min(square.height - 1, y + dy));
    for (let x = 0; x < rect.width; x++) {
      const sx = Math.max(0, Math.min(square.width - 1, x + dx));
      cut[y * rect.width + x] = full[sy * square.width + sx] as number;
    }
  }
  return cut;
}

async function runDepth(
  rgba: Uint8ClampedArray,
  grid: number,
  backend: Backend,
  shaderF16: boolean,
): Promise<DepthOutput> {
  const size = 518;
  const session = await depthSession(backend, shaderF16);

  const small = resizeRgba(rgba, grid, grid, size, size);
  const input = toTensorNCHW(small, size, size, { mean: IMAGENET_MEAN, std: IMAGENET_STD });
  // DA3 は [batch, views, 3, H, W] の5階を取る。決め打ちにしない。
  const dims = inputDims(declaredRank(session), 3, size, size);
  const out = await session.run(feedOf(session, new ort.Tensor('float32', input, dims)));

  const depthTensor = pickOutput(out, session, 'predicted_depth', 'depth', 'output');
  const raw = depthTensor.data as unknown as Float32Array;
  const side = Math.round(Math.sqrt(raw.length));
  const full = resizePlane(raw, side, side, grid, grid);

  // intrinsics は [fx 0 cx; 0 fy cy; 0 0 1]。推論解像度での画素単位なので、
  // 作業グリッドの大きさに直す。
  let focalPx: number | null = null;
  let intrinsicsNote = '出力に intrinsics がありません';
  for (const name of session.outputNames) {
    if (!name.toLowerCase().includes('intrinsic')) continue;
    const t = out[name] as ort.Tensor | undefined;
    const d = t?.data as unknown as ArrayLike<number> | undefined;
    if (!d) {
      intrinsicsNote = `${name} のデータを取り出せません`;
      break;
    }
    const fx = Number(d[0]);
    if (d.length >= 5 && Number.isFinite(fx) && fx > 0) {
      focalPx = fx * (grid / size);
      intrinsicsNote = `fx=${fx.toFixed(1)} @${size}px`;
    } else {
      intrinsicsNote = `${name} の値が使えません（長さ ${d.length}、fx=${String(d[0])}）`;
    }
    break;
  }
  // 焦点距離は遠近感そのものを決める。仮定に落ちたときは、なぜそうなったかを
  // 残しておかないと「なんとなく歪んでいる」で終わってしまう。
  console.info(`[PhotoSplat] 深度モデルの出力: ${session.outputNames.join(', ')} / 内部パラメータ: ${intrinsicsNote}`);

  // DA3 は深度そのもの、V2 は逆深度。出力名で見分ける。
  const isInverse = !session.outputNames.some((n) => n.toLowerCase().includes('predicted_depth'));
  return { raw: full, kind: isInverse ? 'inverse-depth' : 'depth', focalPx, session };
}

/**
 * ② のタイルパス（docs/03 §3.4「2パス構成」）。
 *
 * 全体パスは 1024² の写真を 518² に縮めて推論するので、**顔が小さすぎて
 * 構造が出ない**。実写の人物で確かめたところ、全体パスの深度マップは
 * 顔が「のっぺりした楕円」になり、鼻も眼窩も出ていなかった。同じ顔を
 * 切り出して 518² で推論し直すと、鼻筋・眼窩・顎が現れる。
 * これが「人物の立体感が足りない」の主因である。
 *
 * 被写体の外接矩形を 2×2 に分けて推論し、全体パスに重ねる。
 * タイルごとに独自のスケールを持つので、重なりで合わせてから混ぜる
 * （src/pipeline/2-depth.ts）。
 */
async function runDepthTiles(
  session: ort.InferenceSession,
  rgba: Uint8ClampedArray,
  global: Float32Array,
  alpha: Uint8ClampedArray,
  grid: number,
  withShift: boolean,
): Promise<{ depth: Float32Array; tiles: number }> {
  const bbox = subjectBBox(alpha, grid, grid);
  if (!bbox) return { depth: global, tiles: 0 };

  const rects = depthTiles(bbox, grid, grid);
  // 顔だけを詳しく見るタイルを 1 枚足す（docs/09 §V18）。
  const bodyTileSide = rects.length > 0 ? (rects[0] as Rect).width : Math.min(grid, grid);
  const head = headDepthTile(alpha, grid, grid, bodyTileSide);
  if (head) rects.push(head);
  const tiles: DepthTile[] = [];
  for (const rect of rects) {
    if (rect.width < 32 || rect.height < 32) continue;
    try {
      tiles.push({ depth: await inferDepthPatch(session, rgba, grid, grid, rect), rect });
    } catch {
      // 1枚失敗しても全体パスは使える。落とさずに続ける。
    }
  }
  if (tiles.length === 0) return { depth: global, tiles: 0 };

  const fused = fuseDepth(global, tiles, grid, grid, {
    withShift,
    levels: 6,
    feather: Math.max(8, Math.round(Math.min(bbox.width, bbox.height) * 0.08)),
    minConfidence: 0,
    globalWeight: 0.15,
    // 尺度合わせは被写体の中だけで行う。背景を混ぜると倍率が壊れる。
    subject: alpha,
    // タイルは細部だけを担当し、体全体の形は全体パスに任せる。
    detailScalePx: 16,
  });
  return { depth: fused.depth, tiles: tiles.length };
}

/**
 * ⑧ 遮蔽部のインペイント（docs/03 §3.7）。
 *
 * MI-GAN が読めなければ v1 方式（奥側の色の引き伸ばし）に落とす。
 * 落ちても生成は続く。プレビューは元からその色で出ているので、
 * 利用者から見れば「差し替えが起きない」だけになる（docs/03 §3.7 の縮退）。
 */
async function runInpaint(
  rgba: Uint8ClampedArray,
  masked: InpaintMask,
  grid: number,
  backend: Backend,
  shaderF16: boolean,
): Promise<{ plane: Uint8ClampedArray; used: 'mi-gan' | 'stretch' }> {
  const fallback = (): { plane: Uint8ClampedArray; used: 'stretch' } => ({
    plane: stretchFallback(rgba, masked.mask, grid, grid),
    used: 'stretch',
  });
  if (masked.maskedPixels === 0) return fallback();

  try {
    const size = 512;
    const session = await loadModel(
      'mi-gan',
      backend,
      { repo: 'andraniksargsyan/migan', file: 'migan_pipeline_v2.onnx' },
      shaderF16,
    );

    const small = resizeRgba(rgba, grid, grid, size, size);
    const maskSmall = resizePlane(masked.mask, grid, grid, size, size);
    // MI-GAN の pipeline v2 は uint8 の画像とマスクを取る。
    // マスクは「残す＝255 / 描く＝0」の約束なので、こちらの向きと逆になる。
    const image = new Uint8Array(3 * size * size);
    const maskTensor = new Uint8Array(size * size);
    for (let i = 0; i < size * size; i++) {
      image[i] = small[i * 4] as number;
      image[size * size + i] = small[i * 4 + 1] as number;
      image[2 * size * size + i] = small[i * 4 + 2] as number;
      maskTensor[i] = (maskSmall[i] as number) >= 128 ? 0 : 255;
    }

    const feeds: Record<string, ort.Tensor> = {};
    const names = session.inputNames;
    const rank = declaredRank(session);
    feeds[names[0] as string] = new ort.Tensor('uint8', image, inputDims(rank, 3, size, size));
    if (names[1]) feeds[names[1]] = new ort.Tensor('uint8', maskTensor, inputDims(rank, 1, size, size));

    const out = await session.run(feeds);
    const t = pickOutput(out, session, 'output', 'result');
    const data = t.data as unknown as Uint8Array | Float32Array;
    const side = Math.round(Math.sqrt(data.length / 3));

    // NCHW の uint8（または 0..1 の float）で返る。RGBA へ直す。
    const painted = new Uint8ClampedArray(side * side * 4);
    const isFloat = !(data instanceof Uint8Array);
    for (let i = 0; i < side * side; i++) {
      for (let c = 0; c < 3; c++) {
        const v = data[c * side * side + i] as number;
        painted[i * 4 + c] = isFloat ? v * 255 : v;
      }
      painted[i * 4 + 3] = 255;
    }
    const full = resizeRgba(painted, side, side, grid, grid);
    return { plane: compositeInpaint(rgba, full, masked.mask, grid, grid, 2), used: 'mi-gan' };
  } catch {
    return fallback();
  }
}

/**
 * ⑥⑦（サンプリングとスプラット生成）を回すグリッドの一辺（docs/11 §11.6 S4）。
 *
 * **深度は 1024² のままでよい。** 深度モデルの入力が 518² なので、上げても
 * 情報は増えない。増えるのは**色**である。1116×2000 の写真を 1024² に載せると
 * 中身は 571×1024 で、線密度で 1.95 倍を捨てている。docs/03 §3.2 は
 * 「色プレーンは写真そのものであり、ここが品質の天井になる」と書いているのに、
 * その天井を自分で下げていた。
 *
 * **全段を上げない理由。** 実測で CPU が 2.7 倍（5.5s → 14.7s）になり、10 秒の
 * 予算（D4・D13）に入らない。⑥⑦ だけなら +1.3 秒で済む。
 *
 * **1.5 倍で止める理由。** 写真の実寸（この例では 2000）まで上げると質感はもう
 * 少し上がるが（至近の勾配 4.6 → 5.7）、枚数が 536k → 894k、`.splat` が
 * 17MB → 29MB、CPU が +4.3 秒になり、どちらの予算にも入らない。
 *
 * 作業グリッドへの**比**で持つ。軽量プリセット（512²）でも同じ割合で効き、
 * 「軽くしたのに⑥⑦だけ 3 倍重い」ということが起きない。
 */
const FINE_GRID_RATIO = 1.5;

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
  const shaderF16 = opts.shaderF16 ?? true;
  const rawAlpha = await mark('マット推定', () =>
    runMatte(rgba, grid, opts.mode, opts.backend, shaderF16),
  );
  const alpha = await mark('マット後処理', () => refineMatte(rawAlpha, rgba, grid, grid));

  report(0.35, '奥行きを推定しています');
  const depthOut = await mark('深度推定', () => runDepth(rgba, grid, opts.backend, shaderF16));
  const focalPx = depthOut.focalPx ?? assumedFocal(grid);

  // タイルパス。全体パスだけだと顔が「のっぺりした楕円」になる（docs/03 §3.4）。
  let depthRaw = depthOut.raw;
  let tileCount = 0;
  if (opts.depthTiles !== false) {
    report(0.5, '顔まわりの奥行きを詳しく見ています');
    const tiled = await mark('深度タイルパス', () =>
      runDepthTiles(
        depthOut.session,
        rgba,
        depthOut.raw,
        alpha,
        grid,
        depthOut.kind === 'inverse-depth',
      ),
    );
    depthRaw = tiled.depth;
    tileCount = tiled.tiles;
  }

  // 顔の起伏。深度モデルは顔をほぼ平らな楕円として返すので、顔専用の
  // landmark モデルで細部の帯だけを差し替える（docs/09 §V9）。
  let faceApplied = false;
  if (opts.mode === 'person' && depthOut.kind === 'depth') {
    report(0.58, '顔の立体を起こしています');
    const withFace = await mark('顔の起伏', () =>
      applyFaceDepth(
        depthRaw,
        rgba,
        alpha,
        grid,
        focalPx,
        opts.backend,
        shaderF16,
        nativeFacePatchSource(photo, prepared.box),
      ),
    );
    depthRaw = withFace.depth;
    faceApplied = withFace.applied;
  }

  report(0.62, '奥行きを整えています');
  // 局所強調は顔のためのもの。体に掛けるとあり得ない起伏を作る側に働く
  // （docs/11 §11.2）。人物のときだけ顔の箱を渡して、外は 1 倍にする。
  const boostBox = opts.mode === 'person' ? headBoxFromMatte(alpha, grid, grid) : null;
  const calibrated = await mark('深度較正', () =>
    calibrate({
      raw: depthRaw,
      width: grid,
      height: grid,
      alpha,
      kind: depthOut.kind,
      focalPx,
      ...(boostBox ? { faceBox: boostBox, reliefBoostBody: BODY_RELIEF_BOOST } : {}),
    }),
  );

  // 髪や輪郭の半透明画素（0 < α < 0.5）は、深度モデルには背景が混ざって
  // 見えている。その深度をそのまま使うと、髪が後ろへ長く尾を引く。
  // 最も近い不透明画素の深度で置き換える（docs/03 §3.5.3(c)）。
  // にじみの帯はモデルの推論解像度（518²）で決まるので、作業グリッドに
  // 対する割合で置く。1024² なら 12px（v2.6.4、docs/09 §V19）。
  //
  // v2.6.3 まで 8px にしていたが、にじみは 10px あたりまで届いている。
  // 細い部位（脛）では、はみ出した帯が輪郭を内部より **18.7mm 手前**へ
  // 押し出していた。12px にすると −18.7mm → −0.4mm になり、断面の丸みは
  // 参照実装と同じところに収まる（幅比 0.99 → 0.33、参照実装 0.20）。
  const boundaryBand = Math.max(2, Math.round(grid * BOUNDARY_BAND_RATIO));
  const pulled = await mark('境界深度の引き込み', () =>
    pullBoundaryDepthInward(calibrated.depth, alpha, grid, grid, boundaryBand),
  );

  // 法線は実距離で推定する。正規化した値のままだと焦点距離と単位が合わない。
  const span = calibrated.farZ - calibrated.nearZ;
  const plain = new Float32Array(grid * grid);
  for (let i = 0; i < plain.length; i++) {
    plain[i] = calibrated.nearZ + ((pulled[i] as number) / 65535) * span;
  }
  let metric: Float32Array = plain;

  // 四肢に円柱の断面を与える（docs/09 §V20）。深度モデルは腕や脚の横断面を
  // ほとんど平らに返すので、シルエットの横一線の区間から円柱を起こして
  // 大域だけ差し替える。頭は円柱ではないので外す。
  if (opts.mode === 'person') {
    const headBox = headBoxFromMatte(alpha, grid, grid);
    metric = await mark('四肢の丸み', () =>
      applyLimbRoundness(metric, alpha, grid, grid, focalPx, DEFAULT_LIMB_PARAMS, headBox),
    );

    // 服の合わせや裾で深度が手前へ行き過ぎたぶんを詰める（docs/09 §V21）。
    // 大域は触らず、細部の振幅だけを体幅の 0.8% までに収める。
    metric = await mark('起伏の振幅', () =>
      clampReliefAmplitude(metric, alpha, grid, grid, focalPx, DEFAULT_RELIEF_CLAMP, headBox),
    );

    // 服の開口部の中を奥へ置きすぎるのを止める（docs/09 §V24）。
    // 横断面が手前側の凸包から体幅の 10% より奥へ凹むことは、人体では無い。
    metric = await mark('断面の凹み', () =>
      limitCrossSectionDent(metric, alpha, grid, grid, focalPx, DEFAULT_CROSS_SECTION, headBox),
    );
  }

  // 0..1 に直した深度。⑧ のマスクと UI のプレーンはこの形で受け取る。
  const depth01 = new Float32Array(grid * grid);
  for (let i = 0; i < depth01.length; i++) {
    depth01[i] = Math.max(0, Math.min(1, ((metric[i] as number) - calibrated.nearZ) / span));
  }

  // ⑥⑦ は写真の解像度に近いグリッドで回す（docs/11 §11.6 S4）。
  // 元写真の長辺がグリッドを超えているときだけ意味がある。
  const nativeLong = prepared.box.scale > 0 ? Math.round(grid / prepared.box.scale) : grid;
  const fineSize = Math.min(Math.round(grid * FINE_GRID_RATIO), Math.max(grid, nativeLong));
  const fine = fineSize > grid ? fineSize : grid;
  const fineFocal = focalPx * (fine / grid);
  const fineRgba =
    fine > grid
      ? await mark('色の再サンプル', () => prepareImageScaled(photo, prepared.box, fine))
      : rgba;
  const fineAlpha =
    fine > grid
      ? Uint8ClampedArray.from(resizePlane(alpha, grid, grid, fine, fine))
      : alpha;
  const fineMetric = fine > grid ? resizePlane(metric, grid, grid, fine, fine) : metric;
  const fineDepth01 = new Float32Array(fine * fine);
  for (let i = 0; i < fineDepth01.length; i++) {
    fineDepth01[i] = Math.max(0, Math.min(1, ((fineMetric[i] as number) - calibrated.nearZ) / span));
  }

  report(0.72, '面の向きを求めています');
  const normals = await mark('法線推定', () =>
    estimateNormals(fineMetric, fine, fine, fineFocal, span * 0.05),
  );

  report(0.82, 'ガウシアンを配置しています');
  const samplingParams = await mark('サンプリング閾値', () =>
    solveSamplingParams(fineDepth01, fineRgba, fineAlpha, fine, fine, opts.reduction),
  );
  const cells = await mark('適応サンプリング', () =>
    adaptiveSample(fineDepth01, fineRgba, fineAlpha, fine, fine, samplingParams),
  );

  report(0.9, '厚みをつけています');
  const buildParams: BuildParams = { ...DEFAULT_BUILD_PARAMS, ...opts.params };
  // 厚みマップは背面シェルにしか使わない。作らないなら計算もしない。
  const thickness = buildParams.backShell
    ? await mark('厚みマップ', () =>
        thicknessMap(fineAlpha, fine, fine, {
          maxThickness: BACK_SHELL_THICKNESS,
          profile: 'ellipsoid',
        }),
      )
    : null;
  const camera = {
    cells,
    normals,
    width: fine,
    height: fine,
    focalPx: fineFocal,
    nearZ: calibrated.nearZ,
    farZ: calibrated.farZ,
  };

  // まずインペイント無しで1枚作って見せる（docs/03 §3.1 のプレビュー）。
  // 待たせるより、粗くても先に立体を出すほうが体感が速い。
  const preview = await mark('スプラット組み立て', () =>
    buildSplats(camera, fineRgba, fineAlpha, thickness, null, buildParams),
  );
  opts.onPreview?.(preview);

  let build = preview;
  let inpaintUsed: 'mi-gan' | 'stretch' | 'skipped' = 'skipped';

  if (opts.inpaint !== false) {
    report(0.94, '隠れていた部分を描いています');
    const masked = await mark('インペイントのマスク', () =>
      buildInpaintMask(depth01, alpha, grid, grid, focalPx, calibrated.nearZ, calibrated.farZ),
    );
    if (masked.maskedPixels > 0) {
      const painted = await mark('インペイント', () =>
        runInpaint(rgba, masked, grid, opts.backend, shaderF16),
      );
      inpaintUsed = painted.used;
      // スカートの色だけが変わるので、組み立て直す。⑧ は作業グリッドで回すので、
      // ⑥ のグリッドに合わせて引き伸ばす。
      const plane =
        fine > grid ? resizeRgba(painted.plane, grid, grid, fine, fine) : painted.plane;
      build = await mark('スカート色の差し替え', () =>
        buildSplats(camera, fineRgba, fineAlpha, thickness, null, buildParams, plane),
      );
    }
  }

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
      metricDepth: calibrated.metric,
      depthToHeight: calibrated.depthToHeight,
      depthToWidth: calibrated.depthToWidth,
      depthTiles: tileCount,
      faceRelief: faceApplied,
      inpaint: inpaintUsed,
    },
  };
}
