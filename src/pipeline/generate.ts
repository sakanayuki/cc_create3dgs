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
import { refineMatte } from './1-matte';
import {
  depthTiles,
  expandToSquare,
  prepareImage,
  subjectBBox,
  WORKING_GRID,
  type Letterbox,
  type Rect,
} from './0-preprocess';
import { fuseDepth, type DepthTile } from './2-depth';
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
  shaderF16: boolean,
): Promise<Uint8ClampedArray> {
  const size = mode === 'person' ? 512 : 1024;
  const session =
    mode === 'person'
      ? await loadModel(
          'modnet',
          backend,
          { repo: 'Xenova/modnet', file: 'onnx/model_uint8.onnx' },
          shaderF16,
        )
      : await loadModel(
          'isnet-general',
          backend,
          { repo: 'imgly/isnet-general-onnx', file: 'onnx/model.onnx' },
          shaderF16,
        );

  const small = resizeRgba(rgba, grid, grid, size, size);
  const input = toTensorNCHW(small, size, size, { mean: [0.5, 0.5, 0.5], std: [0.5, 0.5, 0.5] });
  const dims = inputDims(declaredRank(session), 3, size, size);
  const out = await session.run(feedOf(session, new ort.Tensor('float32', input, dims)));
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

  report(0.62, '奥行きを整えています');
  const calibrated = await mark('深度較正', () =>
    calibrate({
      raw: depthRaw,
      width: grid,
      height: grid,
      alpha,
      kind: depthOut.kind,
      focalPx,
    }),
  );

  // 髪や輪郭の半透明画素（0 < α < 0.5）は、深度モデルには背景が混ざって
  // 見えている。その深度をそのまま使うと、髪が後ろへ長く尾を引く。
  // 最も近い不透明画素の深度で置き換える（docs/03 §3.5.3(c)）。
  // にじみの帯はモデルの推論解像度（518²）で決まるので、作業グリッドに
  // 対する割合で置く。1024² なら 8px。実測では 8px で最奥張り付きが
  // 56% → 6.6% まで落ちる。
  const boundaryBand = Math.max(2, Math.round(grid * 0.008));
  const pulled = await mark('境界深度の引き込み', () =>
    pullBoundaryDepthInward(calibrated.depth, alpha, grid, grid, boundaryBand),
  );

  // 0..1 に直した深度。以降の工程はこの形で受け取る。
  const depth01 = new Float32Array(grid * grid);
  for (let i = 0; i < depth01.length; i++) depth01[i] = (pulled[i] as number) / 65535;
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
    thicknessMap(alpha, grid, grid, { maxThickness: BACK_SHELL_THICKNESS, profile: 'ellipsoid' }),
  );

  const buildParams: BuildParams = { ...DEFAULT_BUILD_PARAMS, ...opts.params };
  const camera = {
    cells,
    normals,
    width: grid,
    height: grid,
    focalPx,
    nearZ: calibrated.nearZ,
    farZ: calibrated.farZ,
  };

  // まずインペイント無しで1枚作って見せる（docs/03 §3.1 のプレビュー）。
  // 待たせるより、粗くても先に立体を出すほうが体感が速い。
  const preview = await mark('スプラット組み立て', () =>
    buildSplats(camera, rgba, alpha, thickness, null, buildParams),
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
      // スカートの色だけが変わるので、組み立て直す。
      build = await mark('スカート色の差し替え', () =>
        buildSplats(camera, rgba, alpha, thickness, null, buildParams, painted.plane),
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
      inpaint: inpaintUsed,
    },
  };
}
