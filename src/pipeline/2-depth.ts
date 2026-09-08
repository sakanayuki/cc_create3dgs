/**
 * ② 深度推定の2パス融合（docs/03 §3.4）。
 *
 * 推論そのもの（ORT の呼び出し）はここには置かない。ここが受け持つのは
 * 「全体パス1枚 + タイルパス4枚を、1枚の深度マップにまとめる」ところ。
 * モデルを差し替えても（DA3-S / V2-S）この段は変わらない。
 *
 * 難しいのは、各推論が**自分だけのスケールを持つ**こと。単眼深度モデルは
 * 入力ごとに独立に正規化するので、タイルの「1.0」と全体の「1.0」は違う。
 * そのまま貼ると継ぎ目で段差になる。重なり領域で合わせてから混ぜる。
 */
import type { Rect } from './0-preprocess';
import { blendLaplacian } from './geometry/pyramid';

export interface AffineFit {
  /** 倍率。 */
  readonly a: number;
  /** 切片。DA3 のように深度を直接出すモデルでは 0 に固定する。 */
  readonly b: number;
  /** フィットに使った画素数。少なすぎる結果は信用しない。 */
  readonly count: number;
  /** 残差の二乗平均平方根。合わせきれなかった量。 */
  readonly rmse: number;
}

/**
 * `target ≈ a·source + b` を最小二乗で解く。
 *
 * @param withShift 切片を許すか。V2 系の逆深度は `a·d + b` が要るが、
 *                  DA3 のように深度を直接出すモデルでは切片は不要で、
 *                  許すとむしろ形が歪む（docs/03 §3.4 の1点目と同じ理由）。
 */
export function fitAffine(
  source: ArrayLike<number>,
  target: ArrayLike<number>,
  mask: ArrayLike<number>,
  withShift = true,
): AffineFit {
  let n = 0;
  let sx = 0;
  let sy = 0;
  let sxx = 0;
  let sxy = 0;
  for (let i = 0; i < mask.length; i++) {
    if ((mask[i] as number) <= 0) continue;
    const x = source[i] as number;
    const y = target[i] as number;
    if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
    n++;
    sx += x;
    sy += y;
    sxx += x * x;
    sxy += x * y;
  }
  if (n < 2) return { a: 1, b: 0, count: n, rmse: Infinity };

  let a: number;
  let b: number;
  if (withShift) {
    const den = n * sxx - sx * sx;
    if (Math.abs(den) < 1e-12) return { a: 1, b: 0, count: n, rmse: Infinity };
    a = (n * sxy - sx * sy) / den;
    b = (sy - a * sx) / n;
  } else {
    if (Math.abs(sxx) < 1e-12) return { a: 1, b: 0, count: n, rmse: Infinity };
    a = sxy / sxx;
    b = 0;
  }

  let se = 0;
  for (let i = 0; i < mask.length; i++) {
    if ((mask[i] as number) <= 0) continue;
    const x = source[i] as number;
    const y = target[i] as number;
    if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
    const r = a * x + b - y;
    se += r * r;
  }
  return { a, b, count: n, rmse: Math.sqrt(se / n) };
}

export interface DepthTile {
  /** タイルの推論結果。長さ rect.width × rect.height。 */
  readonly depth: ArrayLike<number>;
  /** 画像内でのタイルの位置。 */
  readonly rect: Rect;
  /** モデルが返す信頼度（DA3 の `confidence`）。無ければ省略。 */
  readonly confidence?: ArrayLike<number>;
}

export interface FuseParams {
  /** 切片を許すか。DA3 は false、V2 は true。 */
  readonly withShift: boolean;
  /** ラプラシアンブレンドの段数。 */
  readonly levels: number;
  /**
   * タイルの端で重みを 0 に落とす幅（画素）。
   * 端まで重み 1 のまま混ぜると、タイル境界に線が出る。
   */
  readonly feather: number;
  /** 信頼度がこの値未満の画素はタイル側の重みを 0 にする。 */
  readonly minConfidence: number;
  /**
   * 被写体マスク（α）。渡すと、タイルと全体パスの尺度合わせに
   * **被写体の画素だけ**を使う。
   *
   * 渡さないと背景も混ざる。単眼深度は遠景に安定した値を返さないので、
   * 切り出し方が違えば空やプールの深度は別物になる。実写で測ると、
   * 頭のタイルは 60% が背景で、背景だけで合わせたときの残差は被写体の
   * 5 倍あった。その背景に引かれて倍率が 1.05 → 1.22 までずれ、
   * 被写体の深度が丸ごと押し出されていた。
   */
  readonly subject?: ArrayLike<number>;
  /** 被写体とみなす α の下限。 */
  readonly subjectThreshold?: number;
  /**
   * タイルが担当する構造の上限（画素）。これより粗い成分は全体パスの
   * ものをそのまま使う。0 なら全帯域をタイルと混ぜる（v2.2 までの挙動）。
   *
   * タイルは全体パスと違う切り出し方で推論されるので、低周波（＝体全体の
   * 形）が一致しない。そこを混ぜると、食い違いがそのまま奥行きの歪みに
   * なる。実写で測ると、タイルパスは被写体の 奥行き÷身長 を 0.506 → 0.760
   * と 50% も膨らませていた（理想は 0.36 前後）。
   *
   * タイルパスの目的は顔の凹凸を出すこと、つまり**細部**である。大域は
   * 全体パスが正しく、そこはタイルに触らせない。
   */
  readonly detailScalePx?: number;
  /**
   * 全体パスの重み。タイルは最大 1 なので、これを小さくするほど
   * タイルの細部が残る。0 にしてはいけない。タイルが1枚も掛からない
   * 領域（背景など）で分母が 0 になる。
   */
  readonly globalWeight: number;
}

export const DEFAULT_FUSE_PARAMS: FuseParams = {
  withShift: false,
  levels: 6,
  feather: 24,
  minConfidence: 0.2,
  // タイル内部ではタイルが約 87% を占める。全体パスと半々にすると、
  // タイルパスに 1.6 秒かけて得た細部が半分に薄まってしまう。
  globalWeight: 0.15,
  // 実写で振って決めた。16px より粗い成分を全体パスに任せると、奥行きの
  // 膨張が最小（奥行÷身長 0.837 → 0.721）になり、しかも顔の起伏が
  // 2.1 倍（87.8 → 183.2）になる。低周波の食い違いが細部を潰していた。
  detailScalePx: 16,
};

/** タイル内での位置に応じた羽根（端で 0、内側で 1）。 */
function featherWeight(x: number, y: number, rect: Rect, feather: number): number {
  if (feather <= 0) return 1;
  const dx = Math.min(x, rect.width - 1 - x);
  const dy = Math.min(y, rect.height - 1 - y);
  const d = Math.min(dx, dy);
  if (d >= feather) return 1;
  const t = d / feather;
  // smoothstep。線形だと羽根の端で重みの傾きが不連続になり、
  // そこがうっすら線に見える。
  return t * t * (3 - 2 * t);
}

export interface FusionResult {
  /** 融合後の深度。長さ width×height。 */
  readonly depth: Float32Array;
  /** 各タイルの整合結果。継ぎ目が残るときの原因追跡に使う。 */
  readonly fits: AffineFit[];
}

/**
 * 全体パスとタイルパスを1枚にまとめる。
 *
 * 手順は3つ。
 *   1. 各タイルを、重なり領域で全体パスに合わせる（最小二乗）
 *   2. 羽根と信頼度から重みを作る
 *   3. ラプラシアンブレンドで混ぜる
 *
 * 2で作った重みで単純加重平均しないのは、タイルと全体で低周波の食い違いが
 * 残るため。羽根の幅を広げれば段差は消えるが、今度はタイルが持ってきた
 * 細部まで薄まる。周波数帯ごとに混ぜ幅を変えるのがラプラシアンブレンド。
 *
 * @param global 全体パスの深度。長さ width×height。
 */
export function fuseDepth(
  global: ArrayLike<number>,
  tiles: readonly DepthTile[],
  width: number,
  height: number,
  params: FuseParams = DEFAULT_FUSE_PARAMS,
): FusionResult {
  const n = width * height;
  const sources: ArrayLike<number>[] = [global];
  const weights: ArrayLike<number>[] = [new Float32Array(n).fill(params.globalWeight)];
  const fits: AffineFit[] = [];

  for (const tile of tiles) {
    const { rect } = tile;
    // タイルを全体と同じ座標系に置いた板を作る。重なりの外は全体パスの値で
    // 埋める。0 で埋めると、そこがラプラシアンピラミッドで巨大な段差になり、
    // 重み 0 でも隣の層へにじんで縞が出る。
    const placed = Float32Array.from(global as ArrayLike<number>);
    const weight = new Float32Array(n);
    const overlap = new Float32Array(n);

    for (let y = 0; y < rect.height; y++) {
      const gy = rect.y + y;
      if (gy < 0 || gy >= height) continue;
      for (let x = 0; x < rect.width; x++) {
        const gx = rect.x + x;
        if (gx < 0 || gx >= width) continue;
        const ti = y * rect.width + x;
        const gi = gy * width + gx;
        const v = tile.depth[ti] as number;
        if (!Number.isFinite(v)) continue;

        let w = featherWeight(x, y, rect, params.feather);
        if (tile.confidence) {
          const c = tile.confidence[ti] as number;
          if (c < params.minConfidence) w = 0;
          else w *= c;
        }
        placed[gi] = v;
        weight[gi] = w;
        // フィットには羽根の内側だけを使う。端は信用しない。
        // 被写体マスクがあれば、さらに被写体の中だけに絞る。
        const inSubject =
          params.subject === undefined ||
          (params.subject[gi] as number) >= (params.subjectThreshold ?? 128);
        overlap[gi] = w > 0.5 && inSubject ? 1 : 0;
      }
    }

    const fit = fitAffine(placed, global, overlap, params.withShift);
    fits.push(fit);

    // 合わせきれないタイルは捨てる。無理に混ぜると全体を壊す。
    if (!Number.isFinite(fit.a) || fit.count < 64) continue;
    for (let i = 0; i < n; i++) {
      if ((weight[i] as number) > 0) placed[i] = fit.a * (placed[i] as number) + fit.b;
      else placed[i] = global[i] as number;
    }

    sources.push(placed);
    weights.push(weight);
  }

  // detailScalePx より粗い層は全体パス（sources[0]）に固定する。
  // 層 l の構造の大きさはおおよそ 2^l 画素。
  const detail = params.detailScalePx ?? 0;
  const keepFrom = detail > 0 ? Math.max(0, Math.ceil(Math.log2(Math.max(detail, 1)))) : 0;
  const referenceLevels = detail > 0 ? Math.max(0, params.levels + 1 - keepFrom) : 0;

  return {
    depth: blendLaplacian(sources, weights, width, height, params.levels, referenceLevels),
    fits,
  };
}
