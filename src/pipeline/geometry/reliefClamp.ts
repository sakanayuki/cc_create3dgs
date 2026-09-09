/**
 * 細部の起伏を、人体としてあり得る振幅に収める（docs/09 §V21、docs/11 §11.6 S2）。
 *
 * **なぜ要るのか。** 単眼深度モデルは、服の合わせ・裾・紐・編み目のような
 * **被写体の内側の段差**で手前へ行き過ぎる。`pullBoundaryDepthInward` は
 * シルエットの縁しか見ないので、ここには効かない。至近で見ると**太い白い
 * ロープ**になり、正面から見ると**胴と裾に縦の筋**が走る。
 *
 * 実測（立ち姿、身長 1 に正規化）。「横断面の局所的な飛び出し」は、行ごとの
 * 深度から区間幅 1/4 の箱平均を引いて、手前へ外れた量を体幅で割ったもの:
 *
 * | | 飛び出し 99% | 90% |
 * |---|---|---|
 * | 参照実装（SHARP） | 4.15% | 2.53% |
 * | この段の前 | 9.11% | 5.00% |
 * | この段の後 | **5.41%** | **3.00%** |
 *
 * **どう決めたか。** 最初は「色の細部と深度の細部が一致するなら模様の写り込み
 * だ」と考えて相関を測ったが、胴での相関は +0.04〜+0.13 しかなく、**この説明は
 * 支持されなかった**ので採らない。採る根拠は**振幅**である。半径 16 画素より
 * 細かい成分だけで 43mm あり、被写体の奥行きの幅 329mm の 13% を占めていた。
 * 布の皺がこの大きさになることはない。そこで、
 *
 * - **大域**（`trendRatio` 倍の箱平均）は触らない。体の形はここに入る。
 * - **細部**（大域からの差）は、体幅の `maxDetailRatio` 倍までに収める。
 *   上限の半分までは素通しで、そこから上を tanh でなだらかに詰める。
 *
 * こうすると、本物の皺（振幅が小さい）は残り、行き過ぎ（振幅が大きい）だけが
 * 詰まる。単に平滑化するのと違い、**残す細部の大きさが決まっている**。
 *
 * SHARP の `L_grad` は同じ場所の α を落として**見えなくする**（docs/10 の P5）。
 * 私たちは**正しい位置へ戻す**。戻せるなら戻すほうがよい（docs/09 §V19）。
 */
import { SUBJECT_ALPHA } from '../1-matte';
import type { Rect } from '../0-preprocess';

export interface ReliefClampParams {
  /** 大域を取る箱平均の半径（被写体の外接幅に対する割合）。 */
  readonly trendRatio: number;
  /** 残す細部の振幅の上限（被写体の外接幅に対する割合）。 */
  readonly maxDetailRatio: number;
  /**
   * 先に掛ける雑音均しの半径（画素）。
   *
   * 深度の画素単位のざらつきは法線を荒らし、`buildSplats` の傾き補正を通して
   * **スプラットを太らせる**。実測で半径÷点間隔が 1.04 → 0.87、至近の勾配が
   * 2.69 → 3.54 になった。0 で無効。
   */
  readonly denoiseRadius: number;
}

export const DEFAULT_RELIEF_CLAMP: ReliefClampParams = {
  trendRatio: 0.10,
  maxDetailRatio: 0.008,
  denoiseRadius: 2,
};

/**
 * 細部の振幅を体らしい範囲に収める。
 *
 * @param depth   実寸深度（大きいほど奥）。破壊しない。
 * @param focalPx 作業グリッドでの焦点距離（画素）。
 * @param exclude 触らない四角（頭の箱）。顔は `applyFaceRelief` の担当で、
 *                鼻や唇は体の物差しでは「行き過ぎ」に見えてしまう。
 */
export function clampReliefAmplitude(
  depth: ArrayLike<number>,
  alpha: ArrayLike<number>,
  width: number,
  height: number,
  focalPx: number,
  params: ReliefClampParams = DEFAULT_RELIEF_CLAMP,
  exclude: Rect | null = null,
): Float32Array {
  const out = Float32Array.from(depth as ArrayLike<number>);
  if (focalPx <= 0) return out;

  const mask = new Uint8Array(width * height);
  let xlo = width;
  let xhi = -1;
  let zsum = 0;
  let zcount = 0;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const i = y * width + x;
      if ((alpha[i] as number) < SUBJECT_ALPHA) continue;
      mask[i] = 1;
      if (x < xlo) xlo = x;
      if (x > xhi) xhi = x;
      zsum += depth[i] as number;
      zcount++;
    }
  }
  if (xhi < xlo || zcount === 0) return out;

  const widthPx = xhi - xlo + 1;
  // 体幅（実寸）。1 画素の実寸は 平均の奥行き ÷ 焦点距離。
  const widthMetric = (widthPx * (zsum / zcount)) / focalPx;
  if (!(widthMetric > 0)) return out;

  const inExcluded = (x: number, y: number): boolean =>
    exclude !== null &&
    x >= exclude.x &&
    x < exclude.x + exclude.width &&
    y >= exclude.y &&
    y < exclude.y + exclude.height;

  // ① 画素単位の雑音を均す。**顔も含めて被写体全体に掛ける**。
  //
  // `exclude` は②の振幅の上限だけを外すための四角である。①は別の目的で、
  // 深度モデルの画素単位のざらつきを落として法線を整えるもの。顔だけ外すと、
  // 顔は較正の局所強調（体の 3 倍）を通ったざらつきをそのまま抱えることに
  // なり、至近で見たときにいちばん荒れる場所になる（docs/09 §V23）。
  const denoise = Math.round(params.denoiseRadius);
  if (denoise >= 1) {
    const blurred = maskedBoxBlur(out, mask, width, height, denoise);
    for (let i = 0; i < out.length; i++) {
      if (mask[i] === 1) out[i] = blurred[i] as number;
    }
  }

  // ② 細部の振幅を詰める。
  const limit = params.maxDetailRatio * widthMetric;
  const trendRadius = Math.round(params.trendRatio * widthPx);
  if (!(limit > 0) || trendRadius < 1) return out;
  const knee = limit * 0.5;
  const trend = maskedBoxBlur(out, mask, width, height, trendRadius);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const i = y * width + x;
      if (mask[i] === 0 || inExcluded(x, y)) continue;
      const base = trend[i] as number;
      const detail = (out[i] as number) - base;
      const size = Math.abs(detail);
      if (size <= knee) continue;
      const packed = knee + (limit - knee) * Math.tanh((size - knee) / (limit - knee));
      out[i] = base + Math.sign(detail) * packed;
    }
  }
  return out;
}

/** 被写体の中だけを混ぜる箱平均（分離可能、走査ごとに窓を転がす）。 */
function maskedBoxBlur(
  src: ArrayLike<number>,
  mask: Uint8Array,
  width: number,
  height: number,
  radius: number,
): Float32Array {
  const rowSum = new Float64Array(width * height);
  const rowCount = new Float64Array(width * height);
  for (let y = 0; y < height; y++) {
    const row = y * width;
    let sum = 0;
    let count = 0;
    for (let x = 0; x <= radius && x < width; x++) {
      if (mask[row + x] === 1) {
        sum += src[row + x] as number;
        count++;
      }
    }
    for (let x = 0; x < width; x++) {
      rowSum[row + x] = sum;
      rowCount[row + x] = count;
      const add = x + radius + 1;
      const drop = x - radius;
      if (add < width && mask[row + add] === 1) {
        sum += src[row + add] as number;
        count++;
      }
      if (drop >= 0 && mask[row + drop] === 1) {
        sum -= src[row + drop] as number;
        count--;
      }
    }
  }

  const out = new Float32Array(width * height);
  for (let x = 0; x < width; x++) {
    let sum = 0;
    let count = 0;
    for (let y = 0; y <= radius && y < height; y++) {
      sum += rowSum[y * width + x] as number;
      count += rowCount[y * width + x] as number;
    }
    for (let y = 0; y < height; y++) {
      const i = y * width + x;
      out[i] = count > 0 ? sum / count : (src[i] as number);
      const add = y + radius + 1;
      const drop = y - radius;
      if (add < height) {
        sum += rowSum[add * width + x] as number;
        count += rowCount[add * width + x] as number;
      }
      if (drop >= 0) {
        sum -= rowSum[drop * width + x] as number;
        count -= rowCount[drop * width + x] as number;
      }
    }
  }
  return out;
}
