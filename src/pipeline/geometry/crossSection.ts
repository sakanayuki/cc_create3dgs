/**
 * 横断面が人体としてあり得ないほど凹むのを止める（docs/09 §V24）。
 *
 * **なぜ要るのか。** 深度モデルは、服の合わせのような**開口部**の中を大きく
 * 奥へ置く。実写（カーディガンの V 字の開き）では、開きの中の肌が両側の生地より
 * **体幅の 21%** も奥にあった。人の胸の断面にそんな溝は無い。見る側には
 * 「みぞおちのあたりが異様にへこんでいる」と映る。
 *
 * **物差しは横断面の凸包。** 正面から見て横一線に切った断面を取り、手前側の
 * 凸包（下側凸包）からどれだけ奥へ外れるかを測る。体幅で割った値を参照実装
 * （SHARP）と比べると:
 *
 * | | 胸〜みぞおち | 腹〜腰 |
 * |---|---|---|
 * | 参照実装 中央 / 最大 | 0.041 / **0.114** | 0.108 / 0.142 |
 * | 私たち 中央 / 最大 | 0.085 / **0.214** | 0.059 / 0.125 |
 *
 * **腹〜腰は私たちのほうが浅い。** 深すぎるのは胸だけで、そこが開口部である。
 * 上限を参照実装の水準（0.10）に置けば、直したい所だけが動く。実測で、開口部の
 * 無い別の 2 枚では 90 パーセンタイルが 0.026 → 0.027、0.111 → 0.107 と
 * ほとんど動かない。**直すところが無ければ何もしない。**
 *
 * **凸包そのものへは戻さない。** 人の断面には本物の凹み（胸の谷、脇の下、
 * 膝の裏）がある。上限の半分までは素通しで、そこから上限へ tanh で寄せ、
 * 上限で頭打ちにする。
 * `clampReliefAmplitude`（§V21）と同じ考え方を、尺度ではなく**断面の形**に
 * 当てたものである。
 *
 * **顔は外す。** 眼窩と鼻の脇は本物の深い凹みで、体の物差しでは測れない。
 */
import { SUBJECT_ALPHA } from '../1-matte';
import type { Rect } from '../0-preprocess';

export interface CrossSectionParams {
  /** 許す凹みの深さ（被写体の外接幅に対する割合）。 */
  readonly maxDentRatio: number;
  /** これより短い区間は見ない（指、髪の房）。 */
  readonly minRunPx: number;
  /** 補正を縦に均す半径（画素）。行ごとの当てはめが横縞にならないように。 */
  readonly smoothRows: number;
}

export const DEFAULT_CROSS_SECTION: CrossSectionParams = {
  maxDentRatio: 0.10,
  minRunPx: 40,
  smoothRows: 6,
};

/**
 * 横断面の凹みを上限まで詰める。
 *
 * @param depth   実寸深度（大きいほど奥）。破壊しない。
 * @param exclude 触らない四角（頭の箱）。
 */
export function limitCrossSectionDent(
  depth: ArrayLike<number>,
  alpha: ArrayLike<number>,
  width: number,
  height: number,
  focalPx: number,
  params: CrossSectionParams = DEFAULT_CROSS_SECTION,
  exclude: Rect | null = null,
): Float32Array {
  const out = Float32Array.from(depth as ArrayLike<number>);
  if (focalPx <= 0 || params.maxDentRatio <= 0) return out;

  let xlo = width;
  let xhi = -1;
  let zsum = 0;
  let zcount = 0;
  for (let i = 0; i < out.length; i++) {
    if ((alpha[i] as number) < SUBJECT_ALPHA) continue;
    const x = i % width;
    if (x < xlo) xlo = x;
    if (x > xhi) xhi = x;
    zsum += depth[i] as number;
    zcount++;
  }
  if (xhi < xlo || zcount === 0) return out;
  const widthMetric = ((xhi - xlo + 1) * (zsum / zcount)) / focalPx;
  const limit = params.maxDentRatio * widthMetric;
  if (!(limit > 0)) return out;
  // 素通しにする深さ。ここから上限へなだらかに寄せ、上限で頭打ちにする。
  const knee = limit * 0.5;

  const fix = new Float32Array(width * height);
  const touched = new Uint8Array(width * height);
  const hull = new Float64Array(width);
  const stack = new Int32Array(width);

  for (let y = 0; y < height; y++) {
    if (exclude && y >= exclude.y && y < exclude.y + exclude.height) continue;
    const row = y * width;
    let x = 0;
    while (x < width) {
      if ((alpha[row + x] as number) < SUBJECT_ALPHA) {
        x++;
        continue;
      }
      let end = x;
      while (end + 1 < width && (alpha[row + end + 1] as number) >= SUBJECT_ALPHA) end++;
      if (end - x + 1 >= params.minRunPx) {
        lowerHull(depth, row, x, end, hull, stack);
        for (let i = x; i <= end; i++) {
          const dent = (depth[row + i] as number) - (hull[i] as number);
          if (dent <= knee) continue;
          // 上限の半分までは素通し、そこから上限へ tanh で寄せる。
          const packed = knee + (limit - knee) * Math.tanh((dent - knee) / (limit - knee));
          fix[row + i] = packed - dent;
          touched[row + i] = 1;
        }
      }
      x = end + 1;
    }
  }

  const smooth = smoothColumns(fix, touched, width, height, params.smoothRows);
  for (let i = 0; i < out.length; i++) {
    if (touched[i] === 1) out[i] = (depth[i] as number) + (smooth[i] as number);
  }
  return out;
}

/**
 * 区間 [x0, x1] の手前側の凸包を、各画素で評価して `hull` に書く。
 *
 * 深度は「大きいほど奥」なので、手前側の包絡は**下側**凸包になる。
 */
function lowerHull(
  depth: ArrayLike<number>,
  row: number,
  x0: number,
  x1: number,
  hull: Float64Array,
  stack: Int32Array,
): void {
  let top = 0;
  for (let i = x0; i <= x1; i++) {
    const zi = depth[row + i] as number;
    while (top >= 2) {
      const a = stack[top - 2] as number;
      const b = stack[top - 1] as number;
      const za = depth[row + a] as number;
      const zb = depth[row + b] as number;
      // b が a→i の線より上（奥）なら b は凸包に要らない
      if ((zb - za) * (i - a) >= (zi - za) * (b - a)) top--;
      else break;
    }
    stack[top++] = i;
  }
  for (let k = 0; k + 1 < top; k++) {
    const a = stack[k] as number;
    const b = stack[k + 1] as number;
    const za = depth[row + a] as number;
    const zb = depth[row + b] as number;
    const span = b - a;
    for (let i = a; i <= b; i++) hull[i] = za + ((zb - za) * (i - a)) / span;
  }
  if (top === 1) hull[stack[0] as number] = depth[row + (stack[0] as number)] as number;
}

/** 補正量を縦に均す。触っていない画素は混ぜない。 */
function smoothColumns(
  fix: Float32Array,
  touched: Uint8Array,
  width: number,
  height: number,
  radius: number,
): Float32Array {
  if (radius < 1) return fix;
  const out = new Float32Array(fix.length);
  for (let x = 0; x < width; x++) {
    for (let y = 0; y < height; y++) {
      const i = y * width + x;
      if (touched[i] === 0) continue;
      let sum = 0;
      let count = 0;
      for (let k = -radius; k <= radius; k++) {
        const yy = y + k;
        if (yy < 0 || yy >= height) continue;
        const j = yy * width + x;
        if (touched[j] === 0) continue;
        sum += fix[j] as number;
        count++;
      }
      out[i] = count > 0 ? sum / count : (fix[i] as number);
    }
  }
  return out;
}
