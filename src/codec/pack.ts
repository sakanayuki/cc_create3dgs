/**
 * 属性のパッキング（docs/04 §4.4 のビット割当）。
 *
 * WGSL 側（src/render/wgsl/common.wgsl）の復号と対になる。
 * ここが両者の唯一の実装なので、往復の一致は単体テストで担保する。
 *
 * ビット割当の根拠は「その属性の誤差が最終画像に何ピクセル分の誤差として現れるか」で、
 * 全属性を1画面ピクセル以下に揃えてある。1024² では 1px ≈ 1.0mm（被写体1m想定）。
 */

// --- 法線: 八面体写像 16bit × 2 -------------------------------------------
// 単位球面を2Dに写す、歪みの最も少ない写像。8bit×2 でも角度誤差 < 0.9° だが、
// 画像プレーンに載せない素の配列表現では 16bit×2 にしても総量に効かないので余裕を取る。

/** 単位ベクトルを八面体写像で u32 に詰める。 */
export function encodeOct(x: number, y: number, z: number): number {
  const l = Math.abs(x) + Math.abs(y) + Math.abs(z);
  if (l < 1e-20) return encodeOctPair(0, 0);
  let px = x / l;
  let py = y / l;
  if (z < 0) {
    const ox = (1 - Math.abs(py)) * (px >= 0 ? 1 : -1);
    const oy = (1 - Math.abs(px)) * (py >= 0 ? 1 : -1);
    px = ox;
    py = oy;
  }
  return encodeOctPair(px, py);
}

function encodeOctPair(px: number, py: number): number {
  const qx = Math.round(((clamp(px, -1, 1) + 1) / 2) * 65535) & 0xffff;
  const qy = Math.round(((clamp(py, -1, 1) + 1) / 2) * 65535) & 0xffff;
  return ((qy << 16) | qx) >>> 0;
}

/** WGSL の decodeOct と同じ復号。テストで往復を確かめるために JS 側にも持つ。 */
export function decodeOct(packed: number): [number, number, number] {
  const ex = ((packed & 0xffff) / 65535) * 2 - 1;
  const ey = ((packed >>> 16) / 65535) * 2 - 1;
  let nx = ex;
  let ny = ey;
  const nz = 1 - Math.abs(ex) - Math.abs(ey);
  const t = Math.max(-nz, 0);
  nx += nx >= 0 ? -t : t;
  ny += ny >= 0 ? -t : t;
  const l = Math.hypot(nx, ny, nz) || 1;
  return [nx / l, ny / l, nz / l];
}

// --- スケール: 対数量子化 --------------------------------------------------
// スケールは対数正規分布に従うので、log 空間の一様量子化で相対誤差が一定になる。
// 線形量子化では小さいサーフェルの精度が壊滅する。

export interface LogRange {
  readonly min: number;
  readonly max: number;
}

/**
 * 既定のスケールレンジ（フォールバック）。
 *
 * 相対誤差はレンジ比 R とビット数 b だけで決まる:
 *     相対誤差 = exp(ln(R) / (2 · (2^b − 1))) − 1
 *
 * 実際に出うるスケールは、1024²・画角55°（focalPx ≈ 984）で
 *   下限 1.4·0.68/984          ≈ 9.7e-4  （最も手前・傾きなし・1px セル）
 *   上限 1.4·1.32/984·3.3·8    ≈ 5.0e-2  （最も奥・傾き最大・8px セル）
 * つまり R ≈ 51、8bit で 0.77%。サーフェル半径 2px に対し 0.016px の誤差で、
 * 1画面ピクセルに対して無視できる。
 *
 * ただし実運用では scaleRangeFor() で書き出しごとに範囲を締める。そのほうが
 * R が小さくなり、同じビット数で精度が上がる。
 */
export const DEFAULT_SCALE_RANGE: LogRange = { min: 8e-4, max: 6e-2 };

/** レンジ比とビット数から相対誤差の上界を返す。設計値の検算に使う。 */
export function logQuantRelativeError(range: LogRange, bits: number): number {
  const ratio = range.max / range.min;
  return Math.exp(Math.log(ratio) / (2 * ((1 << bits) - 1))) - 1;
}

/**
 * 実データからスケールレンジを決める。
 *
 * 外れ値（マットの縁に残る極端なサーフェル）でレンジが広がると、
 * 大多数の精度が犠牲になる。パーセンタイルで刈ってから少し余裕を足す。
 */
export function scaleRangeFor(values: ArrayLike<number>, marginRatio = 1.15): LogRange {
  const finite: number[] = [];
  for (let i = 0; i < values.length; i++) {
    const v = values[i] as number;
    if (Number.isFinite(v) && v > 0) finite.push(v);
  }
  if (finite.length === 0) return DEFAULT_SCALE_RANGE;
  finite.sort((a, b) => a - b);
  const at = (q: number) => finite[Math.min(finite.length - 1, Math.max(0, Math.floor(q * (finite.length - 1))))] as number;
  const min = at(0.001) / marginRatio;
  const max = at(0.999) * marginRatio;
  // レンジが潰れると log 空間で 0 除算になる
  return max / min < 1.01 ? { min: min / 1.05, max: max * 1.05 } : { min, max };
}

export function encodeLogScale(value: number, bits: number, range: LogRange = DEFAULT_SCALE_RANGE): number {
  const lo = Math.log(range.min);
  const hi = Math.log(range.max);
  const t = (Math.log(clamp(value, range.min, range.max)) - lo) / (hi - lo);
  const levels = (1 << bits) - 1;
  return Math.round(clamp(t, 0, 1) * levels);
}

export function decodeLogScale(code: number, bits: number, range: LogRange = DEFAULT_SCALE_RANGE): number {
  const lo = Math.log(range.min);
  const hi = Math.log(range.max);
  const levels = (1 << bits) - 1;
  return Math.exp(lo + (code / levels) * (hi - lo));
}

// --- 不透明度: logit 空間の一様量子化 --------------------------------------
//
// 効くのは α そのものの精度ではなく、透過率 (1−α) の相対精度である。
// α ブレンドでは多数のスプラットの (1−α) が掛け合わされるので、
// 透過率の相対誤差がそのまま累積する。
//
//   α       透過率の相対誤差（8bit）
//           logit      線形
//   0.50     1.18 %     0.39 %
//   0.95     0.31 %     1.96 %
//   0.99     0.68 %    17.65 %
//   0.997    0.56 %    30.72 %
//
// 線形量子化は α→1 で透過率が壊滅する（1/255 刻みが 1−α に対して巨大になる）。
// logit はそこを一定に保つ。代償として α=0.5 付近の絶対誤差は 0.0059 と
// 線形の 1/255=0.0039 より少し大きいが、そこは透過率が十分残る領域なので影響しない。

const LOGIT_LIMIT = 6; // sigmoid(±6) ≈ 0.0025 / 0.9975。これ以上は視覚差が無い

export function encodeOpacity(alpha: number, bits = 8): number {
  const a = clamp(alpha, 1e-4, 1 - 1e-4);
  const logit = Math.log(a / (1 - a));
  const t = (clamp(logit, -LOGIT_LIMIT, LOGIT_LIMIT) + LOGIT_LIMIT) / (2 * LOGIT_LIMIT);
  return Math.round(t * ((1 << bits) - 1));
}

export function decodeOpacity(code: number, bits = 8): number {
  const t = code / ((1 << bits) - 1);
  const logit = t * 2 * LOGIT_LIMIT - LOGIT_LIMIT;
  return 1 / (1 + Math.exp(-logit));
}

// --- 色: YCoCg 非対称割当 ---------------------------------------------------
// 人間の視覚は輝度より色差の感度が低い。YCoCg は RGB からの変換が加減算と
// シフトのみで、GPU 上で無損失かつ高速。Y=8bit, Co=6bit, Cg=6bit。

export interface YCoCg {
  y: number;
  co: number;
  cg: number;
}

/** sRGB 0..255 → YCoCg。co/cg は 0..255 にオフセットして返す。 */
export function rgbToYCoCg(r: number, g: number, b: number): YCoCg {
  const co = r - b;
  const tmp = b + (co >> 1);
  const cg = g - tmp;
  const y = tmp + (cg >> 1);
  return { y, co: co + 128, cg: cg + 128 };
}

export function yCoCgToRgb(y: number, coOff: number, cgOff: number): [number, number, number] {
  const co = coOff - 128;
  const cg = cgOff - 128;
  const tmp = y - (cg >> 1);
  const g = cg + tmp;
  const b = tmp - (co >> 1);
  const r = b + co;
  return [clamp(r, 0, 255) | 0, clamp(g, 0, 255) | 0, clamp(b, 0, 255) | 0];
}

// --- 半精度浮動小数 ---------------------------------------------------------
// WGSL の unpack2x16float と対になる。スケールを 24 バイト表現に詰めるのに使う。

const f32buf = new Float32Array(1);
const i32buf = new Int32Array(f32buf.buffer);

/** f32 を f16 のビット表現（16bit）にする。 */
export function toHalf(value: number): number {
  f32buf[0] = value;
  const x = i32buf[0] as number;
  const sign = (x >>> 16) & 0x8000;
  const exp = ((x >>> 23) & 0xff) - 127 + 15;
  const mant = x & 0x7fffff;
  if (exp <= 0) {
    // 非正規化数は 0 に落とす。スケールの下限は 1e-4 なので実害がない。
    return sign;
  }
  if (exp >= 31) return sign | 0x7c00;
  // 最近接偶数丸め
  let m = mant >>> 13;
  if ((mant & 0x1000) !== 0 && ((mant & 0x0fff) !== 0 || (m & 1) !== 0)) m += 1;
  if (m > 0x3ff) return sign | ((exp + 1) << 10) | (m & 0x3ff);
  return sign | (exp << 10) | m;
}

export function fromHalf(bits: number): number {
  const sign = (bits & 0x8000) !== 0 ? -1 : 1;
  const exp = (bits >>> 10) & 0x1f;
  const mant = bits & 0x3ff;
  if (exp === 0) return sign * mant * 2 ** -24;
  if (exp === 31) return mant === 0 ? sign * Infinity : NaN;
  return sign * (1 + mant / 1024) * 2 ** (exp - 15);
}

/** 2つの f32 を f16×2 として u32 に詰める。WGSL の unpack2x16float と対。 */
export function packHalf2(a: number, b: number): number {
  return ((toHalf(b) << 16) | toHalf(a)) >>> 0;
}

export function unpackHalf2(packed: number): [number, number] {
  return [fromHalf(packed & 0xffff), fromHalf(packed >>> 16)];
}

/** RGBA 8bit を u32 に詰める。WGSL の unpack4x8unorm と対（リトルエンディアン）。 */
export function packRgba8(r: number, g: number, b: number, a: number): number {
  return (
    ((clamp(a, 0, 255) << 24) | (clamp(b, 0, 255) << 16) | (clamp(g, 0, 255) << 8) | clamp(r, 0, 255)) >>> 0
  );
}

export function unpackRgba8(packed: number): [number, number, number, number] {
  return [packed & 0xff, (packed >>> 8) & 0xff, (packed >>> 16) & 0xff, (packed >>> 24) & 0xff];
}

function clamp(v: number, lo: number, hi: number): number {
  return v < lo ? lo : v > hi ? hi : v;
}
