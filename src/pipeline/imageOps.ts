/**
 * 画像とテンソルの変換。推論モデルの前後で要る細々した処理をここに集める。
 *
 * OffscreenCanvas に描いて縮小する手もあるが、モデル入力は数値の正規化まで
 * 面倒を見る必要があるし、出力（深度・マット）は画像ではないので canvas に
 * 戻す意味がない。どちらも素の配列で扱う。
 */

import { SUBJECT_ALPHA } from './1-matte';

/** ImageNet の平均と標準偏差。深度モデルはこれで正規化して学習されている。 */
export const IMAGENET_MEAN = [0.485, 0.456, 0.406] as const;
export const IMAGENET_STD = [0.229, 0.224, 0.225] as const;

/**
 * RGBA8 を双線形で拡縮する。
 *
 * 最近傍だと、縮小したマットの境界が階段になり、そのまま輪郭の階段として
 * 残る。拡大時も同じで、深度を最近傍で戻すと面がブロック状になる。
 */
export function resizeRgba(
  src: ArrayLike<number>,
  sw: number,
  sh: number,
  dw: number,
  dh: number,
): Uint8ClampedArray {
  const out = new Uint8ClampedArray(dw * dh * 4);
  const sx = sw / dw;
  const sy = sh / dh;
  for (let y = 0; y < dh; y++) {
    const fy = Math.min(sh - 1, Math.max(0, (y + 0.5) * sy - 0.5));
    const y0 = Math.floor(fy);
    const y1 = Math.min(sh - 1, y0 + 1);
    const wy = fy - y0;
    for (let x = 0; x < dw; x++) {
      const fx = Math.min(sw - 1, Math.max(0, (x + 0.5) * sx - 0.5));
      const x0 = Math.floor(fx);
      const x1 = Math.min(sw - 1, x0 + 1);
      const wx = fx - x0;
      const o = (y * dw + x) * 4;
      for (let c = 0; c < 4; c++) {
        const a = src[(y0 * sw + x0) * 4 + c] as number;
        const b = src[(y0 * sw + x1) * 4 + c] as number;
        const d = src[(y1 * sw + x0) * 4 + c] as number;
        const e = src[(y1 * sw + x1) * 4 + c] as number;
        out[o + c] = a * (1 - wx) * (1 - wy) + b * wx * (1 - wy) + d * (1 - wx) * wy + e * wx * wy;
      }
    }
  }
  return out;
}

/**
 * 色を鮮鋭化する（アンシャープマスク、docs/13 §13.2 T1）。
 *
 * **これは写真に無いものを足す処理である。** 既定では掛けない。
 *
 * 実測（描画 1 画素 = 写真 1 画素 にそろえて、胴の局所コントラストを測る）:
 *
 * | | 局所コントラスト | 元写真に対する比 |
 * |---|---|---|
 * | **元写真（情報の天井）** | **9.61** | **100%** |
 * | 私たち（強さ 0） | 9.79 | 102% |
 * | 私たち（**強さ 0.25**） | 12.54 | **131%** |
 * | 参照実装（SHARP） | 12.09 | 126% |
 *
 * 参照実装は回帰網が鮮鋭化していて、元写真より 26% コントラストが高い。
 * 強さ 0.25 でそこに並ぶ。ただし**スプラットの格子模様も一緒に持ち上がる**
 * ので、既定は素通しにしてある。
 *
 * **被写体の外は混ぜない。** レターボックスの余白や背景を低域に混ぜると、
 * シルエットの内側に縁取り（ハロ）が出る。α で重みをつけて避ける。
 *
 * @param alpha  被写体のマット。これが薄い画素は低域に混ぜない。
 * @param radius 低域を取る箱平均の半径（画素）。
 * @param amount 強さ。0 で何もしない。
 * @returns 新しい RGBA8。α は入力のまま。
 */
export function unsharpMaskRgba(
  rgba: ArrayLike<number>,
  alpha: ArrayLike<number>,
  width: number,
  height: number,
  radius: number,
  amount: number,
): Uint8ClampedArray {
  const out = Uint8ClampedArray.from(rgba as ArrayLike<number>);
  const r = Math.round(radius);
  if (!(amount > 0) || r < 1) return out;

  const n = width * height;
  const mask = new Uint8Array(n);
  for (let i = 0; i < n; i++) mask[i] = (alpha[i] as number) >= SUBJECT_ALPHA ? 1 : 0;

  // 低域。被写体の中だけを混ぜる分離可能な箱平均（窓を転がすので O(n)）。
  // 内側のループは関数を挟まずに書く。閉包にすると 4M 画素で 10 倍以上遅い。
  const rowSum = new Float32Array(n * 3);
  const rowCount = new Int32Array(n);
  for (let y = 0; y < height; y++) {
    const row = y * width;
    let c0 = 0;
    let c1 = 0;
    let c2 = 0;
    let cnt = 0;
    const first = Math.min(r, width - 1);
    for (let x = 0; x <= first; x++) {
      if (mask[row + x] === 0) continue;
      const o = (row + x) * 4;
      c0 += rgba[o] as number;
      c1 += rgba[o + 1] as number;
      c2 += rgba[o + 2] as number;
      cnt++;
    }
    for (let x = 0; x < width; x++) {
      const i = row + x;
      rowSum[i * 3] = c0;
      rowSum[i * 3 + 1] = c1;
      rowSum[i * 3 + 2] = c2;
      rowCount[i] = cnt;
      const add = x + r + 1;
      if (add < width && mask[row + add] === 1) {
        const o = (row + add) * 4;
        c0 += rgba[o] as number;
        c1 += rgba[o + 1] as number;
        c2 += rgba[o + 2] as number;
        cnt++;
      }
      const drop = x - r;
      if (drop >= 0 && mask[row + drop] === 1) {
        const o = (row + drop) * 4;
        c0 -= rgba[o] as number;
        c1 -= rgba[o + 1] as number;
        c2 -= rgba[o + 2] as number;
        cnt--;
      }
    }
  }

  for (let x = 0; x < width; x++) {
    let c0 = 0;
    let c1 = 0;
    let c2 = 0;
    let cnt = 0;
    const first = Math.min(r, height - 1);
    for (let y = 0; y <= first; y++) {
      const j = (y * width + x) * 3;
      c0 += rowSum[j] as number;
      c1 += rowSum[j + 1] as number;
      c2 += rowSum[j + 2] as number;
      cnt += rowCount[y * width + x] as number;
    }
    for (let y = 0; y < height; y++) {
      const i = y * width + x;
      if (mask[i] === 1 && cnt > 0) {
        const o = i * 4;
        const inv = 1 / cnt;
        out[o] = (rgba[o] as number) + amount * ((rgba[o] as number) - c0 * inv);
        out[o + 1] = (rgba[o + 1] as number) + amount * ((rgba[o + 1] as number) - c1 * inv);
        out[o + 2] = (rgba[o + 2] as number) + amount * ((rgba[o + 2] as number) - c2 * inv);
      }
      const add = y + r + 1;
      if (add < height) {
        const j = (add * width + x) * 3;
        c0 += rowSum[j] as number;
        c1 += rowSum[j + 1] as number;
        c2 += rowSum[j + 2] as number;
        cnt += rowCount[add * width + x] as number;
      }
      const drop = y - r;
      if (drop >= 0) {
        const j = (drop * width + x) * 3;
        c0 -= rowSum[j] as number;
        c1 -= rowSum[j + 1] as number;
        c2 -= rowSum[j + 2] as number;
        cnt -= rowCount[drop * width + x] as number;
      }
    }
  }
  return out;
}

/** 1チャンネルの実数配列を双線形で拡縮する。深度やマットを元の大きさに戻すのに使う。 */
export function resizePlane(
  src: ArrayLike<number>,
  sw: number,
  sh: number,
  dw: number,
  dh: number,
): Float32Array {
  const out = new Float32Array(dw * dh);
  const sx = sw / dw;
  const sy = sh / dh;
  for (let y = 0; y < dh; y++) {
    const fy = Math.min(sh - 1, Math.max(0, (y + 0.5) * sy - 0.5));
    const y0 = Math.floor(fy);
    const y1 = Math.min(sh - 1, y0 + 1);
    const wy = fy - y0;
    for (let x = 0; x < dw; x++) {
      const fx = Math.min(sw - 1, Math.max(0, (x + 0.5) * sx - 0.5));
      const x0 = Math.floor(fx);
      const x1 = Math.min(sw - 1, x0 + 1);
      const wx = fx - x0;
      const a = src[y0 * sw + x0] as number;
      const b = src[y0 * sw + x1] as number;
      const c = src[y1 * sw + x0] as number;
      const d = src[y1 * sw + x1] as number;
      out[y * dw + x] = a * (1 - wx) * (1 - wy) + b * wx * (1 - wy) + c * (1 - wx) * wy + d * wx * wy;
    }
  }
  return out;
}

/** 矩形を切り出す。深度のタイルパスで使う。 */
export function cropRgba(
  src: ArrayLike<number>,
  sw: number,
  x0: number,
  y0: number,
  w: number,
  h: number,
): Uint8ClampedArray {
  const out = new Uint8ClampedArray(w * h * 4);
  for (let y = 0; y < h; y++) {
    const so = ((y0 + y) * sw + x0) * 4;
    for (let i = 0; i < w * 4; i++) out[y * w * 4 + i] = src[so + i] as number;
  }
  return out;
}

export interface TensorOptions {
  /** チャンネルごとの平均。省略すると 0.5（＝[-1,1] へ写す用途）。 */
  readonly mean?: readonly number[];
  readonly std?: readonly number[];
}

/**
 * RGBA8 を NCHW の float32 に直す。
 *
 * α は捨てる。マット推定は α を知らない前提のモデルだし、深度モデルも同じ。
 * レターボックスの余白（α=0）は黒として渡ることになるが、そこは後段で
 * マットが 0 になるので問題にならない。
 */
export function toTensorNCHW(
  rgba: ArrayLike<number>,
  width: number,
  height: number,
  opts: TensorOptions = {},
): Float32Array {
  const mean = opts.mean ?? [0.5, 0.5, 0.5];
  const std = opts.std ?? [0.5, 0.5, 0.5];
  const n = width * height;
  const out = new Float32Array(3 * n);
  for (let i = 0; i < n; i++) {
    for (let c = 0; c < 3; c++) {
      const v = (rgba[i * 4 + c] as number) / 255;
      out[c * n + i] = (v - (mean[c] as number)) / (std[c] as number);
    }
  }
  return out;
}

/** 配列の最小・最大。0..1 への正規化に使う。 */
export function minMax(values: ArrayLike<number>): [number, number] {
  let lo = Infinity;
  let hi = -Infinity;
  for (let i = 0; i < values.length; i++) {
    const v = values[i] as number;
    if (!Number.isFinite(v)) continue;
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  return [Number.isFinite(lo) ? lo : 0, Number.isFinite(hi) ? hi : 1];
}
