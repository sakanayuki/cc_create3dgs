/**
 * 画像とテンソルの変換。推論モデルの前後で要る細々した処理をここに集める。
 *
 * OffscreenCanvas に描いて縮小する手もあるが、モデル入力は数値の正規化まで
 * 面倒を見る必要があるし、出力（深度・マット）は画像ではないので canvas に
 * 戻す意味がない。どちらも素の配列で扱う。
 */

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
