/**
 * 髪の深度を均す（docs/13 §13.5）。
 *
 * 単眼深度モデルは、髪の筋の 1 本 1 本に別々の深度を返す。正面からは写真の
 * 色が見えるだけなので気付かないが、**45° も回すと、髪の縁が段々の板のように
 * 波打ち、首の前を横切る筋が横へ引き伸ばされる**（実写 test26）。参照実装
 * （SHARP）の髪は毛束状にふわっとしていて、こうは割れない。
 *
 * 髪は「顔の外・頭の箱の中・顔より暗い」画素とする。顔は landmark 面が別に
 * 立てているので触らない。明るい髪（金髪など）は顔の明るさに近く、髪と
 * 判定されない。**その場合は素通し**で、悪くはならない。
 */

export interface HairSmoothParams {
  /** 均す半径（画素）。作業グリッドに対する割合で置く。 */
  readonly radiusRatio: number;
  /** 箱平均を何回重ねるか。重ねるほどガウス型に近づく。 */
  readonly passes: number;
  /** 髪とみなす輝度の上限。顔の輝度の中央値に対する比。 */
  readonly darkerThan: number;
}

export const DEFAULT_HAIR_SMOOTH: HairSmoothParams = {
  radiusRatio: 0.02,
  passes: 3,
  darkerThan: 0.75,
};

interface Rect {
  readonly x: number;
  readonly y: number;
  readonly width: number;
  readonly height: number;
}

function luminance(rgba: ArrayLike<number>, i: number): number {
  return (
    0.299 * (rgba[i * 4] as number) +
    0.587 * (rgba[i * 4 + 1] as number) +
    0.114 * (rgba[i * 4 + 2] as number)
  );
}

/** 髪の画素だけを数えて平均する箱平均。外を 0 として混ぜない。 */
function maskedBlur(
  src: Float32Array,
  mask: Uint8Array,
  width: number,
  height: number,
  radius: number,
): Float32Array {
  const n = width * height;
  const sum = new Float64Array(n);
  const cnt = new Float64Array(n);
  // 行方向
  for (let y = 0; y < height; y++) {
    const row = y * width;
    let s = 0;
    let c = 0;
    for (let x = 0; x <= radius && x < width; x++) {
      if (mask[row + x]) {
        s += src[row + x] as number;
        c++;
      }
    }
    for (let x = 0; x < width; x++) {
      sum[row + x] = s;
      cnt[row + x] = c;
      const add = x + radius + 1;
      if (add < width && mask[row + add]) {
        s += src[row + add] as number;
        c++;
      }
      const drop = x - radius;
      if (drop >= 0 && mask[row + drop]) {
        s -= src[row + drop] as number;
        c--;
      }
    }
  }
  // 列方向
  const out = new Float32Array(n);
  for (let x = 0; x < width; x++) {
    let s = 0;
    let c = 0;
    for (let y = 0; y <= radius && y < height; y++) {
      s += sum[y * width + x] as number;
      c += cnt[y * width + x] as number;
    }
    for (let y = 0; y < height; y++) {
      const i = y * width + x;
      out[i] = c > 0 ? s / c : (src[i] as number);
      const add = y + radius + 1;
      if (add < height) {
        s += sum[add * width + x] as number;
        c += cnt[add * width + x] as number;
      }
      const drop = y - radius;
      if (drop >= 0) {
        s -= sum[drop * width + x] as number;
        c -= cnt[drop * width + x] as number;
      }
    }
  }
  return out;
}

/**
 * @param depth    実寸の深度（大きいほど奥）。破壊しない。
 * @param faceMask 作業グリッドと同じ大きさの顔の重み（0..1）。顔の外は 0。
 * @param headBox  頭を含む四角。この外は触らない。
 */
export function smoothHairDepth(
  depth: ArrayLike<number>,
  rgba: ArrayLike<number>,
  alpha: ArrayLike<number>,
  faceMask: ArrayLike<number>,
  width: number,
  height: number,
  headBox: Rect,
  params: HairSmoothParams = DEFAULT_HAIR_SMOOTH,
): Float32Array {
  const out = Float32Array.from(depth);
  const n = width * height;

  // 顔の明るさ（中央値）。髪はそれより暗い。
  const faceLum: number[] = [];
  for (let i = 0; i < n; i++) {
    if ((faceMask[i] as number) >= 0.5 && (alpha[i] as number) >= 128) faceLum.push(luminance(rgba, i));
  }
  if (faceLum.length < 256) return out;
  faceLum.sort((a, b) => a - b);
  const limit = (faceLum[faceLum.length >> 1] as number) * params.darkerThan;

  const hair = new Uint8Array(n);
  let count = 0;
  const x0 = Math.max(0, headBox.x);
  const y0 = Math.max(0, headBox.y);
  const x1 = Math.min(width, headBox.x + headBox.width);
  const y1 = Math.min(height, headBox.y + headBox.height);
  for (let y = y0; y < y1; y++) {
    for (let x = x0; x < x1; x++) {
      const i = y * width + x;
      if ((alpha[i] as number) < 128) continue;
      if ((faceMask[i] as number) > 0.02) continue;
      if (luminance(rgba, i) >= limit) continue;
      if (!Number.isFinite(depth[i] as number)) continue;
      hair[i] = 1;
      count++;
    }
  }
  if (count < 256) return out;

  const radius = Math.max(1, Math.round(width * params.radiusRatio));
  let cur: Float32Array = Float32Array.from(depth);
  for (let p = 0; p < params.passes; p++) cur = maskedBlur(cur, hair, width, height, radius);
  for (let i = 0; i < n; i++) if (hair[i]) out[i] = cur[i] as number;
  return out;
}
