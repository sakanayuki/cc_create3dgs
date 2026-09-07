/**
 * `.spz` 書き出し（docs/05 §5.3）— 既定の書き出し形式。
 *
 * Niantic が策定した 3DGS の圧縮形式。SuperSplat / PlayCanvas / Spark.js /
 * Three.js が対応しており、実質的な標準になりつつある。
 * 「立体を持ち帰る」という用途では、他のビューアで開けることが最も価値が高いので
 * UI の既定にする。
 *
 * 形式（v2, gzip 圧縮された本体）:
 *   ヘッダ 16 バイト
 *     0  u32 magic  = 0x5053474e ("NGSP")
 *     4  u32 version = 2
 *     8  u32 numPoints
 *     12 u8  shDegree
 *     13 u8  fractionalBits
 *     14 u8  flags        (bit0: antialiased)
 *     15 u8  reserved
 *   本体（属性ごとに連続、Structure-of-Arrays）
 *     positions : numPoints × 3 × 3 バイト（24bit 固定小数点）
 *     alphas    : numPoints × 1
 *     colors    : numPoints × 3
 *     scales    : numPoints × 3
 *     rotations : numPoints × 3      （クォータニオンの xyz。w は符号から復元）
 *
 * 背面シェルとスカートは展開して含める。受け取り側は本アプリの導出規則を
 * 知らないため。これが .pgs（約 750 KB）に対して .spz が約 5.9 MB になる主因。
 */

import { dcToLinear, linearToDc } from './ply';

export const SPZ_MAGIC = 0x5053474e;
export const SPZ_VERSION = 2;
/** 位置の小数部ビット数。Niantic の既定値。 */
export const SPZ_FRACTIONAL_BITS = 12;

export interface SpzHeader {
  readonly numPoints: number;
  readonly shDegree: number;
  readonly fractionalBits: number;
  readonly antialiased: boolean;
}

/** 1個ぶんの書き込み先。使い回して割り当てを避ける。 */
export class SpzPointOut {
  x = 0; y = 0; z = 0;
  /** 0..255。sigmoid 済みの α をそのまま入れる。 */
  alpha = 0;
  /** 0..255。SH DC ではなく sRGB 相当を SPZ の色エンコードに合わせて入れる。 */
  r = 0; g = 0; b = 0;
  /** exp 前の対数スケール。SPZ は (logScale + 10) * 16 を u8 にする。 */
  logScale0 = 0; logScale1 = 0; logScale2 = 0;
  /** 正規化クォータニオン (w, x, y, z)。w ≥ 0 に正規化して xyz だけ書く。 */
  qw = 1; qx = 0; qy = 0; qz = 0;
}

/**
 * SPZ の色エンコード。
 *
 * SPZ が u8 に詰めているのは sRGB ではなく **SH の DC 係数**である。
 *   u8 = dc · colorScale · 255 + 127.5      （colorScale = 0.15）
 * したがって線形色からは SH DC を経由する必要がある（linearToDc は ply.ts と共有）。
 *
 * 直接 (c − 0.5)/colorScale を u8 に写すと c=0 や c=1 で飽和して
 * 黒と白が潰れる。単体テストで往復を固定してある。
 */
export const SPZ_COLOR_SCALE = 0.15;
const DC_TO_U8 = SPZ_COLOR_SCALE * 255;

export const linearToSpzColor = (c: number): number =>
  clampU8(Math.round(linearToDc(c) * DC_TO_U8 + 127.5));
export const spzColorToLinear = (u: number): number => dcToLinear((u - 127.5) / DC_TO_U8);

function clampU8(v: number): number {
  return v < 0 ? 0 : v > 255 ? 255 : v | 0;
}

/**
 * 非圧縮の SPZ バイト列を作る。呼び出し側で gzip して `.spz` にする。
 *
 * @param write 1個ずつ埋めるコールバック。巨大な中間配列を作らずに済む。
 */
export function encodeSpzRaw(
  header: SpzHeader,
  write: (i: number, out: SpzPointOut) => void,
): Uint8Array {
  const n = header.numPoints;
  const shCoeffs = shCoeffCount(header.shDegree);
  const bytes = 16 + n * (9 + 1 + 3 + 3 + 3 + shCoeffs * 3);
  const out = new Uint8Array(bytes);
  const dv = new DataView(out.buffer);

  dv.setUint32(0, SPZ_MAGIC, true);
  dv.setUint32(4, SPZ_VERSION, true);
  dv.setUint32(8, n, true);
  out[12] = header.shDegree;
  out[13] = header.fractionalBits;
  out[14] = header.antialiased ? 1 : 0;
  out[15] = 0;

  // Structure-of-Arrays。属性ごとにまとめると gzip がよく効く。
  let pPos = 16;
  let pAlpha = pPos + n * 9;
  let pColor = pAlpha + n;
  let pScale = pColor + n * 3;
  let pRot = pScale + n * 3;

  const scale = 1 << header.fractionalBits;
  const slot = new SpzPointOut();

  for (let i = 0; i < n; i++) {
    write(i, slot);

    for (const v of [slot.x, slot.y, slot.z]) {
      const fixed = Math.round(v * scale);
      out[pPos++] = fixed & 0xff;
      out[pPos++] = (fixed >> 8) & 0xff;
      out[pPos++] = (fixed >> 16) & 0xff;
    }

    out[pAlpha++] = clampU8(slot.alpha);

    out[pColor++] = clampU8(slot.r);
    out[pColor++] = clampU8(slot.g);
    out[pColor++] = clampU8(slot.b);

    for (const s of [slot.logScale0, slot.logScale1, slot.logScale2]) {
      out[pScale++] = clampU8(Math.round((s + 10) * 16));
    }

    // w ≥ 0 に揃えて xyz だけ書く。復元側は w = sqrt(1 − x²−y²−z²)。
    const sign = slot.qw < 0 ? -1 : 1;
    for (const q of [slot.qx, slot.qy, slot.qz]) {
      out[pRot++] = clampU8(Math.round(q * sign * 127.5 + 127.5));
    }
  }
  return out;
}

export function shCoeffCount(degree: number): number {
  return degree === 0 ? 0 : degree === 1 ? 3 : degree === 2 ? 8 : 15;
}

/** ヘッダだけ読む。読み込み対応と、書き出したものの検証に使う。 */
export function decodeSpzHeader(bytes: Uint8Array): SpzHeader {
  if (bytes.length < 16) throw new Error('SPZ ファイルが短すぎます');
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const magic = dv.getUint32(0, true);
  if (magic !== SPZ_MAGIC) throw new Error('SPZ ファイルではありません（マジックが不一致）');
  const version = dv.getUint32(4, true);
  if (version !== SPZ_VERSION) throw new Error(`未対応の SPZ バージョンです: ${version}`);
  return {
    numPoints: dv.getUint32(8, true),
    shDegree: bytes[12] ?? 0,
    fractionalBits: bytes[13] ?? SPZ_FRACTIONAL_BITS,
    antialiased: (bytes[14] ?? 0) === 1,
  };
}

/** 非圧縮 SPZ の期待バイト数。 */
export function spzRawByteLength(numPoints: number, shDegree = 0): number {
  return 16 + numPoints * (9 + 1 + 3 + 3 + 3 + shCoeffCount(shDegree) * 3);
}

/**
 * gzip して `.spz` にする。ブラウザの CompressionStream を使う。
 */
export async function gzipBytes(raw: Uint8Array): Promise<Uint8Array> {
  const cs = new CompressionStream('gzip');
  const stream = new Blob([raw.slice()]).stream().pipeThrough(cs);
  return new Uint8Array(await new Response(stream).arrayBuffer());
}
