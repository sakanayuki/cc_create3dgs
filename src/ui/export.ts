/**
 * 書き出し（docs/05 §5.3）。
 *
 * レンダラが持っている 24 バイトのスプラット列を、外に持ち出せる形式に直す。
 * 既定は `.spz`（SuperSplat / PlayCanvas / Spark.js などが読める）。
 */
import { unpackRgba8 } from '../codec/pack';
import { encodePly, quatFromNormal, type PlyGaussianOut } from '../codec/ply';
import {
  encodeSpzRaw,
  linearToSpzColor,
  SPZ_FRACTIONAL_BITS,
  type SpzPointOut,
} from '../codec/spz';
import { SPLAT_BYTES } from '../render/SplatRenderer';

/** サーフェルは厚みを持たない。法線方向のスケールはこの対数値にする。 */
const FLAT_LOG_SCALE = -8;

/** SH 0次の基底関数の値。PLY の f_dc はこれで割った係数として入る。 */
const SH_C0 = 0.28209479177387814;

interface Unpacked {
  x: number;
  y: number;
  z: number;
  nx: number;
  ny: number;
  nz: number;
  sx: number;
  sy: number;
  r: number;
  g: number;
  b: number;
  a: number;
}

/** 24 バイトの1個を読み出す。WGSL の loadSplat と同じ並び。 */
function unpack(data: Uint8Array, i: number, out: Unpacked): void {
  const f = new Float32Array(data.buffer, data.byteOffset + i * SPLAT_BYTES, 3);
  const u = new Uint32Array(data.buffer, data.byteOffset + i * SPLAT_BYTES, 6);
  out.x = f[0] as number;
  out.y = f[1] as number;
  out.z = f[2] as number;

  const packed = u[3] as number;
  let ex = ((packed & 0xffff) / 65535) * 2 - 1;
  let ey = ((packed >>> 16) / 65535) * 2 - 1;
  const nz = 1 - Math.abs(ex) - Math.abs(ey);
  if (nz < 0) {
    const t = -nz;
    ex += ex >= 0 ? -t : t;
    ey += ey >= 0 ? -t : t;
  }
  const len = Math.hypot(ex, ey, nz) || 1;
  out.nx = ex / len;
  out.ny = ey / len;
  out.nz = nz / len;

  // f16 × 2
  const sPacked = u[4] as number;
  const half = new Uint16Array([sPacked & 0xffff, (sPacked >>> 16) & 0xffff]);
  const conv = new Float32Array(2);
  conv[0] = halfToFloat(half[0] as number);
  conv[1] = halfToFloat(half[1] as number);
  out.sx = conv[0] as number;
  out.sy = conv[1] as number;

  const [r, g, b, a] = unpackRgba8(u[5] as number);
  out.r = r;
  out.g = g;
  out.b = b;
  out.a = a;
}

/** IEEE 754 半精度 → 単精度。 */
function halfToFloat(h: number): number {
  const s = (h & 0x8000) >> 15;
  const e = (h & 0x7c00) >> 10;
  const f = h & 0x03ff;
  if (e === 0) return (s ? -1 : 1) * 2 ** -14 * (f / 1024);
  if (e === 0x1f) return f ? NaN : (s ? -1 : 1) * Infinity;
  return (s ? -1 : 1) * 2 ** (e - 15) * (1 + f / 1024);
}

/** 0..255 の sRGB をリニアに直す。SPZ の色は SH の DC 項なので、リニアで渡す。 */
function srgbToLinear(u8: number): number {
  const c = u8 / 255;
  return c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
}

/** ロジット。SPZ / PLY の不透明度はロジット空間で持つ（docs/04 §4.4）。 */
function logit(a: number): number {
  const clamped = Math.min(0.999, Math.max(0.001, a));
  return Math.log(clamped / (1 - clamped));
}

/** `.spz` のバイト列（gzip 前）。 */
export function splatsToSpzRaw(data: Uint8Array, count: number): Uint8Array {
  const u: Unpacked = {
    x: 0, y: 0, z: 0, nx: 0, ny: 0, nz: 1, sx: 0, sy: 0, r: 0, g: 0, b: 0, a: 0,
  };
  return encodeSpzRaw(
    { numPoints: count, shDegree: 0, fractionalBits: SPZ_FRACTIONAL_BITS, antialiased: true },
    (i: number, out: SpzPointOut) => {
      unpack(data, i, u);
      out.x = u.x;
      out.y = u.y;
      out.z = u.z;
      out.alpha = Math.round(u.a);
      out.r = linearToSpzColor(srgbToLinear(u.r));
      out.g = linearToSpzColor(srgbToLinear(u.g));
      out.b = linearToSpzColor(srgbToLinear(u.b));
      // サーフェルは平ら。法線方向だけ極小にする。
      out.logScale0 = Math.log(Math.max(u.sx, 1e-6));
      out.logScale1 = Math.log(Math.max(u.sy, 1e-6));
      out.logScale2 = FLAT_LOG_SCALE;
      const [qx, qy, qz, qw] = quatFromNormal(u.nx, u.ny, u.nz);
      out.qx = qx;
      out.qy = qy;
      out.qz = qz;
      out.qw = qw;
    },
  );
}

/** `.ply`（vanilla 3DGS 互換）のバイト列。 */
export function splatsToPly(data: Uint8Array, count: number): Uint8Array {
  const u: Unpacked = {
    x: 0, y: 0, z: 0, nx: 0, ny: 0, nz: 1, sx: 0, sy: 0, r: 0, g: 0, b: 0, a: 0,
  };
  return encodePly(count, (i: number, out: PlyGaussianOut) => {
    unpack(data, i, u);
    out.x = u.x;
    out.y = u.y;
    out.z = u.z;
    out.nx = u.nx;
    out.ny = u.ny;
    out.nz = u.nz;
    out.opacityLogit = logit(u.a / 255);
    out.logScale0 = Math.log(Math.max(u.sx, 1e-6));
    out.logScale1 = Math.log(Math.max(u.sy, 1e-6));
    out.logScale2 = FLAT_LOG_SCALE;
    const [qx, qy, qz, qw] = quatFromNormal(u.nx, u.ny, u.nz);
    out.rotW = qw;
    out.rotX = qx;
    out.rotY = qy;
    out.rotZ = qz;
    // PLY の f_dc は SH の DC 項。リニア色から逆算する。
    out.dc0 = (srgbToLinear(u.r) - 0.5) / SH_C0;
    out.dc1 = (srgbToLinear(u.g) - 0.5) / SH_C0;
    out.dc2 = (srgbToLinear(u.b) - 0.5) / SH_C0;
  });
}

/** gzip して `.spz` にする。CompressionStream が無い環境では非圧縮のまま返す。 */
export async function toSpzFile(data: Uint8Array, count: number): Promise<Blob> {
  const raw = splatsToSpzRaw(data, count);
  if (typeof CompressionStream === 'undefined') {
    return new Blob([raw as BlobPart], { type: 'application/octet-stream' });
  }
  const stream = new Blob([raw as BlobPart]).stream().pipeThrough(new CompressionStream('gzip'));
  return new Response(stream).blob();
}
