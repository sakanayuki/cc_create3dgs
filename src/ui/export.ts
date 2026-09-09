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
import { SPLAT_BYTES, SURFEL_FLATNESS } from '../render/SplatRenderer';

/**
 * 法線方向のスケール（厚み）。面内の半径に**比例**させる（v2.6.3）。
 *
 * v2.6.2 まではここを定数 `exp(-8) = 3.35e-4` にしていた。小さいスプラット
 * では比 0.34 と妥当だが、統合された大きいスプラット（半径 0.005）では
 * 0.067 と極端に平たくなり、外部ビューアで寝たときに線に潰れていた。
 * 参照実装（SHARP）の出力は大きさ帯によらず比 0.33〜0.45 で一定である。
 * 詳しくは `SURFEL_FLATNESS`（docs/06 §6.11）。
 */
function thicknessOf(sx: number, sy: number): number {
  return Math.max(SURFEL_FLATNESS * Math.max(sx, sy), 1e-6);
}

/**
 * 書き出す座標系（docs/05 §5.3.1、v2.6.9）。
 *
 * ⑥ は被写体を**一辺 1 の立方体**に正規化して出す。描画にはそれで足りるが、
 * **書き出したファイルまで正規化座標なのは誤り**だった。参照実装（SHARP）の
 * 出力を測ると、どれもカメラを原点にした**実寸（メートル）**で入っている:
 *
 * | | 外接（幅×高さ×奥行き） | z の範囲 |
 * |---|---|---|
 * | SHARP test29 | 0.663 × 1.665 × 0.295 | +1.005 〜 +1.300 |
 * | SHARP test26 | 0.765 × 1.366 × 0.899 | +0.793 〜 +1.692 |
 * | SHARP test31 | 0.810 × 2.035 × 0.635 | +1.545 〜 +2.180 |
 * | 私たち（正規化のまま） | 0.385 × **1.000** × 0.216 | −0.108 〜 +0.108 |
 *
 * 身長 1.67m の人が高さ 1.0 で出ていた。他の資産と並べる・寸法を測る・
 * 実寸で出力する、のどれもできない。`Normalization` が逆変換をそのまま
 * 持っているので、書き出しのときに戻す。
 *
 * `metricFix` は、そのうえで掛ける倍率である。深度モデルの絶対値が当てに
 * ならないぶんを、呼び出し側が推定して渡す（docs/09 §V25）。
 *
 * **半径も位置も同じ倍率で動くので、形も見た目も変わらない。** 変わるのは
 * 「1 単位が何メートルか」だけである。
 */
export interface ExportFrame {
  /** カメラ座標 = 正規化座標 ÷ scale + center。 */
  readonly center: readonly [number, number, number];
  /** 正規化の倍率（`SplatBuild.normalization.scale`）。 */
  readonly scale: number;
  /** 実寸へ直す追加の倍率。省略すると 1（深度モデルの値をそのまま使う）。 */
  readonly metricFix?: number;
}

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

/**
 * 24 バイトの1個を読み出す。WGSL の loadSplat と同じ並び。
 *
 * `frame` を渡すと、正規化を戻して実寸（カメラを原点にしたメートル）で返す。
 */
function unpack(data: Uint8Array, i: number, out: Unpacked, frame?: ExportFrame | null): void {
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

  // 正規化を戻して実寸にする。位置も半径も同じ倍率なので、形は変わらない。
  if (frame && frame.scale > 0) {
    const k = (frame.metricFix ?? 1) / frame.scale;
    out.x = (out.x / frame.scale + (frame.center[0] as number)) * (frame.metricFix ?? 1);
    out.y = (out.y / frame.scale + (frame.center[1] as number)) * (frame.metricFix ?? 1);
    out.z = (out.z / frame.scale + (frame.center[2] as number)) * (frame.metricFix ?? 1);
    out.sx *= k;
    out.sy *= k;
  }
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

/**
 * 0..255 の色を、3DGS の色空間の 0..1 に直す。
 *
 * **ガンマ変換をしてはいけない。** 3DGS の SH DC 項は「学習に使った画像の
 * 画素値」をそのまま表す量である。vanilla 3DGS はラスタライザの出力
 * `SH_C0 · dc + 0.5` を、読み込んだ PNG/JPEG の値（sRGB を 255 で割っただけ）と
 * 直接比べて学習する。つまり dc が符号化しているのは **sRGB/255** であって
 * リニア輝度ではない。
 *
 * 最初ここで sRGB→リニア変換をしていた。両端（0 と 255）は一致するので
 * 単体テストの飽和検査は通ってしまうが、中間調が大きくずれる。
 * 中間の灰色 128 は、他のビューアで 55 として表示されていた。
 */
function toSceneColor(u8: number): number {
  return u8 / 255;
}

/** ロジット。SPZ / PLY の不透明度はロジット空間で持つ（docs/04 §4.4）。 */
function logit(a: number): number {
  const clamped = Math.min(0.999, Math.max(0.001, a));
  return Math.log(clamped / (1 - clamped));
}

/** `.spz` のバイト列（gzip 前）。 */
export function splatsToSpzRaw(
  data: Uint8Array,
  count: number,
  frame: ExportFrame | null = null,
): Uint8Array {
  const u: Unpacked = {
    x: 0, y: 0, z: 0, nx: 0, ny: 0, nz: 1, sx: 0, sy: 0, r: 0, g: 0, b: 0, a: 0,
  };
  return encodeSpzRaw(
    { numPoints: count, shDegree: 0, fractionalBits: SPZ_FRACTIONAL_BITS, antialiased: true },
    (i: number, out: SpzPointOut) => {
      unpack(data, i, u, frame);
      out.x = u.x;
      out.y = u.y;
      out.z = u.z;
      out.alpha = Math.round(u.a);
      out.r = linearToSpzColor(toSceneColor(u.r));
      out.g = linearToSpzColor(toSceneColor(u.g));
      out.b = linearToSpzColor(toSceneColor(u.b));
      // サーフェルは平ら。法線方向だけ極小にする。
      out.logScale0 = Math.log(Math.max(u.sx, 1e-6));
      out.logScale1 = Math.log(Math.max(u.sy, 1e-6));
      out.logScale2 = Math.log(thicknessOf(u.sx, u.sy));
      const [qx, qy, qz, qw] = quatFromNormal(u.nx, u.ny, u.nz);
      out.qx = qx;
      out.qy = qy;
      out.qz = qz;
      out.qw = qw;
    },
  );
}

/** `.ply`（vanilla 3DGS 互換）のバイト列。 */
export function splatsToPly(
  data: Uint8Array,
  count: number,
  frame: ExportFrame | null = null,
): Uint8Array {
  const u: Unpacked = {
    x: 0, y: 0, z: 0, nx: 0, ny: 0, nz: 1, sx: 0, sy: 0, r: 0, g: 0, b: 0, a: 0,
  };
  return encodePly(count, (i: number, out: PlyGaussianOut) => {
    unpack(data, i, u, frame);
    out.x = u.x;
    out.y = u.y;
    out.z = u.z;
    out.nx = u.nx;
    out.ny = u.ny;
    out.nz = u.nz;
    out.opacityLogit = logit(u.a / 255);
    out.logScale0 = Math.log(Math.max(u.sx, 1e-6));
    out.logScale1 = Math.log(Math.max(u.sy, 1e-6));
    out.logScale2 = Math.log(thicknessOf(u.sx, u.sy));
    const [qx, qy, qz, qw] = quatFromNormal(u.nx, u.ny, u.nz);
    out.rotW = qw;
    out.rotX = qx;
    out.rotY = qy;
    out.rotZ = qz;
    // PLY の f_dc は SH の DC 項。
    out.dc0 = (toSceneColor(u.r) - 0.5) / SH_C0;
    out.dc1 = (toSceneColor(u.g) - 0.5) / SH_C0;
    out.dc2 = (toSceneColor(u.b) - 0.5) / SH_C0;
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

// --- .splat（antimatter15/splat 形式） -------------------------------------

/** 1個あたりのバイト数。位置12 + スケール12 + 色4 + 回転4。 */
export const SPLAT_STRIDE = 32;

/**
 * `.splat` を書き出す。
 *
 * antimatter15 の viewer が読む形式で、PlayCanvas / SuperSplat なども対応する。
 * 仕様は同リポジトリの `convert.py` に合わせた（実物を取得して確認済み）。
 *
 *   position  float32 × 3   そのまま
 *   scale     float32 × 3   **線形**（PLY の対数スケールを exp したもの）
 *   color     uint8   × 4   `(0.5 + SH_C0·dc) · 255` = 画素値そのもの。α は 0..255
 *   rotation  uint8   × 4   `(q/|q|) · 128 + 128`、順序は PLY と同じ (w, x, y, z)
 *
 * 参照実装は「大きくて不透明なものから先」に並べ替えてから書く。ビューアが
 * 先頭から順に読み込んで表示するので、その順だと形が早く見えてくる。
 * 同じ理由でこちらも並べ替える。描画結果は順序に依らない。
 */
export function splatsToSplat(
  data: Uint8Array,
  count: number,
  frame: ExportFrame | null = null,
): Uint8Array {
  const u: Unpacked = {
    x: 0, y: 0, z: 0, nx: 0, ny: 0, nz: 1, sx: 0, sy: 0, r: 0, g: 0, b: 0, a: 0,
  };

  // 並べ替えの鍵: 体積 × 不透明度。参照実装と同じ量を使う。
  const order = new Uint32Array(count);
  const key = new Float32Array(count);
  for (let i = 0; i < count; i++) {
    unpack(data, i, u, frame);
    order[i] = i;
    key[i] = u.sx * u.sy * thicknessOf(u.sx, u.sy) * (u.a / 255);
  }
  const sorted = Array.from(order).sort(
    (a, b) => (key[b] as number) - (key[a] as number),
  );

  const out = new Uint8Array(count * SPLAT_STRIDE);
  const f32 = new Float32Array(out.buffer);

  for (let n = 0; n < count; n++) {
    unpack(data, sorted[n] as number, u, frame);
    const o = n * SPLAT_STRIDE;
    const of = o / 4;
    f32[of] = u.x;
    f32[of + 1] = u.y;
    f32[of + 2] = u.z;
    // .splat のスケールは線形。厚みは面内の半径に比例させる。
    f32[of + 3] = u.sx;
    f32[of + 4] = u.sy;
    f32[of + 5] = thicknessOf(u.sx, u.sy);
    out[o + 24] = u.r;
    out[o + 25] = u.g;
    out[o + 26] = u.b;
    out[o + 27] = u.a;

    const [qx, qy, qz, qw] = quatFromNormal(u.nx, u.ny, u.nz);
    const len = Math.hypot(qw, qx, qy, qz) || 1;
    const q = [qw / len, qx / len, qy / len, qz / len];
    for (let k = 0; k < 4; k++) {
      out[o + 28 + k] = Math.max(0, Math.min(255, Math.round((q[k] as number) * 128 + 128)));
    }
  }
  return out;
}

/** 選べる書き出し形式。 */
export type ExportFormat = 'spz' | 'ply' | 'splat';

export interface ExportInfo {
  readonly extension: string;
  readonly label: string;
  readonly mime: string;
  /** 圧縮するか。.spz だけ gzip。 */
  readonly gzip: boolean;
  readonly note: string;
}

export const EXPORT_FORMATS: Record<ExportFormat, ExportInfo> = {
  spz: {
    extension: 'spz',
    label: '.spz',
    mime: 'application/octet-stream',
    gzip: true,
    note: '最小。SuperSplat / PlayCanvas / Spark.js が読む',
  },
  ply: {
    extension: 'ply',
    label: '.ply',
    mime: 'application/octet-stream',
    gzip: false,
    note: '3DGS の標準。ほぼどの実装でも読めるが大きい',
  },
  splat: {
    extension: 'splat',
    label: '.splat',
    mime: 'application/octet-stream',
    gzip: false,
    note: 'antimatter15 系のビューア向け。1個 32 バイト',
  },
};

/** 指定の形式で書き出す。 */
export async function toSplatFile(
  data: Uint8Array,
  count: number,
  format: ExportFormat,
  /** 実寸へ戻す変換。省略すると正規化座標のまま出す。 */
  frame: ExportFrame | null = null,
): Promise<Blob> {
  const info = EXPORT_FORMATS[format];
  const bytes =
    format === 'spz'
      ? splatsToSpzRaw(data, count, frame)
      : format === 'ply'
        ? splatsToPly(data, count, frame)
        : splatsToSplat(data, count, frame);

  if (!info.gzip) return new Blob([bytes as BlobPart], { type: info.mime });
  if (typeof CompressionStream === 'undefined') {
    // 圧縮できない環境でも書き出せるようにする。読み手は gzip を期待するので、
    // その場合だけ拡張子を変えたいところだが、呼び出し側で判断する。
    return new Blob([bytes as BlobPart], { type: info.mime });
  }
  const stream = new Blob([bytes as BlobPart]).stream().pipeThrough(new CompressionStream('gzip'));
  return new Response(stream).blob();
}
