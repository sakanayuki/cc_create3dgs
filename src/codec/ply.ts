/**
 * `.ply` 書き出し（docs/05 §5.4）— 最大互換の出力。
 *
 * INRIA の 3DGS 標準形式（binary_little_endian）に合わせるが、2点だけ違う。
 *
 *   1. `f_rest_*`（SH 1〜3次）を出力しない。単一画像に視点依存の情報は無く、
 *      持たせても無根拠な外挿になるため（docs/04 §4.2）。ほとんどのビューアは
 *      欠如を許容し、次数0として扱う。
 *   2. `nx, ny, nz` に**実際の法線を書き込む**。標準の 3DGS PLY ではここは常に 0 で
 *      情報を持たない無駄なフィールドだが、本設計はサーフェル表現なので法線を持つ。
 *      書いておけば Blender へのインポートやメッシュ化で役に立つ。無視するビューアには影響しない。
 */

/** PLY に書き出す1個ぶんの属性。 */
export interface PlyGaussian {
  readonly x: number;
  readonly y: number;
  readonly z: number;
  readonly nx: number;
  readonly ny: number;
  readonly nz: number;
  /** SH DC 成分（f_dc_0..2）。sRGB 0..1 ではなく SH の係数であることに注意。 */
  readonly dc: readonly [number, number, number];
  /** sigmoid 前の値（3DGS の慣習）。 */
  readonly opacityLogit: number;
  /** exp 前の値（3DGS の慣習）。 */
  readonly logScale: readonly [number, number, number];
  /** クォータニオン (w, x, y, z)。 */
  readonly rot: readonly [number, number, number, number];
}

const PROPS = [
  'x', 'y', 'z',
  'nx', 'ny', 'nz',
  'f_dc_0', 'f_dc_1', 'f_dc_2',
  'opacity',
  'scale_0', 'scale_1', 'scale_2',
  'rot_0', 'rot_1', 'rot_2', 'rot_3',
] as const;

const FLOATS_PER_VERTEX = PROPS.length; // 17
const BYTES_PER_VERTEX = FLOATS_PER_VERTEX * 4;

/** SH 次数0 の DC 係数と sRGB(0..1) の変換定数。3DGS の慣習に合わせる。 */
export const SH_C0 = 0.28209479177387814;

export const linearToDc = (c: number): number => (c - 0.5) / SH_C0;
export const dcToLinear = (dc: number): number => dc * SH_C0 + 0.5;

export function plyHeader(count: number): string {
  return [
    'ply',
    'format binary_little_endian 1.0',
    `element vertex ${count}`,
    ...PROPS.map((p) => `property float ${p}`),
    'end_header',
    '',
  ].join('\n');
}

/**
 * ガウシアン列を PLY のバイト列にする。
 * @param write 1個ずつ書き込むコールバック。巨大配列を一度に作らずに済む。
 */
export function encodePly(count: number, write: (i: number, out: PlyGaussianOut) => void): Uint8Array {
  const header = new TextEncoder().encode(plyHeader(count));
  const out = new Uint8Array(header.length + count * BYTES_PER_VERTEX);
  out.set(header, 0);

  const body = new DataView(out.buffer, header.length);
  const slot = new PlyGaussianOut();
  for (let i = 0; i < count; i++) {
    write(i, slot);
    slot.writeTo(body, i * BYTES_PER_VERTEX);
  }
  return out;
}

/** 書き込み先の使い回し用オブジェクト。1個ごとに new しないための工夫。 */
export class PlyGaussianOut {
  x = 0; y = 0; z = 0;
  nx = 0; ny = 0; nz = 0;
  dc0 = 0; dc1 = 0; dc2 = 0;
  opacityLogit = 0;
  logScale0 = 0; logScale1 = 0; logScale2 = 0;
  rotW = 1; rotX = 0; rotY = 0; rotZ = 0;

  set(g: PlyGaussian): void {
    this.x = g.x; this.y = g.y; this.z = g.z;
    this.nx = g.nx; this.ny = g.ny; this.nz = g.nz;
    this.dc0 = g.dc[0]; this.dc1 = g.dc[1]; this.dc2 = g.dc[2];
    this.opacityLogit = g.opacityLogit;
    this.logScale0 = g.logScale[0]; this.logScale1 = g.logScale[1]; this.logScale2 = g.logScale[2];
    this.rotW = g.rot[0]; this.rotX = g.rot[1]; this.rotY = g.rot[2]; this.rotZ = g.rot[3];
  }

  writeTo(dv: DataView, offset: number): void {
    const v = [
      this.x, this.y, this.z,
      this.nx, this.ny, this.nz,
      this.dc0, this.dc1, this.dc2,
      this.opacityLogit,
      this.logScale0, this.logScale1, this.logScale2,
      this.rotW, this.rotX, this.rotY, this.rotZ,
    ];
    for (let i = 0; i < v.length; i++) dv.setFloat32(offset + i * 4, v[i] as number, true);
  }
}

/**
 * サーフェルの法線から 3DGS 用のクォータニオンを作る。
 *
 * サーフェルは法線方向に厚みを持たないので、接平面内の回転は等方なら任意。
 * ここでは「+Z 軸を法線に合わせる最短回転」を使う。
 *
 * 入力は単位ベクトルを想定するが、法線は深度マップの平面フィットから来るので
 * 微小にずれうる。書き出しは毎フレームの処理ではないので、防御的に正規化する。
 */
export function quatFromNormal(nx: number, ny: number, nz: number): [number, number, number, number] {
  const len = Math.hypot(nx, ny, nz);
  if (len < 1e-12) return [1, 0, 0, 0];
  nx /= len;
  ny /= len;
  nz /= len;

  // v = +Z、u = n としたときの最短回転
  const d = nz; // dot(+Z, n)
  if (d > 0.999999) return [1, 0, 0, 0];
  if (d < -0.999999) return [0, 1, 0, 0]; // 180度反転。軸は +X で任意
  // axis = cross(+Z, n) = (-ny, nx, 0)
  const ax = -ny;
  const ay = nx;
  const s = Math.sqrt((1 + d) * 2);
  return [s / 2, ax / s, ay / s, 0];
}

/**
 * サーフェルの厚みをゼロにすると行列が特異になり、ビューアによっては
 * 数値的に不安定になる。exp(-8) ≈ 0.00034（被写体サイズの 0.03%）を入れておくと、
 * 視覚的には平たいまま計算は安定する（docs/05 §5.3.2）。
 */
export const SURFEL_THICKNESS_LOG = -8;

export const plyByteLength = (count: number): number =>
  new TextEncoder().encode(plyHeader(count)).length + count * BYTES_PER_VERTEX;
