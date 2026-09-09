/**
 * 複数枚モードの剛体変換（docs/12 §12.7）。
 *
 * カメラ座標は既存のパイプラインと同じ約束に従う（src/pipeline/6-splats.ts の `unproject`）。
 *
 *   x = (u − cx) · z / f   … 画像の右が +x
 *   y = (v − cy) · z / f   … 画像の**下**が +y
 *   z                      … カメラから見て奥が +z
 *
 * したがって**鉛直軸は y 軸**（上が −y）で、「被写体が回った」は y 軸まわりの回転になる。
 * 素材ではカメラが1画素も動いていないことを実測で確かめてある（docs/12 §12.2 O1）ので、
 * この当て方には裏づけがある。ただしカメラが動いた3枚も入りうるので、
 * pitch・roll・平行移動も未知数として持つ（docs/12 §12.14 R21）。
 *
 * ## yaw の符号
 *
 * `ViewPose` は **その view の点を基準 view の座標へ運ぶ**変換である。
 * 「画面の右を向いた写真（右向き）は yaw = +90°」と決める（docs/12 §12.2）。
 *
 * 確かめ方: 右向きの写真では被写体は +x（画面の右）を向いている。
 * R_y(+90°) を (1,0,0) に掛けると (0,0,−1) すなわちカメラの方向になり、
 * 基準 view で正面を向く。向きが合う。
 *
 * ## 回転の中心（実装で踏んだ穴）
 *
 * **回転は原点（カメラの中心）まわりではなく、被写体の軸まわりである。**
 * カメラから 3m 先に立っている人が回っても、その人はカメラの位置を中心には回らない。
 * 原点まわりで回すと、被写体は 3m 先から横 3m へ飛んでいってしまう
 * （最初の実装がこれで、投影が全部カメラの後ろに落ちた）。
 *
 * そこで変換は「重心を中心に回して、基準 view の重心へ運ぶ」形にする。
 *
 *   world = scale · R · (X − source) + target + t
 *
 * `source` はその view の点の重心、`target` は基準 view の点の重心。
 * これが docs/12 §12.7 の「回転軸の水平位置」にあたり、`t` はその残りを吸収する
 * 小さな平行移動になる。基準 view では source = target なので恒等変換に落ちる。
 */

/** 画像平面と3Dを結ぶ内部パラメータ。docs/02 §2.4 の `camera` と同じ。 */
export interface CameraIntrinsics {
  /** 焦点距離（画素）。 */
  readonly focalPx: number;
  readonly cx: number;
  readonly cy: number;
}

/** UI の枠（docs/12 D24）。利用者がどれに何を入れたかを、そのまま型にする。 */
export type ViewSlot = 'front' | 'right' | 'left';

/**
 * view の点を基準 view の座標へ運ぶ、尺度つきの剛体変換。
 *
 * `world = scale · R(yaw, pitch, roll) · X + t`
 */
export interface ViewPose {
  /** 鉛直軸まわり[rad]。主要な未知数。 */
  readonly yaw: number;
  /** x 軸まわり[rad]。カメラの上下の傾き。小さい値を想定。 */
  readonly pitch: number;
  /** z 軸まわり[rad]。カメラのねじれ。小さい値を想定。 */
  readonly roll: number;
  /** 深度モデルの尺度ずれを吸収する。1 付近。 */
  readonly scale: number;
  readonly tx: number;
  readonly ty: number;
  readonly tz: number;
}

/** 基準 view（動かさない）。 */
export const REFERENCE_POSE: ViewPose = {
  yaw: 0,
  pitch: 0,
  roll: 0,
  scale: 1,
  tx: 0,
  ty: 0,
  tz: 0,
};

/**
 * 枠から yaw の初期値を決める（docs/12 §12.7）。
 *
 * 利用者が手で指定する（D24）ので、初期値は入力の時点で分かっている。
 * 顔ランドマークやシルエットは、この ±90° を実際の角度へ寄せる役に降りる。
 */
export function slotYaw(slot: ViewSlot): number {
  switch (slot) {
    case 'front':
      return 0;
    case 'right':
      return Math.PI / 2;
    case 'left':
      return -Math.PI / 2;
  }
}

/** 3×3 を行優先で持つ。 */
export type Mat3 = Float64Array;

/**
 * R = R_y(yaw) · R_x(pitch) · R_z(roll)。
 *
 * 順序を変えると pitch・roll の意味が変わるが、どちらも小さい値を想定しているので
 * 実用上の差は出ない。ここで一度決めて、以降は変えない。
 */
export function rotationMatrix(pose: ViewPose, out?: Mat3): Mat3 {
  const m = out ?? new Float64Array(9);
  const cy = Math.cos(pose.yaw);
  const sy = Math.sin(pose.yaw);
  const cp = Math.cos(pose.pitch);
  const sp = Math.sin(pose.pitch);
  const cr = Math.cos(pose.roll);
  const sr = Math.sin(pose.roll);

  // R_x(pitch) · R_z(roll)
  const a00 = cr;
  const a01 = -sr;
  const a02 = 0;
  const a10 = cp * sr;
  const a11 = cp * cr;
  const a12 = -sp;
  const a20 = sp * sr;
  const a21 = sp * cr;
  const a22 = cp;

  // R_y(yaw) · A
  m[0] = cy * a00 + sy * a20;
  m[1] = cy * a01 + sy * a21;
  m[2] = cy * a02 + sy * a22;
  m[3] = a10;
  m[4] = a11;
  m[5] = a12;
  m[6] = -sy * a00 + cy * a20;
  m[7] = -sy * a01 + cy * a21;
  m[8] = -sy * a02 + cy * a22;
  return m;
}

/**
 * 回転の中心。`source` はその view の点の重心、`target` は基準 view の点の重心。
 * 詳しくは冒頭の「回転の中心」。
 */
export interface PoseFrame {
  readonly source: readonly [number, number, number];
  readonly target: readonly [number, number, number];
}

/** 基準 view 用（重心を動かさない）。 */
export const IDENTITY_FRAME: PoseFrame = { source: [0, 0, 0], target: [0, 0, 0] };

/** `world = scale · R · (X − source) + target + t`。`out` に書き、同じ配列を返す。 */
export function toReference(
  R: Mat3,
  pose: ViewPose,
  frame: PoseFrame,
  x: number,
  y: number,
  z: number,
  out: Float64Array,
): Float64Array {
  const s = pose.scale;
  const px = x - frame.source[0];
  const py = y - frame.source[1];
  const pz = z - frame.source[2];
  out[0] =
    s * ((R[0] as number) * px + (R[1] as number) * py + (R[2] as number) * pz) + frame.target[0] + pose.tx;
  out[1] =
    s * ((R[3] as number) * px + (R[4] as number) * py + (R[5] as number) * pz) + frame.target[1] + pose.ty;
  out[2] =
    s * ((R[6] as number) * px + (R[7] as number) * py + (R[8] as number) * pz) + frame.target[2] + pose.tz;
  return out;
}

/** `toReference` の逆。基準 view の点を、その view のカメラ座標へ戻す。 */
export function fromReference(
  R: Mat3,
  pose: ViewPose,
  frame: PoseFrame,
  x: number,
  y: number,
  z: number,
  out: Float64Array,
): Float64Array {
  const dx = (x - frame.target[0] - pose.tx) / pose.scale;
  const dy = (y - frame.target[1] - pose.ty) / pose.scale;
  const dz = (z - frame.target[2] - pose.tz) / pose.scale;
  // R は回転なので転置が逆行列
  out[0] = (R[0] as number) * dx + (R[3] as number) * dy + (R[6] as number) * dz + frame.source[0];
  out[1] = (R[1] as number) * dx + (R[4] as number) * dy + (R[7] as number) * dz + frame.source[1];
  out[2] = (R[2] as number) * dx + (R[5] as number) * dy + (R[8] as number) * dz + frame.source[2];
  return out;
}

/** 画素と深度から、そのカメラの3D座標へ。 */
export function unproject(
  u: number,
  v: number,
  z: number,
  cam: CameraIntrinsics,
  out: Float64Array,
): Float64Array {
  out[0] = ((u - cam.cx) * z) / cam.focalPx;
  out[1] = ((v - cam.cy) * z) / cam.focalPx;
  out[2] = z;
  return out;
}

/**
 * カメラ座標から画素へ。
 *
 * `z <= 0`（カメラの後ろ）のときは投影が定義できないので `false` を返す。
 * 呼び出し側はこれを「合っていない」として罰する。
 */
export function project(
  x: number,
  y: number,
  z: number,
  cam: CameraIntrinsics,
  out: Float64Array,
): boolean {
  if (!(z > 1e-6)) return false;
  out[0] = cam.cx + (cam.focalPx * x) / z;
  out[1] = cam.cy + (cam.focalPx * y) / z;
  out[2] = z;
  return true;
}
