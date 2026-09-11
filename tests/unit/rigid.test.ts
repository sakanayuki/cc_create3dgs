/**
 * 剛体変換の約束を、言葉で書いたとおりに固定する（docs/12 §12.7）。
 *
 * 符号を1つ取り違えると、位置合わせは「必ず 180° ずれた解」に落ちる。
 * しかも見た目は「それらしく」動くので気づきにくい。ここで釘を打っておく。
 */
import { describe, expect, it } from 'vitest';
import {
  fromReference,
  IDENTITY_FRAME,
  project,
  REFERENCE_POSE,
  rotationMatrix,
  slotYaw,
  toReference,
  unproject,
  type ViewPose,
} from '../../src/pipeline/align/rigid';

const CAM = { focalPx: 800, cx: 100, cy: 160 };

describe('剛体変換の約束', () => {
  it('枠から決まる yaw の初期値', () => {
    expect(slotYaw('front')).toBe(0);
    expect(slotYaw('right')).toBeCloseTo(Math.PI / 2, 12);
    expect(slotYaw('left')).toBeCloseTo(-Math.PI / 2, 12);
  });

  it('右向き（yaw = +90°）は、被写体の正面をカメラの方へ戻す', () => {
    // 右向きの写真では被写体は +x（画面の右）を向いている。
    // その向きベクトルを基準 view へ運ぶと、−z（カメラの方）になるはず。
    const pose: ViewPose = { ...REFERENCE_POSE, yaw: Math.PI / 2 };
    const R = rotationMatrix(pose);
    const out = new Float64Array(3);
    toReference(R, pose, IDENTITY_FRAME, 1, 0, 0, out);
    expect(out[0]).toBeCloseTo(0, 12);
    expect(out[1]).toBeCloseTo(0, 12);
    expect(out[2]).toBeCloseTo(-1, 12);
  });

  it('左向き（yaw = −90°）は逆向きに戻す', () => {
    const pose: ViewPose = { ...REFERENCE_POSE, yaw: -Math.PI / 2 };
    const R = rotationMatrix(pose);
    const out = new Float64Array(3);
    toReference(R, pose, IDENTITY_FRAME, -1, 0, 0, out);
    expect(out[2]).toBeCloseTo(-1, 12);
  });

  it('鉛直軸まわりの回転は y を動かさない', () => {
    const pose: ViewPose = { ...REFERENCE_POSE, yaw: 0.7 };
    const R = rotationMatrix(pose);
    const out = new Float64Array(3);
    toReference(R, pose, IDENTITY_FRAME, 0.3, 1.25, -0.4, out);
    expect(out[1]).toBeCloseTo(1.25, 12);
  });

  it('toReference と fromReference は互いに逆', () => {
    const pose: ViewPose = { yaw: 0.6, pitch: 0.05, roll: -0.03, scale: 1.17, tx: 0.2, ty: -0.1, tz: 0.4 };
    const R = rotationMatrix(pose);
    const a = new Float64Array(3);
    const b = new Float64Array(3);
    const frame = { source: [0.05, -0.02, 3.1] as const, target: [-0.01, 0.03, 3.0] as const };
    toReference(R, pose, frame, 0.11, -0.22, 3.3, a);
    fromReference(R, pose, frame, a[0] as number, a[1] as number, a[2] as number, b);
    expect(b[0]).toBeCloseTo(0.11, 10);
    expect(b[1]).toBeCloseTo(-0.22, 10);
    expect(b[2]).toBeCloseTo(3.3, 10);
  });

  it('回転の中心は原点ではなく重心（実装で踏んだ穴）', () => {
    // カメラから 3.2m 先に立っている人が回っても、カメラの位置を中心には回らない。
    // 重心を中心に回すので、重心そのものは動かない。
    const pose: ViewPose = { ...REFERENCE_POSE, yaw: Math.PI / 2 };
    const R = rotationMatrix(pose);
    const c = [0, 0, 3.2] as const;
    const frame = { source: c, target: c };
    const out = new Float64Array(3);
    toReference(R, pose, frame, c[0], c[1], c[2], out);
    expect(out[0]).toBeCloseTo(0, 12);
    expect(out[2]).toBeCloseTo(3.2, 12);

    // 重心から 0.1m 手前（カメラ側）の点は、90° 回すと 0.1m 右へ来る。
    toReference(R, pose, frame, 0, 0, 3.1, out);
    expect(out[0]).toBeCloseTo(-0.1, 12);
    expect(out[2]).toBeCloseTo(3.2, 12);
  });

  it('unproject と project は互いに逆', () => {
    const p = new Float64Array(3);
    const q = new Float64Array(3);
    unproject(37.5, 210.5, 2.8, CAM, p);
    expect(project(p[0] as number, p[1] as number, p[2] as number, CAM, q)).toBe(true);
    expect(q[0]).toBeCloseTo(37.5, 9);
    expect(q[1]).toBeCloseTo(210.5, 9);
  });

  it('カメラの後ろは投影できない', () => {
    const q = new Float64Array(3);
    expect(project(0.1, 0.1, -1, CAM, q)).toBe(false);
    expect(project(0.1, 0.1, 0, CAM, q)).toBe(false);
  });
});
