/**
 * Ⓒ 合成の検算（docs/12 §12.8）。
 *
 * パック済みの 24 バイト × N を解いて座標を移し、詰め直す経路なので、
 * **解いて詰め直しただけで形が変わっていないこと**を最初に確かめる。
 * ここがずれると、位置合わせが正しくても立体が壊れる。
 */
import { describe, expect, it } from 'vitest';
import { mergeBuilds } from '../../src/pipeline/align/mergeViews';
import { REFERENCE_POSE, IDENTITY_FRAME, rotationMatrix, toReference } from '../../src/pipeline/align/rigid';
import type { SplatBuild } from '../../src/pipeline/6-splats';
import { encodeOct, packHalf2, packRgba8 } from '../../src/codec/pack';

const STRIDE32 = 6;

/** カメラ座標の点から、buildSplats と同じ詰め方で SplatBuild を作る。 */
function makeBuild(points: readonly [number, number, number][], radius = 0.01): SplatBuild {
  let minX = Infinity, minY = Infinity, minZ = Infinity;
  let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
  for (const p of points) {
    minX = Math.min(minX, p[0]); maxX = Math.max(maxX, p[0]);
    minY = Math.min(minY, p[1]); maxY = Math.max(maxY, p[1]);
    minZ = Math.min(minZ, p[2]); maxZ = Math.max(maxZ, p[2]);
  }
  const center: [number, number, number] = [(minX + maxX) / 2, (minY + maxY) / 2, (minZ + maxZ) / 2];
  const scale = 1 / Math.max(maxX - minX, maxY - minY, maxZ - minZ, 1e-6);
  const buf = new ArrayBuffer(points.length * 24);
  const f32 = new Float32Array(buf);
  const u32 = new Uint32Array(buf);
  points.forEach((p, i) => {
    const o = i * STRIDE32;
    f32[o] = (p[0] - center[0]) * scale;
    f32[o + 1] = (p[1] - center[1]) * scale;
    f32[o + 2] = (p[2] - center[2]) * scale;
    u32[o + 3] = encodeOct(0, 0, -1);
    u32[o + 4] = packHalf2(radius * scale, radius * scale);
    u32[o + 5] = packRgba8(10 + i, 20, 30, 255);
  });
  return {
    data: new Uint8Array(buf),
    count: points.length,
    frontCount: points.length,
    backCount: 0,
    skirtCount: 0,
    normalization: { center, scale },
    metricHeight: maxY - minY,
    nearZ: 0.5,
    farZ: 1.5,
  };
}

/** 正規化を戻してカメラ座標に読む。 */
function readWorld(b: SplatBuild, i: number): [number, number, number] {
  const f32 = new Float32Array(b.data.buffer, b.data.byteOffset, b.count * STRIDE32);
  const inv = 1 / b.normalization.scale;
  const c = b.normalization.center;
  return [
    (f32[i * STRIDE32] as number) * inv + c[0],
    (f32[i * STRIDE32 + 1] as number) * inv + c[1],
    (f32[i * STRIDE32 + 2] as number) * inv + c[2],
  ];
}

const CUBE: [number, number, number][] = [
  [-0.1, -0.2, 3.0], [0.1, -0.2, 3.0], [-0.1, 0.2, 3.0], [0.1, 0.2, 3.0],
  [-0.1, -0.2, 3.2], [0.1, -0.2, 3.2], [-0.1, 0.2, 3.2], [0.1, 0.2, 3.2],
];

describe('複数 view の合成', () => {
  it('1枚だけなら、そのまま返す', () => {
    const b = makeBuild(CUBE);
    const out = mergeBuilds([{ build: b, pose: REFERENCE_POSE, frame: IDENTITY_FRAME }]);
    expect(out).toBe(b);
  });

  it('恒等の姿勢で2枚重ねても、形が変わらない', () => {
    const b = makeBuild(CUBE);
    const out = mergeBuilds([
      { build: b, pose: REFERENCE_POSE, frame: IDENTITY_FRAME },
      { build: b, pose: REFERENCE_POSE, frame: IDENTITY_FRAME },
    ]);
    expect(out.count).toBe(CUBE.length * 2);
    // 同じ点が2つずつ入るだけなので、外接は元と同じ
    for (let i = 0; i < CUBE.length; i++) {
      const got = readWorld(out, i);
      const want = CUBE[i] as [number, number, number];
      for (let k = 0; k < 3; k++) expect(got[k]).toBeCloseTo(want[k] as number, 4);
    }
  });

  it('90° 回した view が、基準の座標へ正しく運ばれる', () => {
    const b = makeBuild(CUBE);
    const pose = { ...REFERENCE_POSE, yaw: Math.PI / 2 };
    const frame = { source: [0, 0, 3.1] as const, target: [0, 0, 3.1] as const };
    const out = mergeBuilds([
      { build: b, pose: REFERENCE_POSE, frame: IDENTITY_FRAME },
      { build: b, pose, frame },
    ]);

    const R = rotationMatrix(pose);
    const tmp = new Float64Array(3);
    for (let i = 0; i < CUBE.length; i++) {
      const c = CUBE[i] as [number, number, number];
      toReference(R, pose, frame, c[0], c[1], c[2], tmp);
      const got = readWorld(out, CUBE.length + i);
      for (let k = 0; k < 3; k++) expect(got[k]).toBeCloseTo(tmp[k] as number, 4);
    }
  });

  it('個数の内訳は足し合わされる', () => {
    const b = makeBuild(CUBE);
    const out = mergeBuilds([
      { build: b, pose: REFERENCE_POSE, frame: IDENTITY_FRAME },
      { build: b, pose: REFERENCE_POSE, frame: IDENTITY_FRAME },
      { build: b, pose: REFERENCE_POSE, frame: IDENTITY_FRAME },
    ]);
    expect(out.frontCount).toBe(CUBE.length * 3);
    expect(out.count).toBe(CUBE.length * 3);
  });
});
