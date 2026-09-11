/**
 * Ⓒ 合成の検算（docs/12 §12.8）。
 *
 * パック済みの 24 バイト × N を解いて座標を移し、詰め直す経路なので、
 * **解いて詰め直しただけで形が変わっていないこと**を最初に確かめる。
 * ここがずれると、位置合わせが正しくても立体が壊れる。
 *
 * 運ぶところ（`dedupe: false`）と、重複を落とすところを分けて見る。
 * 混ぜて測ると、座標が狂ったのか落としすぎたのかが分からなくなる。
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

/** 法線を差し替える（表と裏、見込み角の試験のため）。 */
function withNormal(b: SplatBuild, nx: number, ny: number, nz: number): SplatBuild {
  const u32 = new Uint32Array(b.data.buffer, b.data.byteOffset, b.count * STRIDE32);
  const l = Math.hypot(nx, ny, nz) || 1;
  for (let i = 0; i < b.count; i++) u32[i * STRIDE32 + 3] = encodeOct(nx / l, ny / l, nz / l);
  return b;
}

/** 色を差し替える（どちらが残ったかを見分けるため）。 */
function withRgba(b: SplatBuild, r: number, g: number, bl: number): SplatBuild {
  const u32 = new Uint32Array(b.data.buffer, b.data.byteOffset, b.count * STRIDE32);
  for (let i = 0; i < b.count; i++) u32[i * STRIDE32 + 5] = packRgba8(r, g, bl, 255);
  return b;
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
  it('1枚だけなら、中身はそのまま（内訳だけ添える）', () => {
    const b = makeBuild(CUBE);
    const out = mergeBuilds([{ build: b, pose: REFERENCE_POSE, frame: IDENTITY_FRAME }]);
    expect(out.data).toBe(b.data); // 詰め直さない
    expect(out.count).toBe(b.count);
    expect(out.normalization).toBe(b.normalization);
    expect(out.mergeStats).toEqual({ before: 8, after: 8, dropped: 0, cell: 0 });
  });

  it('恒等の姿勢で2枚重ねても、形が変わらない（落とさない設定）', () => {
    const b = makeBuild(CUBE);
    const out = mergeBuilds(
      [
        { build: b, pose: REFERENCE_POSE, frame: IDENTITY_FRAME },
        { build: b, pose: REFERENCE_POSE, frame: IDENTITY_FRAME },
      ],
      { dedupe: false },
    );
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
    const out = mergeBuilds(
      [
        { build: b, pose: REFERENCE_POSE, frame: IDENTITY_FRAME },
        { build: b, pose, frame },
      ],
      { dedupe: false },
    );

    const R = rotationMatrix(pose);
    const tmp = new Float64Array(3);
    for (let i = 0; i < CUBE.length; i++) {
      const c = CUBE[i] as [number, number, number];
      toReference(R, pose, frame, c[0], c[1], c[2], tmp);
      const got = readWorld(out, CUBE.length + i);
      for (let k = 0; k < 3; k++) expect(got[k]).toBeCloseTo(tmp[k] as number, 4);
    }
  });

  it('個数の内訳は足し合わされる（落とさない設定）', () => {
    const b = makeBuild(CUBE);
    const out = mergeBuilds(
      [
        { build: b, pose: REFERENCE_POSE, frame: IDENTITY_FRAME },
        { build: b, pose: REFERENCE_POSE, frame: IDENTITY_FRAME },
        { build: b, pose: REFERENCE_POSE, frame: IDENTITY_FRAME },
      ],
      { dedupe: false },
    );
    expect(out.frontCount).toBe(CUBE.length * 3);
    expect(out.count).toBe(CUBE.length * 3);
  });
});

/**
 * 重複除去（docs/12 §12.8）。
 *
 * **落としてよいのは view をまたいだ重なりだけ**である。同じ view の中で
 * 間引いてしまうと、重複除去ではなく単なる劣化になる。ここを試験で固定する。
 */
describe('重なった面を落とす', () => {
  it('同じものを3枚重ねたら、1枚ぶんに戻る', () => {
    const b = makeBuild(CUBE);
    const out = mergeBuilds([
      { build: b, pose: REFERENCE_POSE, frame: IDENTITY_FRAME },
      { build: b, pose: REFERENCE_POSE, frame: IDENTITY_FRAME },
      { build: b, pose: REFERENCE_POSE, frame: IDENTITY_FRAME },
    ]);
    expect(out.count).toBe(CUBE.length);
    expect(out.mergeStats.before).toBe(CUBE.length * 3);
    expect(out.mergeStats.dropped).toBe(CUBE.length * 2);
    // 内訳も数え直されている（足し算のままにしない）
    expect(out.frontCount).toBe(CUBE.length);
  });

  it('同じ view の中では間引かない', () => {
    // 半径よりずっと近い点を並べた1枚。重複除去は手を出してはいけない。
    const dense: [number, number, number][] = [];
    for (let i = 0; i < 40; i++) dense.push([i * 0.001, 0, 3.0]);
    const a = makeBuild(dense, 0.01);
    const b = makeBuild([[0, 0.5, 3.0]], 0.01);
    const out = mergeBuilds([
      { build: a, pose: REFERENCE_POSE, frame: IDENTITY_FRAME },
      { build: b, pose: REFERENCE_POSE, frame: IDENTITY_FRAME },
    ]);
    expect(out.count).toBe(dense.length + 1);
    expect(out.mergeStats.dropped).toBe(0);
  });

  it('向きが違う面は、同じ場所にあっても残す（体の表と裏）', () => {
    // 同じ座標に、法線だけ逆向きの点を2つ。薄い部位の表と裏に当たる。
    const front = makeBuild([[0, 0, 3.0]]);
    const back = withNormal(makeBuild([[0, 0, 3.0]]), 0, 0, 1);
    const out = mergeBuilds([
      { build: front, pose: REFERENCE_POSE, frame: IDENTITY_FRAME },
      { build: back, pose: REFERENCE_POSE, frame: IDENTITY_FRAME },
    ]);
    expect(out.count).toBe(2);
    expect(out.mergeStats.dropped).toBe(0);
  });

  it('残るのは、その面をより正面から見ていた view の点', () => {
    // 同じ場所・同じ向きの面を2つ。片方は法線が z にまっすぐ（正面から見た）、
    // もう片方は大きく傾いている（かすめて見た）。色で見分ける。
    const straight = withRgba(withNormal(makeBuild([[0, 0, 3.0]]), 0, 0, -1), 1, 2, 3);
    // 30° 傾ける。内積 0.866 で「同じ面」の範囲（既定 0.7）に入る。
    const oblique = withRgba(withNormal(makeBuild([[0, 0, 3.0]]), 0.5, 0, -0.866), 200, 100, 50);
    // 入れる順を変えても、残るほうは変わらない
    for (const order of [
      [straight, oblique],
      [oblique, straight],
    ]) {
      const out = mergeBuilds(
        order.map((build) => ({ build, pose: REFERENCE_POSE, frame: IDENTITY_FRAME })),
      );
      expect(out.count).toBe(1);
      const u32 = new Uint32Array(out.data.buffer, out.data.byteOffset, STRIDE32);
      const rgba = u32[5] as number;
      expect(rgba & 0xff).toBe(1); // まっすぐ見たほうの色が残る
    }
  });
});
