/**
 * 細部の起伏の振幅を人体らしい範囲に収める（docs/09 §V21）。
 *
 * 大域（体の形）は触らず、大域からの差だけを上限まで詰める。
 */
import { describe, expect, it } from 'vitest';
import {
  clampReliefAmplitude,
  DEFAULT_RELIEF_CLAMP,
  type ReliefClampParams,
} from '../../src/pipeline/geometry/reliefClamp';

const W = 160;
const H = 160;
const FOCAL = 500;
const Z = 1.0;
/** 被写体は幅 100px。焦点 500・距離 1 なので、体幅は実寸 0.2。 */
const BODY_PX = 100;
const BODY_METRIC = (BODY_PX * Z) / FOCAL;

/** 素通しの設定（雑音均しを切って、振幅の詰めだけを見る）。 */
const CLIP_ONLY: ReliefClampParams = { ...DEFAULT_RELIEF_CLAMP, denoiseRadius: 0 };

function scene(): { depth: Float32Array; alpha: Uint8ClampedArray } {
  const depth = new Float32Array(W * H).fill(Z);
  const alpha = new Uint8ClampedArray(W * H);
  for (let y = 20; y < 140; y++) {
    for (let x = 30; x < 30 + BODY_PX; x++) alpha[y * W + x] = 255;
  }
  return { depth, alpha };
}

describe('起伏の振幅を体らしい範囲に収める', () => {
  it('行き過ぎた起伏は上限まで詰まる', () => {
    const { depth, alpha } = scene();
    // 周期 7px の波。大域の窓（半径 10 → 幅 21px）はちょうど 3 周期ぶんなので、
    // 窓の平均は 0 になる。つまり波はまるごと「細部」に入り、上限が素直に効く。
    const amp = 0.05 * BODY_METRIC; // 上限 0.8% の 6 倍
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) depth[y * W + x] = Z + amp * Math.cos((2 * Math.PI * x) / 7);
    }

    const out = clampReliefAmplitude(depth, alpha, W, H, FOCAL, CLIP_ONLY);
    let lo = Infinity;
    let hi = -Infinity;
    for (let x = 60; x < 100; x++) {
      const v = out[80 * W + x] as number;
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
    const half = (hi - lo) / 2 / BODY_METRIC;
    expect(half, `波の振幅が体幅の ${half.toFixed(4)} 残っています`).toBeLessThan(
      DEFAULT_RELIEF_CLAMP.maxDetailRatio * 1.05,
    );
    expect(half, '波が消えています（詰めるだけで消してはいけない）').toBeGreaterThan(
      DEFAULT_RELIEF_CLAMP.maxDetailRatio * 0.5,
    );
  });

  it('上限の半分までの皺は素通しする', () => {
    const { depth, alpha } = scene();
    const fold = 0.003 * BODY_METRIC; // 上限 0.8% の半分（0.4%）より小さい
    for (let y = 20; y < 140; y++) depth[y * W + 60] = Z - fold;

    const out = clampReliefAmplitude(depth, alpha, W, H, FOCAL, CLIP_ONLY);
    expect(out[80 * W + 60], '本物の皺が削られています').toBeCloseTo(Z - fold, 6);
  });

  it('大域の形は残る', () => {
    const { depth, alpha } = scene();
    // 体幅の 20% の振幅で、体をまたぐ緩やかな山（＝体の形）。
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const t = (x - (30 + BODY_PX / 2)) / (BODY_PX / 2);
        depth[y * W + x] = Z - 0.2 * BODY_METRIC * Math.max(0, 1 - t * t);
      }
    }
    const before = (depth[80 * W + 50] as number) - (depth[80 * W + 80] as number);
    const out = clampReliefAmplitude(depth, alpha, W, H, FOCAL, CLIP_ONLY);
    const after = (out[80 * W + 50] as number) - (out[80 * W + 80] as number);
    expect(after / before, `大域が ${(after / before).toFixed(3)} 倍に潰れています`).toBeGreaterThan(
      0.9,
    );
  });

  it('被写体の外は動かさない', () => {
    const { depth, alpha } = scene();
    for (let i = 0; i < depth.length; i++) depth[i] = Z + (i % 7) * 0.01;
    const out = clampReliefAmplitude(depth, alpha, W, H, FOCAL);
    for (let i = 0; i < depth.length; i += 11) {
      if ((alpha[i] as number) >= 128) continue;
      expect(out[i]).toBe(depth[i]);
    }
  });

  it('外す四角の中は触らない（顔）', () => {
    const { depth, alpha } = scene();
    const spike = 0.05 * BODY_METRIC;
    for (let y = 24; y < 60; y++) depth[y * W + 79] = Z - spike;
    const head = { x: 60, y: 20, width: 40, height: 44 };
    const out = clampReliefAmplitude(depth, alpha, W, H, FOCAL, CLIP_ONLY, head);
    expect(out[40 * W + 79], '顔の中が詰められています').toBeCloseTo(Z - spike, 6);
  });

  it('雑音均しは画素単位のざらつきを落とす', () => {
    const { depth, alpha } = scene();
    for (let i = 0; i < depth.length; i++) depth[i] = Z + ((i % 2 === 0 ? 1 : -1) * 0.002);
    const out = clampReliefAmplitude(depth, alpha, W, H, FOCAL);
    let rough = 0;
    let n = 0;
    for (let y = 40; y < 120; y++) {
      for (let x = 40; x < 120; x++) {
        rough += Math.abs((out[y * W + x] as number) - Z);
        n++;
      }
    }
    expect(rough / n, 'ざらつきが残っています').toBeLessThan(0.0005);
  });

  it('焦点距離が 0 なら何もしない', () => {
    const { depth, alpha } = scene();
    const out = clampReliefAmplitude(depth, alpha, W, H, 0);
    for (let i = 0; i < depth.length; i += 13) expect(out[i]).toBe(depth[i]);
  });
});
