/**
 * 横断面の凹みを人体としてあり得る範囲に収める（docs/09 §V24）。
 *
 * 手前側の凸包から上限より奥へ落ちた所だけを詰める。凸包そのものへは戻さない。
 */
import { describe, expect, it } from 'vitest';
import {
  limitCrossSectionDent,
  DEFAULT_CROSS_SECTION,
} from '../../src/pipeline/geometry/crossSection';

const W = 240;
const H = 200;
const FOCAL = 500;
const Z = 1.0;
const BODY_PX = 160;
const BODY_METRIC = (BODY_PX * Z) / FOCAL;
const X0 = 40;

function scene(): { depth: Float32Array; alpha: Uint8ClampedArray } {
  const depth = new Float32Array(W * H).fill(Z);
  const alpha = new Uint8ClampedArray(W * H);
  for (let y = 20; y < 180; y++) {
    for (let x = X0; x < X0 + BODY_PX; x++) alpha[y * W + x] = 255;
  }
  return { depth, alpha };
}

/** 区間の中央に、幅 `w` px・深さ `depth` の溝を掘る。 */
function groove(depth: Float32Array, halfWidth: number, deep: number): void {
  const cx = X0 + BODY_PX / 2;
  for (let y = 0; y < H; y++) {
    for (let x = cx - halfWidth; x < cx + halfWidth; x++) {
      const t = Math.abs(x - cx) / halfWidth;
      depth[y * W + x] = Z + deep * (1 - t * t);
    }
  }
}

describe('横断面の凹みを上限まで詰める', () => {
  it('深すぎる溝は上限まで浅くなる', () => {
    const { depth, alpha } = scene();
    // 体幅の 30%（上限 10% の 3 倍）の深さ
    groove(depth, 30, 0.3 * BODY_METRIC);
    const before = (depth[100 * W + X0 + BODY_PX / 2] as number) - Z;

    const out = limitCrossSectionDent(depth, alpha, W, H, FOCAL);
    const after = (out[100 * W + X0 + BODY_PX / 2] as number) - Z;
    expect(before / BODY_METRIC).toBeCloseTo(0.3, 2);
    expect(
      after / BODY_METRIC,
      `溝が体幅の ${(after / BODY_METRIC).toFixed(3)} 残っています`,
    ).toBeLessThan(DEFAULT_CROSS_SECTION.maxDentRatio * 1.05);
    expect(after, '溝が消えています（詰めるだけで消してはいけない）').toBeGreaterThan(0);
  });

  it('浅い溝は素通しする', () => {
    const { depth, alpha } = scene();
    // 上限 10% の半分（5%）より浅い
    groove(depth, 30, 0.04 * BODY_METRIC);
    const out = limitCrossSectionDent(depth, alpha, W, H, FOCAL);
    for (let x = X0 + 50; x < X0 + 110; x++) {
      expect(out[100 * W + x], `x=${x} の本物の凹みが削られています`).toBeCloseTo(
        depth[100 * W + x] as number,
        6,
      );
    }
  });

  it('凸な断面は動かさない', () => {
    const { depth, alpha } = scene();
    for (let y = 0; y < H; y++) {
      for (let x = X0; x < X0 + BODY_PX; x++) {
        const t = (x - (X0 + BODY_PX / 2)) / (BODY_PX / 2);
        depth[y * W + x] = Z - 0.2 * BODY_METRIC * Math.max(0, 1 - t * t);
      }
    }
    const out = limitCrossSectionDent(depth, alpha, W, H, FOCAL);
    for (let i = 0; i < depth.length; i += 7) {
      expect(out[i], `i=${i} が動いています`).toBeCloseTo(depth[i] as number, 6);
    }
  });

  it('被写体の外は動かさない', () => {
    const { depth, alpha } = scene();
    groove(depth, 30, 0.3 * BODY_METRIC);
    const out = limitCrossSectionDent(depth, alpha, W, H, FOCAL);
    for (let i = 0; i < depth.length; i += 11) {
      if ((alpha[i] as number) >= 128) continue;
      expect(out[i]).toBe(depth[i]);
    }
  });

  it('外す四角の中は触らない（顔）', () => {
    const { depth, alpha } = scene();
    groove(depth, 30, 0.3 * BODY_METRIC);
    const head = { x: 0, y: 0, width: W, height: 120 };
    const out = limitCrossSectionDent(depth, alpha, W, H, FOCAL, DEFAULT_CROSS_SECTION, head);
    const cx = X0 + BODY_PX / 2;
    expect(out[60 * W + cx], '外した四角の中が動いています').toBeCloseTo(
      depth[60 * W + cx] as number,
      6,
    );
    expect(out[150 * W + cx], '四角の外が動いていません').toBeLessThan(
      depth[150 * W + cx] as number,
    );
  });

  it('短すぎる区間は触らない', () => {
    const depth = new Float32Array(W * H).fill(Z);
    const alpha = new Uint8ClampedArray(W * H);
    for (let y = 40; y < 120; y++) for (let x = 100; x < 120; x++) alpha[y * W + x] = 255;
    for (let y = 40; y < 120; y++) depth[y * W + 110] = Z + 0.3 * BODY_METRIC;
    const out = limitCrossSectionDent(depth, alpha, W, H, FOCAL);
    expect(out[80 * W + 110]).toBe(depth[80 * W + 110]);
  });

  it('上限 0 なら何もしない', () => {
    const { depth, alpha } = scene();
    groove(depth, 30, 0.3 * BODY_METRIC);
    const out = limitCrossSectionDent(depth, alpha, W, H, FOCAL, {
      ...DEFAULT_CROSS_SECTION,
      maxDentRatio: 0,
    });
    for (let i = 0; i < depth.length; i += 13) expect(out[i]).toBe(depth[i]);
  });
});
