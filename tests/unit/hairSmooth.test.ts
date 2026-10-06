/**
 * 髪の深度を均す（docs/13 §13.5）。顔と髪以外は触らないこと、髪の波が
 * 小さくなること、明るい髪では素通しになることを見る。
 */
import { describe, expect, it } from 'vitest';
import { smoothHairDepth } from '../../src/pipeline/geometry/hairSmooth';

const W = 120;
const H = 120;
const BOX = { x: 10, y: 10, width: 100, height: 100 };

/** 中央に顔（明るい）、その左右に髪（暗い）が並ぶ被写体。 */
function scene(hairValue: number): {
  depth: Float32Array;
  rgba: Uint8ClampedArray;
  alpha: Uint8ClampedArray;
  face: Float32Array;
} {
  const depth = new Float32Array(W * H).fill(1);
  const rgba = new Uint8ClampedArray(W * H * 4);
  const alpha = new Uint8ClampedArray(W * H);
  const face = new Float32Array(W * H);
  for (let y = 10; y < 110; y++) {
    for (let x = 10; x < 110; x++) {
      const i = y * W + x;
      alpha[i] = 255;
      const isFace = x >= 40 && x < 80 && y >= 30 && y < 90;
      const v = isFace ? 200 : hairValue;
      rgba[i * 4] = v;
      rgba[i * 4 + 1] = v;
      rgba[i * 4 + 2] = v;
      rgba[i * 4 + 3] = 255;
      if (isFace) face[i] = 1;
      // 髪は行ごとにばらつく深度、顔は一定
      depth[i] = isFace ? 0.5 : 1 + 0.05 * Math.sin(y * 1.7);
    }
  }
  return { depth, rgba, alpha, face };
}

function rowStd(d: Float32Array, x: number): number {
  let m = 0;
  for (let y = 15; y < 105; y++) m += d[y * W + x] as number;
  m /= 90;
  let v = 0;
  for (let y = 15; y < 105; y++) v += ((d[y * W + x] as number) - m) ** 2;
  return Math.sqrt(v / 90);
}

describe('髪の深度を均す', () => {
  it('髪の行ごとのばらつきが小さくなる', () => {
    const s = scene(60);
    const out = smoothHairDepth(s.depth, s.rgba, s.alpha, s.face, W, H, BOX);
    expect(rowStd(out, 20)).toBeLessThan(rowStd(s.depth, 20) * 0.5);
  });

  it('顔と、箱の外には触らない', () => {
    const s = scene(60);
    const out = smoothHairDepth(s.depth, s.rgba, s.alpha, s.face, W, H, BOX);
    for (let y = 30; y < 90; y++) {
      for (let x = 40; x < 80; x++) expect(out[y * W + x]).toBe(s.depth[y * W + x]);
    }
    expect(out[5 * W + 5]).toBe(s.depth[5 * W + 5]);
  });

  it('顔の深度を髪へ混ぜない（顔は 0.5、髪は 1 前後）', () => {
    const s = scene(60);
    const out = smoothHairDepth(s.depth, s.rgba, s.alpha, s.face, W, H, BOX);
    for (let y = 15; y < 105; y++) expect(out[y * W + 38] as number).toBeGreaterThan(0.9);
  });

  it('髪が顔と同じくらい明るければ、素通しにする', () => {
    const s = scene(190);
    const out = smoothHairDepth(s.depth, s.rgba, s.alpha, s.face, W, H, BOX);
    expect(Array.from(out)).toEqual(Array.from(s.depth));
  });

  it('顔が見つかっていなければ（重みが空）、素通しにする', () => {
    const s = scene(60);
    const out = smoothHairDepth(s.depth, s.rgba, s.alpha, new Float32Array(W * H), W, H, BOX);
    expect(Array.from(out)).toEqual(Array.from(s.depth));
  });
});
