/**
 * 色の鮮鋭化（docs/13 §13.2 T1）。
 *
 * これは写真に無いものを足す処理なので、既定では掛からない。掛けたときに
 * 「被写体の外を混ぜない」ことと「局所コントラストが上がる」ことを見る。
 */
import { describe, expect, it } from 'vitest';
import { unsharpMaskRgba } from '../../src/pipeline/imageOps';

const W = 64;
const H = 64;

/** 左半分が暗く右半分が明るい被写体。外は真っ白（背景）。 */
function scene(): { rgba: Uint8ClampedArray; alpha: Uint8ClampedArray } {
  const rgba = new Uint8ClampedArray(W * H * 4);
  const alpha = new Uint8ClampedArray(W * H);
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const i = y * W + x;
      const inside = x >= 16 && x < 48;
      const v = inside ? (x < 32 ? 80 : 160) : 255;
      rgba[i * 4] = v;
      rgba[i * 4 + 1] = v;
      rgba[i * 4 + 2] = v;
      rgba[i * 4 + 3] = inside ? 255 : 0;
      alpha[i] = inside ? 255 : 0;
    }
  }
  return { rgba, alpha };
}

/** 横一線の、隣どうしの差の合計。局所コントラストの代わり。 */
function contrast(rgba: ArrayLike<number>, y: number, x0: number, x1: number): number {
  let sum = 0;
  for (let x = x0; x < x1 - 1; x++) {
    sum += Math.abs((rgba[(y * W + x + 1) * 4] as number) - (rgba[(y * W + x) * 4] as number));
  }
  return sum;
}

describe('色の鮮鋭化', () => {
  it('強さ 0 なら 1 画素も動かない', () => {
    const { rgba, alpha } = scene();
    const out = unsharpMaskRgba(rgba, alpha, W, H, 2, 0);
    for (let i = 0; i < rgba.length; i += 7) expect(out[i]).toBe(rgba[i]);
  });

  it('半径 0 でも何もしない', () => {
    const { rgba, alpha } = scene();
    const out = unsharpMaskRgba(rgba, alpha, W, H, 0, 0.5);
    for (let i = 0; i < rgba.length; i += 7) expect(out[i]).toBe(rgba[i]);
  });

  it('段差が強調される', () => {
    const { rgba, alpha } = scene();
    const before = contrast(rgba, 32, 20, 44);
    const out = unsharpMaskRgba(rgba, alpha, W, H, 2, 0.5);
    const after = contrast(out, 32, 20, 44);
    expect(after, `段差が ${before} → ${after} としか変わっていません`).toBeGreaterThan(
      before * 1.2,
    );
  });

  it('被写体の外は動かさない', () => {
    const { rgba, alpha } = scene();
    const out = unsharpMaskRgba(rgba, alpha, W, H, 2, 0.5);
    for (let y = 0; y < H; y += 3) {
      for (let x = 0; x < W; x += 3) {
        if ((alpha[y * W + x] as number) >= 128) continue;
        expect(out[(y * W + x) * 4], `外の (${x},${y}) が動いています`).toBe(
          rgba[(y * W + x) * 4],
        );
      }
    }
  });

  it('シルエットの内側に背景が染み出さない（ハロが出ない）', () => {
    // 背景は 255。被写体の左端（値 80）が背景に引っ張られると、
    // 低域が明るくなって内側が暗く沈む。α で重みをつけていれば起きない。
    const { rgba, alpha } = scene();
    const out = unsharpMaskRgba(rgba, alpha, W, H, 3, 1.0);
    // 左端の数画素は、平坦な部分なので動かないはず
    for (let x = 16; x < 24; x++) {
      const v = out[(32 * W + x) * 4] as number;
      expect(Math.abs(v - 80), `x=${x} が ${v}（元 80）まで動いています`).toBeLessThan(6);
    }
  });

  it('α は触らない', () => {
    const { rgba, alpha } = scene();
    const out = unsharpMaskRgba(rgba, alpha, W, H, 2, 0.5);
    for (let i = 0; i < W * H; i += 5) expect(out[i * 4 + 3]).toBe(rgba[i * 4 + 3]);
  });
});
