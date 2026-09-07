/**
 * ⑧ 遮蔽部のインペイント（docs/03 §3.7）。
 *
 * モデルの推論はここでは動かせないので、確かめるのは
 * 「どこをマスクするか」と「返ってきた絵をどう混ぜるか」。
 * マスクの位置を間違えると、写っている部分を塗り替えてしまう。
 */
import { describe, expect, it } from 'vitest';
import {
  buildInpaintMask,
  compositeInpaint,
  DEFAULT_MASK_PARAMS,
  exposedBandPx,
  stretchFallback,
} from '../../src/pipeline/8-inpaint';

const S = 64;

/** 左半分が手前、右半分が奥。境界は x = 32。 */
function step(): { depth: Float32Array; alpha: Uint8ClampedArray } {
  const depth = new Float32Array(S * S);
  const alpha = new Uint8ClampedArray(S * S);
  for (let y = 8; y < S - 8; y++) {
    for (let x = 8; x < S - 8; x++) {
      const i = y * S + x;
      alpha[i] = 255;
      depth[i] = x < S / 2 ? 0.3 : 0.7;
    }
  }
  return { depth, alpha };
}

describe('露出する帯の幅', () => {
  it('段差が大きいほど広い', () => {
    const w1 = exposedBandPx(500, 1.0, 1.1, Math.PI / 4);
    const w2 = exposedBandPx(500, 1.0, 1.5, Math.PI / 4);
    expect(w2).toBeGreaterThan(w1);
  });

  it('振り角が大きいほど広い', () => {
    const small = exposedBandPx(500, 1.0, 1.3, Math.PI / 12);
    const large = exposedBandPx(500, 1.0, 1.3, Math.PI / 4);
    expect(large).toBeGreaterThan(small);
  });

  it('段差が無ければ 0', () => {
    expect(exposedBandPx(500, 1.0, 1.0, Math.PI / 4)).toBe(0);
    expect(exposedBandPx(500, 1.0, 0.5, Math.PI / 4)).toBe(0);
  });

  it('式のとおりの値を返す', () => {
    // w = f·b·Δz/(z_near·z_far), b = z_near·tan(θ)
    const f = 500, zn = 1.0, zf = 1.4, yaw = Math.PI / 4;
    const expected = (f * zn * Math.tan(yaw) * (zf - zn)) / (zn * zf);
    expect(exposedBandPx(f, zn, zf, yaw)).toBeCloseTo(expected, 6);
  });
});

describe('マスクの位置', () => {
  it('段差の奥側だけをマスクする', () => {
    const { depth, alpha } = step();
    const { mask, maskedPixels } = buildInpaintMask(depth, alpha, S, S, 100, 1.0, 1.5);
    expect(maskedPixels).toBeGreaterThan(0);

    const row = S / 2;
    // 手前側（左）は写っているので触らない
    for (let x = 8; x < S / 2; x++) {
      expect(mask[row * S + x], `x=${x} は手前側なのにマスクされています`).toBe(0);
    }
    // 奥側（右）の境界すぐは隠れていた領域なのでマスクされる
    expect(mask[row * S + S / 2]).toBe(255);
  });

  it('帯の幅が上限を超えない', () => {
    const { depth, alpha } = step();
    const params = { ...DEFAULT_MASK_PARAMS, maxBand: 6 };
    const { mask } = buildInpaintMask(depth, alpha, S, S, 4000, 1.0, 1.5, params);
    const row = S / 2;
    let width = 0;
    for (let x = S / 2; x < S - 8; x++) if (mask[row * S + x] === 255) width++;
    expect(width).toBeLessThanOrEqual(6);
  });

  it('段差が無ければ何もマスクしない', () => {
    const depth = new Float32Array(S * S).fill(0.5);
    const alpha = new Uint8ClampedArray(S * S).fill(255);
    const { maskedPixels } = buildInpaintMask(depth, alpha, S, S, 100, 1.0, 1.5);
    expect(maskedPixels).toBe(0);
  });

  it('被写体の外は段差とみなさない', () => {
    // 背景（α=0）の深度は推定されていない。そこを段差にすると
    // シルエット全周がマスクされてしまう。
    const depth = new Float32Array(S * S);
    const alpha = new Uint8ClampedArray(S * S);
    for (let y = 20; y < 44; y++) {
      for (let x = 20; x < 44; x++) {
        alpha[y * S + x] = 255;
        depth[y * S + x] = 0.4;
      }
    }
    for (let i = 0; i < depth.length; i++) if (alpha[i] === 0) depth[i] = 0.95;
    const { maskedPixels } = buildInpaintMask(depth, alpha, S, S, 100, 1.0, 1.5);
    expect(maskedPixels).toBe(0);
  });
});

describe('取り込み', () => {
  it('マスクの中は補完結果、外は元の絵', () => {
    const n = S * S;
    const base = new Uint8ClampedArray(n * 4).fill(0);
    const painted = new Uint8ClampedArray(n * 4).fill(255);
    const mask = new Uint8ClampedArray(n);
    for (let y = 20; y < 44; y++) for (let x = 20; x < 44; x++) mask[y * S + x] = 255;

    const out = compositeInpaint(base, painted, mask, S, S, 2);
    // 中心はほぼ補完結果
    expect(out[(32 * S + 32) * 4]).toBeGreaterThan(240);
    // 十分外は元のまま
    expect(out[(2 * S + 2) * 4]).toBe(0);
  });

  it('境界が段差にならない（羽根が効いている）', () => {
    const n = S * S;
    const base = new Uint8ClampedArray(n * 4).fill(0);
    const painted = new Uint8ClampedArray(n * 4).fill(255);
    const mask = new Uint8ClampedArray(n);
    for (let y = 0; y < S; y++) for (let x = 32; x < S; x++) mask[y * S + x] = 255;

    const out = compositeInpaint(base, painted, mask, S, S, 3);
    const row = S / 2;
    // 境界付近に中間値が現れる
    const values = [];
    for (let x = 28; x < 37; x++) values.push(out[(row * S + x) * 4] as number);
    const mid = values.filter((v) => v > 20 && v < 235);
    expect(mid.length, `境界の値: ${values.join(',')}`).toBeGreaterThan(0);
  });
});

describe('縮退（引き伸ばし）', () => {
  it('マスクされた画素を隣の色で埋める', () => {
    const n = S * S;
    const color = new Uint8ClampedArray(n * 4);
    for (let i = 0; i < n; i++) {
      color[i * 4] = 100;
      color[i * 4 + 1] = 150;
      color[i * 4 + 2] = 200;
      color[i * 4 + 3] = 255;
    }
    const mask = new Uint8ClampedArray(n);
    const row = 10;
    for (let x = 20; x < 30; x++) {
      mask[row * S + x] = 255;
      color[(row * S + x) * 4] = 0; // 埋められるべき値
    }
    const out = stretchFallback(color, mask, S, S);
    expect(out[(row * S + 25) * 4]).toBe(100); // 左隣の色が伸びた
  });
});
