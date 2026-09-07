/**
 * 深度の2プレーン分離（docs/04 §4.5.4）とプレーンの詰め替え。
 *
 * 「16bit をそのまま持ってはいけない」という設計判断が、実際に量子化として
 * 正しく効いているか（＝誤差が量子化幅に収まり、系統的な偏りが無いか）を測る。
 */
import { describe, expect, it } from 'vitest';
import {
  grayToRgba,
  mergeDepth,
  rgbToRgba,
  rgbaToGray,
  rgbaToRgb,
  splitDepth,
} from '../../src/codec/images';

/** 滑らかな深度マップ（実際の深度に近い性質）。 */
function smoothDepth(n: number): Uint16Array {
  const out = new Uint16Array(n);
  for (let i = 0; i < n; i++) {
    const t = i / (n - 1);
    out[i] = Math.round((0.3 + 0.4 * t + 0.05 * Math.sin(t * 20)) * 65535);
  }
  return out;
}

describe('深度の2プレーン分離', () => {
  it('12bit 量子化の誤差が量子化幅の半分に収まる', () => {
    const src = smoothDepth(4096);
    const back = mergeDepth(splitDepth(src, 12), 12);
    const step = 1 << (16 - 12); // 16
    let worst = 0;
    for (let i = 0; i < src.length; i++) {
      worst = Math.max(worst, Math.abs((back[i] as number) - (src[i] as number)));
    }
    expect(worst).toBeLessThanOrEqual(step / 2);
  });

  it('丸め誤差に系統的な偏りが無い（中央へ戻している）', () => {
    const src = smoothDepth(8192);
    const back = mergeDepth(splitDepth(src, 12), 12);
    let sum = 0;
    for (let i = 0; i < src.length; i++) sum += (back[i] as number) - (src[i] as number);
    const bias = sum / src.length;
    // 切り捨てのままだと平均 -8 程度ずれ、被写体が系統的に手前へ寄る
    expect(Math.abs(bias)).toBeLessThan(1.0);
  });

  it('10bit（軽量）でも誤差が量子化幅の半分に収まる', () => {
    const src = smoothDepth(4096);
    const back = mergeDepth(splitDepth(src, 10), 10);
    const step = 1 << (16 - 10); // 64
    for (let i = 0; i < src.length; i++) {
      expect(Math.abs((back[i] as number) - (src[i] as number))).toBeLessThanOrEqual(step / 2);
    }
  });

  it('上位プレーンが滑らかで、下位プレーンだけが暴れる', () => {
    // これが2プレーンに分ける理由。1枚だとこの2つが交互に並んで予測が当たらない。
    const src = smoothDepth(2048);
    const { high, low } = splitDepth(src, 12);
    const variation = (a: ArrayLike<number>) => {
      let sum = 0;
      for (let i = 1; i < a.length; i++) sum += Math.abs((a[i] as number) - (a[i - 1] as number));
      return sum / (a.length - 1);
    };
    // 上位は隣接差分がほぼゼロ、下位は数十のオーダー
    expect(variation(high)).toBeLessThan(1);
    expect(variation(low)).toBeGreaterThan(variation(high) * 10);
  });

  it('下位ビットを上位ニブルに寄せている（8bit プレーンとして扱うため）', () => {
    const src = Uint16Array.from([0x0fff, 0x1000, 0xffff]);
    const { low } = splitDepth(src, 12);
    // 12bit なので下位は 4bit。<< 4 されて上位ニブルに入る
    for (const v of low) expect(v & 0x0f).toBe(0);
  });

  it('範囲外のビット数を拒否する', () => {
    expect(() => splitDepth(new Uint16Array(4), 7)).toThrow(/範囲外/);
    expect(() => splitDepth(new Uint16Array(4), 17)).toThrow(/範囲外/);
  });

  it('端の値を飽和させない', () => {
    const src = Uint16Array.from([0, 65535]);
    const back = mergeDepth(splitDepth(src, 12), 12);
    expect(back[0]).toBeLessThan(16);
    expect(back[1]).toBeGreaterThan(65535 - 16);
  });
});

describe('プレーンの詰め替え', () => {
  it('1ch → RGBA → 1ch を往復する', () => {
    const gray = Uint8ClampedArray.from([0, 40, 128, 255, 7, 200]);
    const rgba = grayToRgba(gray, 3, 2);
    expect(rgba.length).toBe(24);
    expect([...rgbaToGray(rgba, 3, 2)]).toEqual([...gray]);
  });

  it('1ch を広げると RGB が同値、A は 255 になる（コーデックの予測が効くように）', () => {
    const rgba = grayToRgba(Uint8ClampedArray.from([77]), 1, 1);
    expect([...rgba]).toEqual([77, 77, 77, 255]);
  });

  it('3ch → RGBA → 3ch を往復する', () => {
    const rgb = Uint8ClampedArray.from([1, 2, 3, 250, 251, 252]);
    const rgba = rgbToRgba(rgb, 2, 1);
    expect([...rgba]).toEqual([1, 2, 3, 255, 250, 251, 252, 255]);
    expect([...rgbaToRgb(rgba, 2, 1)]).toEqual([...rgb]);
  });

  it('長さが合わなければ拒否する', () => {
    expect(() => grayToRgba(new Uint8ClampedArray(5), 3, 2)).toThrow(/長さが合いません/);
    expect(() => rgbToRgba(new Uint8ClampedArray(5), 2, 1)).toThrow(/長さが合いません/);
  });
});
