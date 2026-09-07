/** `.spz` 書き出しの構造（docs/05 §5.3）。 */
import { describe, expect, it } from 'vitest';
import {
  SPZ_FRACTIONAL_BITS,
  SPZ_MAGIC,
  SPZ_VERSION,
  SpzPointOut,
  decodeSpzHeader,
  encodeSpzRaw,
  linearToSpzColor,
  shCoeffCount,
  spzColorToLinear,
  spzRawByteLength,
} from '../../src/codec/spz';

const header = (n: number) => ({
  numPoints: n,
  shDegree: 0,
  fractionalBits: SPZ_FRACTIONAL_BITS,
  antialiased: true,
});

describe('SPZ ヘッダ', () => {
  it('マジック・バージョン・点数を書き、読み戻せる', () => {
    const bytes = encodeSpzRaw(header(7), () => {});
    const dv = new DataView(bytes.buffer);
    expect(dv.getUint32(0, true)).toBe(SPZ_MAGIC);
    expect(dv.getUint32(4, true)).toBe(SPZ_VERSION);
    expect(decodeSpzHeader(bytes)).toEqual(header(7));
  });

  it('SH 次数0 で書く（単一画像に視点依存の情報が無いため）', () => {
    const bytes = encodeSpzRaw(header(3), () => {});
    expect(decodeSpzHeader(bytes).shDegree).toBe(0);
    expect(shCoeffCount(0)).toBe(0);
  });

  it('壊れた入力を拒否する', () => {
    expect(() => decodeSpzHeader(new Uint8Array(4))).toThrow(/短すぎ/);
    const bad = encodeSpzRaw(header(1), () => {});
    new DataView(bad.buffer).setUint32(0, 0xdeadbeef, true);
    expect(() => decodeSpzHeader(bad)).toThrow(/マジック/);
  });

  it('未対応バージョンを拒否する', () => {
    const bytes = encodeSpzRaw(header(1), () => {});
    new DataView(bytes.buffer).setUint32(4, 99, true);
    expect(() => decodeSpzHeader(bytes)).toThrow(/未対応の SPZ バージョン/);
  });
});

describe('SPZ 本体', () => {
  it('SoA の並びで、宣言どおりのバイト数になる', () => {
    for (const n of [0, 1, 1000, 424_000]) {
      const bytes = encodeSpzRaw(header(n), () => {});
      expect(bytes.length).toBe(spzRawByteLength(n));
      // 16 ヘッダ + 19 バイト/点（位置9 + α1 + 色3 + スケール3 + 回転3）
      expect(bytes.length).toBe(16 + n * 19);
    }
  });

  it('位置を 24bit 固定小数点で書く', () => {
    const bytes = encodeSpzRaw(header(1), (_i, out) => {
      out.x = 0.25;
      out.y = -0.5;
      out.z = 1.0;
    });
    const scale = 1 << SPZ_FRACTIONAL_BITS;
    const read24 = (off: number) => {
      const v = (bytes[off] as number) | ((bytes[off + 1] as number) << 8) | ((bytes[off + 2] as number) << 16);
      return (v << 8) >> 8; // 24bit 符号拡張
    };
    expect(read24(16) / scale).toBeCloseTo(0.25, 4);
    expect(read24(19) / scale).toBeCloseTo(-0.5, 4);
    expect(read24(22) / scale).toBeCloseTo(1.0, 4);
  });

  it('位置の量子化幅が 1/4096 で、被写体1mに対し 0.24mm', () => {
    const step = 1 / (1 << SPZ_FRACTIONAL_BITS);
    expect(step).toBeCloseTo(0.000244, 6);
    // 1024² で 1px ≈ 1.0mm なので、その 1/4 より細かい
    expect(step).toBeLessThan(0.001 / 2);
  });

  it('属性を SoA でまとめる（gzip が効くように）', () => {
    const n = 4;
    const bytes = encodeSpzRaw(header(n), (i, out) => {
      out.alpha = 10 + i;
      out.r = 100 + i;
    });
    // α は位置ブロック（n×9）の直後に連続して並ぶ
    const alphaBase = 16 + n * 9;
    expect([...bytes.subarray(alphaBase, alphaBase + n)]).toEqual([10, 11, 12, 13]);
    // 色はその直後
    const colorBase = alphaBase + n;
    expect(bytes[colorBase]).toBe(100);
    expect(bytes[colorBase + 3]).toBe(101);
  });

  it('対数スケールを (logScale + 10) × 16 で書く', () => {
    const bytes = encodeSpzRaw(header(1), (_i, out) => {
      out.logScale0 = -5;
      out.logScale1 = -8; // サーフェルの厚み
      out.logScale2 = 0;
    });
    const base = 16 + 9 + 1 + 3;
    expect(bytes[base]).toBe((-5 + 10) * 16);
    expect(bytes[base + 1]).toBe((-8 + 10) * 16);
    expect(bytes[base + 2]).toBe(10 * 16);
  });

  it('クォータニオンは w≥0 に揃えて xyz だけ書く', () => {
    const bytes = encodeSpzRaw(header(2), (i, out) => {
      // 2個目は w が負。符号を反転して同じ回転を表すはず。
      const s = i === 0 ? 1 : -1;
      out.qw = 0.7071 * s;
      out.qx = 0.7071 * s;
      out.qy = 0;
      out.qz = 0;
    });
    const base = 16 + 2 * (9 + 1 + 3 + 3);
    // 両方とも同じバイト列になる（同じ回転だから）
    expect(bytes[base]).toBe(bytes[base + 3]);
    expect(bytes[base]).toBeGreaterThan(127); // qx が正側
  });

  it('範囲外の値を飽和させる（NaN や極端な入力で壊れない）', () => {
    const bytes = encodeSpzRaw(header(1), (_i, out) => {
      out.alpha = 9999;
      out.r = -50;
      out.logScale0 = 1000;
    });
    expect(bytes[16 + 9]).toBe(255);
    expect(bytes[16 + 9 + 1]).toBe(0);
    expect(bytes[16 + 9 + 1 + 3]).toBe(255);
  });
});

describe('SPZ の色エンコード', () => {
  it('往復する（誤差 0.005 未満）', () => {
    for (const c of [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1]) {
      expect(spzColorToLinear(linearToSpzColor(c))).toBeCloseTo(c, 2);
    }
  });

  it('中間色が中央値にマップされる', () => {
    expect(linearToSpzColor(0.5)).toBe(128);
  });

  it('黒と白を飽和させない（SH DC を経由する理由）', () => {
    // (c−0.5)/colorScale を直接 u8 に写すと c=0 で 0、c=1 で 255 に張り付き、
    // 暗部と明部の階調が消える。DC 経由なら 60〜195 の範囲に収まる。
    const black = linearToSpzColor(0);
    const white = linearToSpzColor(1);
    expect(black).toBeGreaterThan(0);
    expect(white).toBeLessThan(255);
    expect(black).toBeCloseTo(60, -1);
    expect(white).toBeCloseTo(195, -1);
  });

  it('単調増加する', () => {
    let prev = -1;
    for (let i = 0; i <= 100; i++) {
      const u = linearToSpzColor(i / 100);
      expect(u).toBeGreaterThanOrEqual(prev);
      prev = u;
    }
  });
});

describe('SpzPointOut', () => {
  it('既定値が単位クォータニオンと不透明ゼロ', () => {
    const p = new SpzPointOut();
    expect([p.qw, p.qx, p.qy, p.qz]).toEqual([1, 0, 0, 0]);
    expect(p.alpha).toBe(0);
  });
});
