/**
 * 属性パッキングの往復誤差を、設計（docs/04 §4.4）が主張する精度に対して検証する。
 *
 * ここで測っているのは「設計書が書いた数値が本当に出るか」であって、
 * 実装が自分自身と一致するかではない。閾値は設計書の表から取っている。
 */
import { describe, expect, it } from 'vitest';
import {
  DEFAULT_SCALE_RANGE,
  logQuantRelativeError,
  scaleRangeFor,
  decodeLogScale,
  decodeOct,
  decodeOpacity,
  encodeLogScale,
  encodeOct,
  encodeOpacity,
  fromHalf,
  packHalf2,
  packRgba8,
  rgbToYCoCg,
  toHalf,
  unpackHalf2,
  unpackRgba8,
  yCoCgToRgb,
} from '../../src/codec/pack';

/** 単位球面上に決定的に点を撒く（フィボナッチ格子）。 */
function spherePoints(n: number): [number, number, number][] {
  const out: [number, number, number][] = [];
  const ga = Math.PI * (1 + Math.sqrt(5));
  for (let i = 0; i < n; i++) {
    const t = (i + 0.5) / n;
    const phi = Math.acos(1 - 2 * t);
    const theta = ga * i;
    out.push([Math.sin(phi) * Math.cos(theta), Math.sin(phi) * Math.sin(theta), Math.cos(phi)]);
  }
  return out;
}

describe('八面体写像（法線）', () => {
  it('往復の角度誤差が 0.9° 未満に収まる（設計 docs/04 §4.4）', () => {
    let worst = 0;
    for (const [x, y, z] of spherePoints(20_000)) {
      const [dx, dy, dz] = decodeOct(encodeOct(x, y, z));
      const cos = Math.min(1, Math.max(-1, x * dx + y * dy + z * dz));
      worst = Math.max(worst, (Math.acos(cos) * 180) / Math.PI);
    }
    // 16bit×2 なので設計の 8bit×2 前提（0.9°）より遥かに良いはず
    expect(worst).toBeLessThan(0.9);
    expect(worst).toBeLessThan(0.01);
  });

  it('軸に揃った法線を正確に往復する', () => {
    for (const v of [[1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, 0, 0], [0, -1, 0], [0, 0, -1]] as const) {
      const [dx, dy, dz] = decodeOct(encodeOct(v[0], v[1], v[2]));
      expect(dx).toBeCloseTo(v[0], 4);
      expect(dy).toBeCloseTo(v[1], 4);
      expect(dz).toBeCloseTo(v[2], 4);
    }
  });

  it('z が負の半球（背面シェルの法線）も往復する', () => {
    const n = spherePoints(5000).filter(([, , z]) => z < -0.1);
    expect(n.length).toBeGreaterThan(1000);
    for (const [x, y, z] of n) {
      const [dx, dy, dz] = decodeOct(encodeOct(x, y, z));
      expect(dz).toBeLessThan(0); // 半球を取り違えていない
      expect(Math.hypot(dx - x, dy - y, dz - z)).toBeLessThan(1e-3);
    }
  });
});

describe('対数スケール量子化', () => {
  /** 指定レンジ内を掃いて往復の最大相対誤差を測る。 */
  function worstRelError(range: { min: number; max: number }, bits: number): number {
    let worst = 0;
    for (let i = 0; i < 4000; i++) {
      const t = i / 3999;
      const v = Math.exp(Math.log(range.min) + t * (Math.log(range.max) - Math.log(range.min)));
      const back = decodeLogScale(encodeLogScale(v, bits, range), bits, range);
      worst = Math.max(worst, Math.abs(back - v) / v);
    }
    return worst;
  }

  it('相対誤差が閉形式 exp(ln(R)/(2(2^b−1)))−1 に一致する', () => {
    // 設計書の数値がどの条件で成り立つかを、式そのもので固定しておく
    for (const ratio of [16, 51, 120]) {
      for (const bits of [6, 8]) {
        const range = { min: 1e-3, max: 1e-3 * ratio };
        const predicted = logQuantRelativeError(range, bits);
        expect(worstRelError(range, bits)).toBeLessThanOrEqual(predicted * 1.02);
        expect(worstRelError(range, bits)).toBeGreaterThan(predicted * 0.5);
      }
    }
  });

  it('設計書の 8bit=0.55% / 6bit=2.2% は、レンジ比 16 のときの値である', () => {
    const r16 = { min: 1e-3, max: 16e-3 };
    expect(worstRelError(r16, 8)).toBeLessThan(0.0055);
    expect(worstRelError(r16, 6)).toBeLessThan(0.0223);
  });

  it('実際に出うるレンジ（比 ≈ 51）でも 8bit で 0.8% 未満に収まる', () => {
    // 1024²・画角55° の下限 9.7e-4 〜 上限 5.0e-2。サーフェル半径2pxに対し 0.016px。
    const real = { min: 9.7e-4, max: 5.0e-2 };
    expect(real.max / real.min).toBeGreaterThan(45);
    expect(worstRelError(real, 8)).toBeLessThan(0.008);
  });

  it('既定レンジが実際に出うる範囲を覆っている', () => {
    expect(DEFAULT_SCALE_RANGE.min).toBeLessThanOrEqual(9.7e-4);
    expect(DEFAULT_SCALE_RANGE.max).toBeGreaterThanOrEqual(5.0e-2);
  });

  it('相対誤差が大きさによらず一定（対数量子化の要点）', () => {
    const rel = (v: number) => Math.abs(decodeLogScale(encodeLogScale(v, 8), 8) - v) / v;
    const small = rel(9e-4);
    const large = rel(5e-2);
    // 対数量子化なら比が 1 に近い。線形量子化ならここが50倍以上ずれる。
    expect(Math.max(small, large) / Math.max(Math.min(small, large), 1e-12)).toBeLessThan(6);
  });

  describe('scaleRangeFor（書き出しごとにレンジを締める）', () => {
    it('外れ値に引きずられない', () => {
      const data = [...Array(1000)].map((_, i) => 1e-3 * (1 + i / 999));
      data.push(1.0); // マットの縁に残る極端なサーフェル
      const r = scaleRangeFor(data);
      expect(r.max).toBeLessThan(0.01);
      expect(logQuantRelativeError(r, 8)).toBeLessThan(0.003);
    });

    it('レンジを締めると同じビット数で精度が上がる', () => {
      const data = [...Array(2000)].map((_, i) => 1.2e-3 * Math.pow(8, i / 1999));
      const tight = scaleRangeFor(data);
      expect(logQuantRelativeError(tight, 8)).toBeLessThan(
        logQuantRelativeError(DEFAULT_SCALE_RANGE, 8),
      );
    });

    it('空配列や不正値では既定レンジに落ちる', () => {
      expect(scaleRangeFor([])).toEqual(DEFAULT_SCALE_RANGE);
      expect(scaleRangeFor([NaN, -1, 0])).toEqual(DEFAULT_SCALE_RANGE);
    });
  });
});

describe('不透明度（logit 空間）', () => {
  /** 透過率 (1−α) の相対誤差。α ブレンドで累積するのはこちら。 */
  const transmittanceError = (a: number, roundTrip: (x: number) => number) =>
    Math.abs(1 - roundTrip(a) - (1 - a)) / (1 - a);

  const viaLogit = (a: number) => decodeOpacity(encodeOpacity(a));
  const viaLinear = (a: number) => Math.round(a * 255) / 255;

  it('α→1 で透過率の相対誤差が線形量子化より桁違いに小さい', () => {
    // これが logit を選ぶ理由。線形は 1/255 刻みが (1−α) に対して巨大になる。
    for (const a of [0.95, 0.99, 0.997]) {
      const logit = transmittanceError(a, viaLogit);
      const linear = transmittanceError(a, viaLinear);
      expect(logit).toBeLessThan(linear);
    }
    // α=0.99 では 25 倍以上の差がつく
    expect(transmittanceError(0.99, viaLinear) / transmittanceError(0.99, viaLogit)).toBeGreaterThan(20);
  });

  it('透過率が意味を持つ範囲（α ≤ 0.995）で相対誤差が 2.5% 以内', () => {
    for (let i = 1; i <= 995; i++) {
      const a = i / 1000;
      expect(transmittanceError(a, viaLogit)).toBeLessThan(0.025);
    }
  });

  it('logit のクリップ点より先では、透過率の絶対値が見えない水準に収まる', () => {
    // LOGIT_LIMIT = 6 なので α は 0.9975 で頭打ちになる。
    // そこより不透明な入力は透過率の「相対」誤差が跳ねる（α=0.999 で約 150%）が、
    // 透過率の絶対値が 0.0025 以下なので背景の漏れは知覚できない。
    for (const a of [0.998, 0.999, 0.9999]) {
      const t = 1 - viaLogit(a);
      expect(t).toBeLessThan(0.003);
      expect(transmittanceError(a, viaLogit)).toBeGreaterThan(0.1); // 跳ねること自体を固定
    }
  });

  it('α の絶対誤差は最大 0.006（α=0.5 付近）— 線形より粗いが透過率が十分残る領域', () => {
    let worst = 0;
    for (let i = 1; i < 1000; i++) {
      const a = i / 1000;
      worst = Math.max(worst, Math.abs(viaLogit(a) - a));
    }
    expect(worst).toBeLessThan(0.006);
    expect(worst).toBeGreaterThan(1 / 255); // 線形より粗いことを明示的に固定
  });

  it('単調性が保たれる（順序が入れ替わらない）', () => {
    let prev = -1;
    for (let i = 1; i < 1000; i++) {
      const v = viaLogit(i / 1000);
      expect(v).toBeGreaterThanOrEqual(prev);
      prev = v;
    }
  });
});

describe('YCoCg 色変換', () => {
  it('整数 RGB を無損失で往復する（加減算とシフトのみ）', () => {
    for (let i = 0; i < 4096; i++) {
      const r = (i * 61) % 256;
      const g = (i * 137) % 256;
      const b = (i * 251) % 256;
      const { y, co, cg } = rgbToYCoCg(r, g, b);
      expect(yCoCgToRgb(y, co, cg)).toEqual([r, g, b]);
    }
  });
});

describe('半精度浮動小数（WGSL の unpack2x16float と対）', () => {
  it('スケールの実用域で相対誤差が 0.1% 未満', () => {
    for (let i = 0; i < 1000; i++) {
      const v = 1e-4 * Math.pow(2000, i / 999); // 1e-4 .. 0.2
      const back = fromHalf(toHalf(v));
      expect(Math.abs(back - v) / v).toBeLessThan(0.001);
    }
  });

  it('2つ詰めて取り出せる', () => {
    const [a, b] = unpackHalf2(packHalf2(0.0125, 0.0031));
    expect(a).toBeCloseTo(0.0125, 5);
    expect(b).toBeCloseTo(0.0031, 5);
  });

  it('0 と特殊値を壊さない', () => {
    expect(fromHalf(toHalf(0))).toBe(0);
    expect(fromHalf(toHalf(1))).toBe(1);
    expect(fromHalf(toHalf(-1))).toBe(-1);
  });
});

describe('RGBA8 パッキング（WGSL の unpack4x8unorm と対）', () => {
  it('往復する', () => {
    for (const v of [[0, 0, 0, 0], [255, 255, 255, 255], [12, 200, 7, 220], [1, 2, 3, 4]] as const) {
      expect(unpackRgba8(packRgba8(v[0], v[1], v[2], v[3]))).toEqual([...v]);
    }
  });

  it('リトルエンディアンで R が最下位バイトにある', () => {
    expect(packRgba8(0xab, 0, 0, 0)).toBe(0xab);
    expect(packRgba8(0, 0, 0, 0xcd)).toBe(0xcd000000);
  });
});
