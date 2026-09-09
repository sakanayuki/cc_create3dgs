/**
 * 書き出しを実寸へ直す倍率（docs/09 §V25）。
 *
 * 深度モデルの絶対値は当てにならないので、人物モードでは身長を仮定して
 * 倍率を出す。これは**推定であって実測ではない**。
 */
import { describe, expect, it } from 'vitest';
import { ASSUMED_STATURE_M, statureFix } from '../../src/pipeline/generate';

describe('身長の仮定から実寸の倍率を出す', () => {
  it('人物モードでは、縦の広がりが仮定の身長になる倍率を返す', () => {
    // 実写で出てきた値。立ち姿が 0.427m と出る。
    const fix = statureFix('person', 0.427);
    expect(fix).toBeCloseTo(ASSUMED_STATURE_M / 0.427, 6);
    expect(0.427 * fix, '直した身長が仮定と合いません').toBeCloseTo(ASSUMED_STATURE_M, 6);
  });

  it('もともと妥当な大きさなら、ほとんど動かさない', () => {
    expect(statureFix('person', 1.65)).toBeCloseTo(1, 6);
  });

  it('物体モードは触らない（仮定できるものが無い）', () => {
    expect(statureFix('object', 0.427)).toBe(1);
  });

  it('桁が違うときは仮定を重ねない', () => {
    // 較正が壊れて 1mm や 1km になったときは、そのまま出す。
    expect(statureFix('person', 1e-4)).toBe(1);
    expect(statureFix('person', 1e4)).toBe(1);
  });

  it('高さが 0 や不正なら触らない', () => {
    expect(statureFix('person', 0)).toBe(1);
    expect(statureFix('person', Number.NaN)).toBe(1);
  });
});
