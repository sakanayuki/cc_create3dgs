/**
 * ビューアの可動範囲（docs/06 §6.7）。
 *
 * 半球カバー（決定 D2）なので、視点を自由に回させてはいけない。
 * 裏側の粗が見えると「壊れている」と受け取られる。
 */
import { describe, expect, it } from 'vitest';
import { LIMITS, rubberBand } from '../../src/ui/viewer';

describe('ラバーバンド', () => {
  const { yawComfort: c, yawHard: h } = LIMITS;

  it('快適範囲の中は素通し', () => {
    for (const v of [0, c * 0.3, -c * 0.7, c]) {
      expect(rubberBand(v, c, h)).toBeCloseTo(v, 6);
    }
  });

  it('どれだけ押しても上限を超えない', () => {
    // tanh は 1 に漸近するので、極端に押し込むと浮動小数では上限ちょうどに
    // 飽和する。超えないことが要件なので、そこは等号を許す。
    for (const v of [c * 1.5, c * 5, c * 100, -c * 100]) {
      expect(Math.abs(rubberBand(v, c, h))).toBeLessThanOrEqual(h);
    }
    // 現実的な行き過ぎ（快適範囲の2倍）では、まだ上限に届いていない
    expect(Math.abs(rubberBand(c * 2, c, h))).toBeLessThan(h);
  });

  it('快適範囲の境界で滑らかに繋がる', () => {
    // 境界の内側と外側で傾きが揃っていること（硬い壁だと段差になる）
    const eps = 1e-4;
    const inner = (rubberBand(c, c, h) - rubberBand(c - eps, c, h)) / eps;
    const outer = (rubberBand(c + eps, c, h) - rubberBand(c, c, h)) / eps;
    expect(inner).toBeCloseTo(1, 3);
    expect(outer).toBeCloseTo(1, 3);
  });

  it('押し込むほど抵抗が増す', () => {
    const d1 = rubberBand(c * 1.2, c, h) - rubberBand(c * 1.1, c, h);
    const d2 = rubberBand(c * 2.2, c, h) - rubberBand(c * 2.1, c, h);
    expect(d2).toBeLessThan(d1);
    expect(d2).toBeGreaterThan(0);
  });

  it('符号を保つ', () => {
    expect(rubberBand(-c * 3, c, h)).toBeLessThan(0);
    expect(rubberBand(c * 3, c, h)).toBeGreaterThan(0);
  });
});
