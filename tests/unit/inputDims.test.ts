/**
 * モデル入力の階数合わせ（scripts/calibrate.py と対になる）。
 *
 * NCHW を決め打ちにすると Depth Anything 3 で落ちる。DA3 は多視点モデルで、
 * 入力が [batch, views, 3, H, W] の5階になる。CI で実際にこう落ちた。
 *
 *   Invalid rank for input: pixel_values Got: 4 Expected: 5
 */
import { describe, expect, it } from 'vitest';
import { inputDims } from '../../src/pipeline/generate';

describe('入力の階数合わせ', () => {
  it('4階のモデルには NCHW を渡す', () => {
    expect(inputDims(4, 3, 512, 512)).toEqual([1, 3, 512, 512]);
  });

  it('5階のモデルには視点軸を挟む（DA3）', () => {
    expect(inputDims(5, 3, 518, 518)).toEqual([1, 1, 3, 518, 518]);
  });

  it('3階のモデルにはバッチ軸を付けない', () => {
    expect(inputDims(3, 3, 256, 256)).toEqual([3, 256, 256]);
  });

  it('階数が読めなければ NCHW を仮定する', () => {
    expect(inputDims(undefined, 3, 512, 512)).toEqual([1, 3, 512, 512]);
  });

  it('要素数はどの階数でも変わらない', () => {
    const total = (dims: number[]): number => dims.reduce((a, b) => a * b, 1);
    for (const rank of [3, 4, 5, 6]) {
      expect(total(inputDims(rank, 3, 64, 64)), `階数 ${rank}`).toBe(3 * 64 * 64);
    }
  });

  it('チャンネル数が 1 でも通る（マスク入力）', () => {
    expect(inputDims(4, 1, 512, 512)).toEqual([1, 1, 512, 512]);
    expect(inputDims(5, 1, 512, 512)).toEqual([1, 1, 1, 512, 512]);
  });
});
