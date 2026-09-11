/**
 * 合成の手前のゲート（docs/12 §12.14 R17 / R21）。
 *
 * **設計にはゲートにすると書いてあったのに、実装で抜けていた。**
 * PR #4 のレビューで指摘されるまで気づかなかった。同じ人物でない3枚や、
 * カメラが動いた3枚を入れられると位置合わせは解けず、そのまま重ねると
 * 壊れた立体が画面と書き出しにそのまま出る。しかも利用者には理由が分からない。
 * 二度と落とさないよう、判定を純粋な関数に切り出して固定する。
 */
import { describe, expect, it } from 'vitest';
import { fallbackMessage, registrationGate } from '../../src/pipeline/generateMulti';
import { REFERENCE_POSE, IDENTITY_FRAME, type ViewSlot } from '../../src/pipeline/align/rigid';
import type { RegisterResult } from '../../src/pipeline/align/registerViews';

function view(slot: ViewSlot, insideRatio: number): RegisterResult['views'][number] {
  return {
    slot,
    pose: REFERENCE_POSE,
    containmentPx: 0,
    insideRatio,
    iou: 0.7,
    slotFlipped: false,
    headYawDisagrees: false,
    frame: IDENTITY_FRAME,
  };
}

function result(ratios: readonly number[]): RegisterResult {
  const slots: ViewSlot[] = ['front', 'right', 'left'];
  const views = ratios.map((r, i) => view(slots[i] as ViewSlot, r));
  return {
    views,
    referenceIndex: 0,
    cost: 0.01,
    worstInsideRatio: Math.min(...ratios),
    anySlotFlipped: false,
  };
}

describe('合成の手前のゲート', () => {
  it('よく合っていれば通す', () => {
    const g = registrationGate(result([0.985, 0.987, 0.991]));
    expect(g.ok).toBe(true);
  });

  it('ちょうど合格ライン（0.95）は通す', () => {
    expect(registrationGate(result([0.95, 0.98, 0.99])).ok).toBe(true);
  });

  it('1つでも割っていたら止める', () => {
    const g = registrationGate(result([0.985, 0.62, 0.991]));
    expect(g.ok).toBe(false);
    // いちばん合っていなかった view を指せること。理由の文面に使う。
    expect(g.worst.slot).toBe('right');
    expect(g.worst.insideRatio).toBeCloseTo(0.62, 5);
  });

  it('2枚でも効く', () => {
    expect(registrationGate(result([0.99, 0.80])).ok).toBe(false);
  });

  it('理由の文面に、どの写真がどれだけ合わなかったかが入る', () => {
    const g = registrationGate(result([0.985, 0.62, 0.991]));
    const msg = fallbackMessage(g.worst, 'front');
    expect(msg).toContain('右向き');
    expect(msg).toContain('62%');
    expect(msg).toContain('正面'); // 何で作ったかを伝える
  });
});
