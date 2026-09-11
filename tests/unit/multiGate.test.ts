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
import {
  droppedMessage,
  fallbackMessage,
  registrationGate,
  solveWithDrops,
} from '../../src/pipeline/generateMulti';
import { REFERENCE_POSE, IDENTITY_FRAME, type ViewSlot } from '../../src/pipeline/align/rigid';
import type { AlignView, RegisterResult } from '../../src/pipeline/align/registerViews';
import { DEFAULT_BODY, renderBody } from '../helpers/syntheticBody';

const W = 200;
const H = 320;
const CAM = { focalPx: 800, cx: W / 2, cy: H / 2 };
const RIGHT = -Math.PI / 2;
const LEFT = Math.PI / 2;

/** 標準の体から1枚。`subjectYaw` は被写体の回転（姿勢の yaw とは符号が逆）。 */
function body(slot: ViewSlot, subjectYaw: number): AlignView {
  const r = renderBody(DEFAULT_BODY, CAM, W, H, subjectYaw);
  return { slot, width: W, height: H, camera: CAM, alpha: r.alpha, depth: r.depth };
}

/** 幅と奥行きの比だけを変えた体。比を変えないと剛体で合ってしまう。 */
function reshaped(slot: ViewSlot, subjectYaw: number, kx: number, kz: number): AlignView {
  const shape = {
    ...DEFAULT_BODY,
    parts: DEFAULT_BODY.parts.map((p) => ({ ...p, rx: p.rx * kx, rz: p.rz * kz })),
  };
  const r = renderBody(shape, CAM, W, H, subjectYaw);
  return { slot, width: W, height: H, camera: CAM, alpha: r.alpha, depth: r.depth };
}

/**
 * **別人**。縦の長さは揃えたまま、幅と奥行きの比だけを変える。
 *
 * 尺度は縦の広がりで縛られている（docs/12 §12.7）ので、ただ大きい／小さいだけの
 * 体は素直に合ってしまう。合わない写真を作るには**形の比**を崩す必要がある。
 * ここでは胴を深く細く、脚を太くして、どの角度から見ても元の体と重ならなくする。
 */
function other(slot: ViewSlot, subjectYaw: number): AlignView {
  const shape = {
    ...DEFAULT_BODY,
    parts: DEFAULT_BODY.parts.map((p) => ({
      ...p,
      rx: p.rx * (p.y < 0 ? 0.5 : 2.0),
      rz: p.rz * (p.y < 0 ? 2.2 : 0.5),
    })),
  };
  const r = renderBody(shape, CAM, W, H, subjectYaw);
  return { slot, width: W, height: H, camera: CAM, alpha: r.alpha, depth: r.depth };
}

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

  it('ちょうど合格ライン（3枚なら 0.90）は通す', () => {
    expect(registrationGate(result([0.9, 0.98, 0.99])).ok).toBe(true);
    expect(registrationGate(result([0.9, 0.98, 0.99])).threshold).toBe(0.9);
  });

  /**
   * 2枚のときは線が違う（docs/12 §12.16.5）。
   *
   * 角度を縛るのは3枚目である。2枚だと拘束が足りず、**正しい組でも**
   * M1 が下がる（実素材で 0.920 / 0.928）。3枚と同じ 0.90 を当てると
   * 正しい2枚を落とす。
   */
  it('2枚のときは 0.85（正しい2枚でも M1 が下がるため）', () => {
    expect(registrationGate(result([0.928, 0.974])).threshold).toBe(0.85);
    // 実素材の正しい2枚。0.90 のままなら落ちていた。
    expect(registrationGate(result([0.92, 0.942])).ok).toBe(true);
    expect(0.92).toBeLessThan(0.9 + 0.021); // 0.90 との余裕は 0.02 しかない
    // 大きく取り違えた2枚は落とせる
    expect(registrationGate(result([0.667, 0.877])).ok).toBe(false);
    expect(registrationGate(result([0.75, 0.854])).ok).toBe(false);
  });

  /**
   * 閾値を 0.95 から 0.90 へ下げた根拠を、試験でも押さえる（docs/12 §12.16.5）。
   *
   * `scripts/gate_probe.ts` で実素材の両側を測った値である。上を通し、
   * 下を落とす。**どちらか片方だけを見て線を引いてはいけない。**
   */
  it('実測した「正解」は通り、実測した「間違い」は落ちる', () => {
    // 正解（registerViews が解いた姿勢）
    expect(registrationGate(result([0.977, 0.969, 0.983])).ok).toBe(true);
    // yaw を 10° 取り違えた
    expect(registrationGate(result([0.965, 0.991, 0.85])).ok).toBe(false);
    // 合わせを一切しない（枠の値をそのまま信じた）
    expect(registrationGate(result([0.939, 0.84, 0.818])).ok).toBe(false);
    // 尺度が 15% 外れた
    expect(registrationGate(result([0.874, 0.984, 0.994])).ok).toBe(false);
  });

  it('0.95 のままなら実素材の正解を落としていた（下げた理由）', () => {
    // 利用者の写真3枚が合成できなかったのはこれ。0.949 は正しく合った値である。
    expect(0.949).toBeLessThan(0.95);
    expect(registrationGate(result([0.949, 0.961, 0.964])).ok).toBe(true);
  });

  it('1つでも割っていたら止める', () => {
    const g = registrationGate(result([0.985, 0.62, 0.991]));
    expect(g.ok).toBe(false);
    // いちばん合っていなかった view を指せること。理由の文面に使う。
    expect(g.worst.slot).toBe('right');
    expect(g.worst.insideRatio).toBeCloseTo(0.62, 5);
  });

  it('2枚でも効く（大きく壊れた場合だけ）', () => {
    expect(registrationGate(result([0.99, 0.8])).ok).toBe(false);
  });

  /**
   * **2枚では取り違えを見つけられない。** 認めて書いておく。
   *
   * 合成データで別人の2枚を組ませると 0.929〜0.934 で、実素材の正しい2枚
   * （0.920〜0.928）と重なる。どこに線を引いても分けられない。ここを
   * 「そのうち直る」と書くと、次に見た人が閾値をいじって正しい組を落とす。
   */
  it('2枚では別人の組を見分けられない（既知の限界）', () => {
    // 別人の2枚（合成データ、実測 0.929）。通ってしまう。
    expect(registrationGate(result([0.929, 0.972])).ok).toBe(true);
    // 正しい2枚（実素材、実測 0.920）。こちらのほうが低い。
    expect(registrationGate(result([0.92, 0.942])).ok).toBe(true);
    expect(0.92).toBeLessThan(0.929);
  });

  it('外した写真があることを、合成できていても伝える', () => {
    const msg = droppedMessage(['left'], ['front', 'right']);
    expect(msg).toContain('左向き');
    expect(msg).toContain('正面と右向き');
    expect(msg).toContain('2枚');
    // 撮り直せば元の枚数に戻せる、と道筋を示す
    expect(msg).toContain('3枚');
  });

  it('理由の文面に、どの写真がどれだけ合わなかったかが入る', () => {
    const g = registrationGate(result([0.985, 0.62, 0.991]));
    const msg = fallbackMessage(g.worst, 'front', g.threshold);
    expect(msg).toContain('右向き');
    expect(msg).toContain('62%');
    expect(msg).toContain('正面'); // 何で作ったかを伝える
  });
});

/**
 * 合わない写真を1枚外して、残りで作る（docs/12 §12.16.5）。
 *
 * **1枚が合わないだけで全部を捨ててはいけない。** 元の作りは、3枚のうち
 * 1枚でも閾値を割ると基準の1枚だけに戻していた。3枚撮って3枚とも処理させた
 * のに手元に残るのは1枚ぶん、では失うものが大きすぎる。
 *
 * ここは合成データで見る。実素材の数字は `scripts/gate_probe.ts` の側。
 */
describe('合わない写真を外して、残りで作る', () => {
  it('3枚とも合うなら、1枚も外さない', () => {
    const r = solveWithDrops(
      [body('front', 0), body('right', RIGHT), body('left', LEFT)],
      undefined,
    );
    expect(r.dropped).toEqual([]);
    expect(r.keep).toEqual([0, 1, 2]);
    expect(registrationGate(r.registration).ok).toBe(true);
  }, 180000);

  it('1枚だけ別人なら、その1枚を外して2枚で作る', () => {
    // 左向きの枠に、**体つきの違う被写体**を入れる。位置合わせは解けない。
    const steps: number[] = [];
    const r = solveWithDrops(
      [body('front', 0), body('right', RIGHT), other('left', LEFT)],
      (remaining) => steps.push(remaining),
    );

    expect(r.dropped).toEqual([2]); // 左向きだけ外れる
    expect(r.keep).toEqual([0, 1]); // 正面と右向きで作る
    expect(registrationGate(r.registration).ok).toBe(true);
    // 外したことを進捗に出している（画面が黙って減るのを避ける）
    expect(steps).toEqual([2]);
  }, 180000);

  /**
   * **最下位が基準なら、1枚も外さずに降りる**（PR #8 の Codex の指摘 P1）。
   *
   * 横2枚が「同じ別人」だと互いに整合するので、ずれて見えるのは基準のほうに
   * なる。ここで非基準を1枚外すと、残るのは「基準 ＋ もう1枚のずれた view」。
   * しかも2枚になると合格ラインが 0.85 に緩むので、**そのまま通って壊れた
   * 立体が出る**。ゲートが止めるはずのものを、ゲートの手前で作ってしまう。
   *
   * docs/12 §12.16.5 にこの降伏を書いておきながら、コードに書いていなかった。
   */
  it('最下位が基準なら、1枚も外さずに降りる', () => {
    // 実測でこの形（幅 2.5 倍・奥行き 0.4 倍）にすると
    // front 0.809 / right 0.904 / left 0.905 になり、基準が最下位で
    // かつ 3枚の合格ライン 0.90 を割る。
    const r = solveWithDrops(
      [body('front', 0), reshaped('right', RIGHT, 2.5, 0.4), reshaped('left', LEFT, 2.5, 0.4)],
      undefined,
    );

    const gate = registrationGate(r.registration);
    expect(gate.ok).toBe(false);
    expect(gate.worst.slot).toBe('front'); // 最下位は基準
    // **外していない。** 外すと2枚の緩い線（0.85）で通ってしまう。
    expect(r.dropped).toEqual([]);
    expect(r.keep).toEqual([0, 1, 2]);
  }, 180000);

  it('2枚しか無いなら外さない（基準は外せないので、残るのは1枚になってしまう）', () => {
    const r = solveWithDrops([body('front', 0), other('right', RIGHT)], undefined);
    expect(r.dropped).toEqual([]);
    expect(r.keep).toEqual([0, 1]);
    // **別人の2枚でも通ってしまう（実測 0.929）。** §12.16.5 の既知の限界で、
    // ここで落とせないことを試験にも書いておく。落とそうとして線を上げると、
    // 実素材の正しい2枚（0.920）のほうが先に落ちる。
    expect(registrationGate(r.registration).ok).toBe(true);
  }, 180000);
});
