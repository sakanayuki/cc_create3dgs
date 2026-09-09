/**
 * 位置合わせを、既知の角度で回した合成データで検算する（docs/12 §12.15）。
 *
 * 実素材で測る前にここを通す。閉形式のシフト推定を合成球で確かめた
 * （docs/01 改訂 #16）のと同じ順序である。ここが通らなければ、
 * 実写で出た数字は「合わせ込みが効いた」のか「たまたま」なのか判別できない。
 *
 * 合成の体は `tests/helpers/syntheticBody.ts`。楕円体を縦に積んだ形で、
 * 高さごとに幅と奥行きの比が違う。**その比の違いが yaw の手がかりである。**
 */
import { describe, expect, it } from 'vitest';
import { DEFAULT_BODY, renderBody } from '../helpers/syntheticBody';
import { registerViews, type AlignView } from '../../src/pipeline/align/registerViews';

const W = 200;
const H = 320;
const CAM = { focalPx: 800, cx: W / 2, cy: H / 2 };
const deg = (rad: number): number => (rad * 180) / Math.PI;

/**
 * 枠と、被写体の回転角から1枚作る。
 *
 * 被写体の回転（`subjectYaw`）と、姿勢の yaw は**符号が逆**である。
 * 姿勢は「その view の点を基準へ運ぶ」向きなので（rigid.ts 冒頭）。
 */
function view(slot: 'front' | 'right' | 'left', subjectYaw: number): AlignView {
  const r = renderBody(DEFAULT_BODY, CAM, W, H, subjectYaw);
  return { slot, width: W, height: H, camera: CAM, alpha: r.alpha, depth: r.depth };
}

const RIGHT = -Math.PI / 2; // 画面の右を向く
const LEFT = Math.PI / 2;
const OPTS = { samplesPerView: 2500, gridLongSide: 128 } as const;

describe('複数枚の位置合わせ（合成データ）', () => {
  it('正面・右向き・左向きの3枚から、回転角を復元する', () => {
    const res = registerViews([view('front', 0), view('right', RIGHT), view('left', LEFT)], OPTS);

    expect(res.referenceIndex).toBe(0);
    expect(res.views[0]?.pose.yaw).toBe(0); // 基準は動かさない

    // 真値は右 +90°、左 −90°。合成データなので厳しく見る。
    expect(deg(res.views[1]?.pose.yaw ?? 0)).toBeCloseTo(90, -0.9); // ±6°
    expect(deg(res.views[2]?.pose.yaw ?? 0)).toBeCloseTo(-90, -0.9);

    // 尺度は3枚とも同じ体なので 1 のはず（縦の広がりで縛っている）
    expect(res.views[1]?.pose.scale).toBeCloseTo(1, 1);
    expect(res.views[2]?.pose.scale).toBeCloseTo(1, 1);
  }, 120000);

  it('2枚（正面＋右向き）でも動く', () => {
    const res = registerViews([view('front', 0), view('right', RIGHT)], OPTS);
    expect(deg(res.views[1]?.pose.yaw ?? 0)).toBeCloseTo(90, -0.9);
  }, 120000);

  it('90° から離れた角度は、寄るが当たらない（既知の限界）', () => {
    // 枠は「右向き」だが、実際には 60° しか回っていない写真。
    //
    // **実測 71.8°。初期値 90° から 18° ぶん寄るが、真値まで 12° 残る。**
    // 甘い閾値で通すのではなく、いまの精度をそのまま書いて固定しておく。
    // 90° 付近では 3° 以内に入る（上の2つ）ので、素材が「正面と真横」である限り
    // 困らないが、斜め 45° のような撮り方には足りない。docs/12 §12.15.2。
    const res = registerViews([view('front', 0), view('right', (-60 * Math.PI) / 180)], OPTS);
    const got = deg(res.views[1]?.pose.yaw ?? 0);
    expect(got).toBeLessThan(80); // 初期値の 90° から真値の側へ動いている
    expect(got).toBeGreaterThan(60); // まだ届いていない。届くようになったらこの行が落ちる
  }, 120000);

  it('枠を入れ違えると、残差が悪化して気づける（docs/12 R20）', () => {
    const ok = registerViews([view('front', 0), view('right', RIGHT)], OPTS);
    // 左向きの写真を「右向き」の枠に入れてしまった場合
    const swapped = registerViews([view('front', 0), view('right', LEFT)], OPTS);

    // 入れ違えたほうが目的関数が悪い。ここが逆転すると R20 の検出が成り立たない。
    expect(swapped.cost).toBeGreaterThan(ok.cost);
  }, 120000);
});
