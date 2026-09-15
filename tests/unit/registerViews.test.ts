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

  it('枠を入れ違えても、顔の符号で直す（docs/12 R20）', () => {
    // 左向きの写真を「右向き」の枠に入れてしまった場合。
    // 顔から測ったヨーが負なら、枠と食い違うので逆向きから探し直す。
    const v = view('right', LEFT);
    const res = registerViews([view('front', 0), { ...v, headYawDeg: -45 }], OPTS);

    // **符号が戻ることが、この仕掛けの仕事である。**
    // 大きさは実測 −69.2°（真値 −90° に対して 21° の誤差）。枠どおりに入れた
    // ときの精度（±6°）には届かない。逆向きから探し直すと、走査の中心が
    // 変わるぶん別の谷に落ち着くため。符号が戻れば向きの取り違えは消えるので、
    // ここは甘い閾値ではなく、いまの実力をそのまま書いて固定しておく。
    const got = deg(res.views[1]?.pose.yaw ?? 0);
    expect(got).toBeLessThan(0);
    expect(Math.abs(got - -90)).toBeLessThan(25);
    expect(res.views[1]?.slotFlipped).toBe(true);
    expect(res.anySlotFlipped).toBe(true);
  }, 120000);

  it('顔が無ければ枠を信じる（目的関数では符号を決めない）', () => {
    // **合成の体は左右がほぼ対称なので、目的関数は符号を見分けられない**
    // （正しい配置 0.00901 に対し、両方反転が 0.00908。差は 0.7%）。
    // その揺らぎで利用者の指定を覆さない、というのがここの取り決めである。
    // 顔の手がかりを渡さなければ、入れ違えていても枠のまま解く。
    const res = registerViews([view('front', 0), view('right', LEFT)], OPTS);
    expect(res.views[1]?.slotFlipped).toBe(false);
    expect(deg(res.views[1]?.pose.yaw ?? 0)).toBeGreaterThan(0); // 枠どおり + 側
  }, 120000);

  it('枠が正しければ、読み替えたとは言わない', () => {
    const res = registerViews(
      [view('front', 0), { ...view('right', RIGHT), headYawDeg: 40 }, { ...view('left', LEFT), headYawDeg: -40 }],
      OPTS,
    );
    expect(res.anySlotFlipped).toBe(false);
    for (const v of res.views) expect(v.slotFlipped).toBe(false);
  }, 120000);

  it('顔から測ったヨーの符号が枠と食い違えば、そう言う', () => {
    // 顔の手がかりは大きさが当てにならないので符号だけ使う（docs/12 §12.15.5）。
    const views = [view('front', 0), view('right', LEFT)];
    const withHint = [views[0] as AlignView, { ...(views[1] as AlignView), headYawDeg: -45 }];
    const res = registerViews(withHint, OPTS);
    expect(res.views[1]?.headYawDisagrees).toBe(true);
  }, 120000);
});
