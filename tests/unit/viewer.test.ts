/**
 * ビューアの可動範囲（docs/06 §6.7）。
 *
 * 半球カバー（決定 D2）なので、視点を自由に回させてはいけない。
 * 裏側の粗が見えると「壊れている」と受け取られる。
 */
import { describe, expect, it } from 'vitest';
import {
  dragToView,
  GestureTracker,
  LIMITS,
  rubberBand,
  type DragAnchor,
} from '../../src/ui/viewer';

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

describe('ドラッグの角度換算', () => {
  const W = 400;
  const H = 400;
  const at = (x: number, y: number, yaw = 0, pitch = 0): DragAnchor => ({ x, y, yaw, pitch });

  it('画面の幅いっぱいで快適範囲の2倍だけ回る', () => {
    const a = at(0, 0);
    // 左端から右端まで（+W）引くと、逆向きに 2×yawComfort
    expect(dragToView(a, W, 0, W, H).yaw).toBeCloseTo(-LIMITS.yawComfort * 4 * 1, 6);
    // 半分なら半分
    expect(dragToView(a, W / 2, 0, W, H).yaw).toBeCloseTo(-LIMITS.yawComfort * 2, 6);
  });

  it('小さく動かしても、その分だけちゃんと動く', () => {
    // 20px 動かして角度が動かないようでは「反応しない」と受け取られる。
    const a = at(200, 200, 0, 0);
    const moved = dragToView(a, 220, 200, W, H);
    expect(Math.abs(moved.yaw)).toBeGreaterThan(0.02); // 1度以上
    expect(Math.abs(moved.yaw)).toBeCloseTo((20 / W) * LIMITS.yawComfort * 4, 9);
  });

  it('動かさなければ視点は変わらない', () => {
    const a = at(50, 60, 0.3, -0.1);
    const same = dragToView(a, 50, 60, W, H);
    expect(same.yaw).toBeCloseTo(0.3, 12);
    expect(same.pitch).toBeCloseTo(-0.1, 12);
  });

  it('起点の視点を引き継ぐ（掴み直しで飛ばない）', () => {
    // 一度回してから掴み直す。新しい起点の角度から続きが始まること。
    const first = dragToView(at(0, 0, 0, 0), 100, 0, W, H);
    const second = dragToView(at(100, 0, first.yaw, first.pitch), 200, 0, W, H);
    const oneShot = dragToView(at(0, 0, 0, 0), 200, 0, W, H);
    expect(second.yaw).toBeCloseTo(oneShot.yaw, 12);
  });

  it('縦は pitch、横は yaw に効く', () => {
    const a = at(0, 0, 0, 0);
    expect(dragToView(a, 100, 0, W, H).pitch).toBe(0);
    expect(dragToView(a, 0, 100, W, H).yaw).toBe(0);
    expect(dragToView(a, 0, 100, W, H).pitch).toBeCloseTo(-(100 / H) * LIMITS.pitchComfort * 4, 9);
  });

  it('大きさが 0 でも壊れない', () => {
    const r = dragToView(at(0, 0, 0, 0), 10, 10, 0, 0);
    expect(Number.isFinite(r.yaw)).toBe(true);
    expect(Number.isFinite(r.pitch)).toBe(true);
  });
});

describe('指の出入りから視点を決める（実機で動かなかった件）', () => {
  const W = 400;
  const H = 400;

  /** ラバーバンドまで含めて、実際の Viewer と同じように視点を進める。 */
  function session() {
    let view = { yaw: 0, pitch: 0, distance: 1 };
    const g = new GestureTracker();
    const apply = (v: { yaw: number; pitch: number; distance?: number }): void => {
      view = {
        yaw: rubberBand(v.yaw, LIMITS.yawComfort, LIMITS.yawHard),
        pitch: rubberBand(v.pitch, LIMITS.pitchComfort, LIMITS.pitchHard),
        distance: Math.max(
          LIMITS.distanceMin,
          Math.min(LIMITS.distanceMax, v.distance ?? view.distance),
        ),
      };
    };
    return {
      get view() {
        return view;
      },
      down: (id: number, x: number, y: number) => g.down(id, x, y, view),
      move: (id: number, x: number, y: number) => {
        const next = g.move(id, x, y, W, H, view);
        if (next) apply(next);
      },
      up: (id: number) => g.up(id, view),
    };
  }

  it('少しずつ動かしても、指の総移動量ぶん回る', () => {
    // ここが壊れていた。1 イベント分の差分しか効かず、200px 引いても
    // 最後の 5px ぶん（約 0.6 度）しか回らないので、掴んでも動かないように
    // 見えていた。
    const s = session();
    s.down(1, 100, 200);
    for (let x = 105; x <= 300; x += 5) s.move(1, x, 200);
    const deg = (s.view.yaw * 180) / Math.PI;
    // 200px = 画面の半分 → 快適範囲(45°)の 2 倍 = 90°… だがラバーバンドで
    // 上限 60° に収まる。いずれにせよ「ほぼ 0」ではないこと。
    expect(Math.abs(deg)).toBeGreaterThan(40);
  });

  it('刻み方を変えても同じ場所なら同じ角度', () => {
    const coarse = session();
    coarse.down(1, 100, 200);
    coarse.move(1, 250, 200);
    const fine = session();
    fine.down(1, 100, 200);
    for (let x = 105; x <= 250; x += 5) fine.move(1, x, 200);
    expect(fine.view.yaw).toBeCloseTo(coarse.view.yaw, 12);
  });

  it('掴み直しても視点が飛ばない', () => {
    const s = session();
    s.down(1, 200, 200);
    s.move(1, 260, 200);
    const mid = s.view.yaw;
    s.up(1);
    expect(s.view.yaw).toBeCloseTo(mid, 12);
    // 別の場所で掴み直し、動かさなければ変わらない
    s.down(2, 50, 50);
    s.move(2, 50, 50);
    expect(s.view.yaw).toBeCloseTo(mid, 12);
    // そこから続きが始まる
    s.move(2, 80, 50);
    expect(s.view.yaw).toBeLessThan(mid);
  });

  it('2本目を置いた瞬間に視点が飛ばない', () => {
    const s = session();
    s.down(1, 150, 200);
    s.move(1, 250, 200);
    const before = s.view.yaw;
    s.down(2, 350, 220); // ピンチ開始
    s.move(2, 350, 220);
    expect(s.view.yaw).toBeCloseTo(before, 12);
  });

  it('ピンチで距離が変わり、離しても飛ばない', () => {
    const s = session();
    s.down(1, 150, 200);
    s.down(2, 250, 200);
    s.move(2, 350, 200); // 指の間隔を 2 倍に
    expect(s.view.distance).toBeLessThan(1);
    const zoomed = s.view.distance;
    s.up(2);
    expect(s.view.distance).toBeCloseTo(zoomed, 12);
    // 残った指で回しても距離は変わらない
    s.move(1, 170, 200);
    expect(s.view.distance).toBeCloseTo(zoomed, 12);
  });

  it('離した指の動きは無視する', () => {
    const s = session();
    s.down(1, 200, 200);
    s.move(1, 240, 200);
    const held = s.view.yaw;
    s.up(1);
    s.move(1, 400, 200);
    expect(s.view.yaw).toBeCloseTo(held, 12);
  });
});
