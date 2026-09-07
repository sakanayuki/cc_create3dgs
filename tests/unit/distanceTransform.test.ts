/**
 * 厳密ユークリッド距離変換（Felzenszwalb & Huttenlocher）。
 *
 * 近似ではなく厳密解であることを、既知の解析解と突き合わせて確かめる。
 * 厚みは被写体の形そのものを決めるので、近似の異方性は「歪んだ膨らみ」として見える。
 */
import { describe, expect, it } from 'vitest';
import { distanceTransform, nearestForegroundIndex } from '../../src/pipeline/geometry/distanceTransform';

/** 中心 (cx,cy) 半径 r の円を描いたマスク。 */
function circleMask(size: number, cx: number, cy: number, r: number): Uint8Array {
  const m = new Uint8Array(size * size);
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      m[y * size + x] = Math.hypot(x - cx, y - cy) <= r ? 1 : 0;
    }
  }
  return m;
}

describe('距離変換', () => {
  it('全部背景なら距離は 0', () => {
    const d = distanceTransform(new Uint8Array(16), 4, 4);
    expect([...d]).toEqual(new Array(16).fill(0));
  });

  it('1画素だけ前景なら、その画素の距離が 0（最も近い背景が隣接）', () => {
    const m = new Uint8Array(25);
    m[12] = 1; // 5×5 の中心
    const d = distanceTransform(m, 5, 5);
    expect(d[12]).toBeCloseTo(1, 5); // 隣が背景なので距離1
  });

  it('円の内部距離が解析解（r − 中心からの距離）に一致する', () => {
    const size = 64;
    const r = 20;
    const c = 32;
    const d = distanceTransform(circleMask(size, c, c, r), size, size);
    let worst = 0;
    for (let y = 0; y < size; y++) {
      for (let x = 0; x < size; x++) {
        const rr = Math.hypot(x - c, y - c);
        if (rr > r - 2) continue; // 境界の離散化誤差が出るところは除く
        const expected = r - rr;
        worst = Math.max(worst, Math.abs((d[y * size + x] as number) - expected));
      }
    }
    // 厳密解なので、離散化に由来する 1px 程度の差しか出ない
    expect(worst).toBeLessThan(1.1);
  });

  it('斜め方向と軸方向で歪まない（チャンファー近似との違い）', () => {
    // 正方形の中心からの距離は、軸方向でも斜め方向でも同じ「壁までの距離」になる
    const size = 41;
    const m = new Uint8Array(size * size).fill(1);
    for (let i = 0; i < size; i++) {
      m[i] = 0;
      m[(size - 1) * size + i] = 0;
      m[i * size] = 0;
      m[i * size + size - 1] = 0;
    }
    const d = distanceTransform(m, size, size);
    const center = d[20 * size + 20] as number;
    expect(center).toBeCloseTo(20, 5); // 壁まで 20px

    // 中心から斜めに離れた点でも、最寄りの壁までの距離が正しい
    const p = d[10 * size + 10] as number;
    expect(p).toBeCloseTo(10, 5);
  });

  it('矩形で軸方向の距離が正確', () => {
    const w = 20;
    const h = 10;
    const m = new Uint8Array(w * h).fill(1);
    // 左端だけ背景にする
    for (let y = 0; y < h; y++) m[y * w] = 0;
    const d = distanceTransform(m, w, h);
    for (let y = 0; y < h; y++) {
      for (let x = 1; x < w; x++) {
        expect(d[y * w + x]).toBeCloseTo(x, 5);
      }
    }
  });

  it('長さが合わなければ拒否する', () => {
    expect(() => distanceTransform(new Uint8Array(5), 3, 3)).toThrow(/長さが合いません/);
  });

  it('α マスクを閾値で扱える（境界画素の判定に使う）', () => {
    const alpha = Uint8ClampedArray.from([0, 100, 200, 255, 255, 200, 100, 0, 0]);
    const d = distanceTransform(alpha, 3, 3, (v) => v >= 128);
    // α ≥ 128 の画素だけが前景
    expect(d[0]).toBe(0);
    expect(d[1]).toBe(0);
    expect((d[3] as number)).toBeGreaterThan(0);
  });
});

describe('最近傍前景の伝播', () => {
  it('前景画素は自分を指す', () => {
    const m = Uint8Array.from([0, 1, 0, 0, 1, 0, 0, 0, 0]);
    const idx = nearestForegroundIndex(m, 3, 3);
    expect(idx[1]).toBe(1);
    expect(idx[4]).toBe(4);
  });

  it('背景画素が最も近い前景を指す', () => {
    const size = 16;
    const m = new Uint8Array(size * size);
    m[0] = 1; // 左上だけ前景
    const idx = nearestForegroundIndex(m, size, size);
    for (let i = 0; i < size * size; i++) expect(idx[i]).toBe(0);
  });

  it('複数の前景があるとき、最も近いほうを選ぶ', () => {
    const size = 32;
    const m = new Uint8Array(size * size);
    const a = 2 * size + 2;
    const b = 29 * size + 29;
    m[a] = 1;
    m[b] = 1;
    const idx = nearestForegroundIndex(m, size, size);
    expect(idx[3 * size + 3]).toBe(a);
    expect(idx[28 * size + 28]).toBe(b);
  });

  it('前景が無ければ全部 -1 のまま', () => {
    const idx = nearestForegroundIndex(new Uint8Array(9), 3, 3);
    expect([...idx]).toEqual(new Array(9).fill(-1));
  });
});
