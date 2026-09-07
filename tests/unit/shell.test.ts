/**
 * シェル構築（docs/03 §3.6）。半球カバー（決定 D2）が成立するかの根拠。
 */
import { describe, expect, it } from 'vitest';
import { backColor, detectDepthEdges, thicknessMap } from '../../src/pipeline/4-shell';

const SIZE = 32;

/** 中央に円の被写体があるマット。 */
function circleAlpha(size = SIZE, r = 12): Uint8ClampedArray {
  const a = new Uint8ClampedArray(size * size);
  const c = size / 2;
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) a[y * size + x] = Math.hypot(x - c, y - c) <= r ? 255 : 0;
  }
  return a;
}

describe('厚みマップ（楕円断面）', () => {
  it('シルエット境界で厚み 0、中心で最大になる', () => {
    const alpha = circleAlpha();
    const t = thicknessMap(alpha, SIZE, SIZE, { maxThickness: 0.32, profile: 'ellipsoid' });
    const c = SIZE / 2;
    expect(t[c * SIZE + c]).toBeCloseTo(0.32, 5); // 中心 = 最大
    // 輪郭に近づくほど薄くなる
    expect(t[c * SIZE + (c + 11)] as number).toBeLessThan(0.32 * 0.5);
    // シルエットの外は厚み 0（そこで前面と背面が接して殻が閉じる）
    expect(t[c * SIZE + (c + 13)]).toBe(0);
    expect(t[0]).toBe(0);
  });

  it('楕円プロファイルが境界で滑らかに立ち上がる（円柱との違い）', () => {
    const alpha = circleAlpha();
    const ell = thicknessMap(alpha, SIZE, SIZE, { maxThickness: 1, profile: 'ellipsoid' });
    const cyl = thicknessMap(alpha, SIZE, SIZE, { maxThickness: 1, profile: 'cylinder' });
    const c = SIZE / 2;
    // 円柱は境界のすぐ内側でも最大値。角張って見える原因。
    expect(cyl[c * SIZE + (c + 11)]).toBeCloseTo(1, 5);
    // 楕円は境界付近で薄い（円柱の半分以下）
    expect(ell[c * SIZE + (c + 11)] as number).toBeLessThan(0.5);
    // 中心では両者とも最大
    expect(ell[c * SIZE + c]).toBeCloseTo(1, 5);
  });

  it('輪郭の1画素内側でも厚みは最大の3〜4割ある（球の幾何そのもの）', () => {
    // sqrt プロファイルは r=0 で傾きが無限大。半径 R の球は輪郭から 1px 内側で
    // 厚み 2R·sqrt(2/R) に達する。誤差ではなく正しい振る舞いなので固定しておく。
    const t = thicknessMap(circleAlpha(), SIZE, SIZE, { maxThickness: 1, profile: 'ellipsoid' });
    const c = SIZE / 2;
    const rim = t[c * SIZE + (c + 12)] as number;
    expect(rim).toBeGreaterThan(0.2);
    expect(rim).toBeLessThan(0.5);
  });

  it('厚みが中心に向かって単調に増える', () => {
    const t = thicknessMap(circleAlpha(), SIZE, SIZE, { maxThickness: 1, profile: 'ellipsoid' });
    const c = SIZE / 2;
    let prev = -1;
    for (let x = c + 12; x >= c; x--) {
      const v = t[c * SIZE + x] as number;
      expect(v).toBeGreaterThanOrEqual(prev - 1e-6);
      prev = v;
    }
  });

  it('被写体が無ければ全部 0', () => {
    const t = thicknessMap(new Uint8ClampedArray(SIZE * SIZE), SIZE, SIZE, {
      maxThickness: 1, profile: 'ellipsoid',
    });
    expect([...t].every((v) => v === 0)).toBe(true);
  });
});

describe('背面色', () => {
  /** 左半分が赤、右半分が青の前面色。鏡像かどうかを見分けるため。 */
  function halfColor(size = SIZE): Uint8ClampedArray {
    const c = new Uint8ClampedArray(size * size * 3);
    for (let y = 0; y < size; y++) {
      for (let x = 0; x < size; x++) {
        const i = (y * size + x) * 3;
        if (x < size / 2) { c[i] = 240; c[i + 1] = 30; c[i + 2] = 30; }
        else { c[i] = 30; c[i + 1] = 30; c[i + 2] = 240; }
      }
    }
    return c;
  }

  it('物体モードは水平反転する（左右の色が入れ替わる）', () => {
    const alpha = circleAlpha();
    const t = thicknessMap(alpha, SIZE, SIZE, { maxThickness: 1, profile: 'ellipsoid' });
    const back = backColor(halfColor(), alpha, t, SIZE, SIZE, {
      mode: 'mirror-h', shadeBase: 1, shadeRange: 0,
    });
    const hw = SIZE / 2;
    // 出力の左側は、元の右側（青）の色になっているはず
    const left = (8 * hw + 3) * 3;
    expect(back[left + 2] as number).toBeGreaterThan(back[left] as number);
  });

  it('人物モードは顔のパーツを持ち込まない（縁色の伸長）', () => {
    // 中央に強い特徴（緑）を置く。鏡像なら中央に緑が残る。
    const size = SIZE;
    const color = halfColor();
    const c = size / 2;
    for (let y = c - 3; y < c + 3; y++) {
      for (let x = c - 3; x < c + 3; x++) {
        const i = (y * size + x) * 3;
        color[i] = 0; color[i + 1] = 255; color[i + 2] = 0;
      }
    }
    const alpha = circleAlpha();
    const t = thicknessMap(alpha, size, size, { maxThickness: 1, profile: 'ellipsoid' });
    const back = backColor(color, alpha, t, size, size, {
      mode: 'edge-extend', shadeBase: 1, shadeRange: 0,
    });
    const hw = size / 2;
    const center = ((hw / 2) * hw + hw / 2) * 3;
    // 中央の緑（顔のパーツに相当）は背面に出てこない
    expect(back[center + 1] as number).toBeLessThan(150);
  });

  it('厚みに応じて減光する', () => {
    const alpha = circleAlpha();
    const t = thicknessMap(alpha, SIZE, SIZE, { maxThickness: 1, profile: 'ellipsoid' });
    const bright = backColor(halfColor(), alpha, t, SIZE, SIZE, {
      mode: 'mirror-h', shadeBase: 1.0, shadeRange: 0,
    });
    const dim = backColor(halfColor(), alpha, t, SIZE, SIZE, {
      mode: 'mirror-h', shadeBase: 0.5, shadeRange: 0,
    });
    const hw = SIZE / 2;
    const i = ((hw / 2) * hw + 4) * 3;
    expect(dim[i] as number).toBeLessThan(bright[i] as number);
  });

  it('半解像度で出す（背面は 1/4 密度なので）', () => {
    const alpha = circleAlpha();
    const t = thicknessMap(alpha, SIZE, SIZE, { maxThickness: 1, profile: 'ellipsoid' });
    const back = backColor(halfColor(), alpha, t, SIZE, SIZE, {
      mode: 'mirror-h', shadeBase: 1, shadeRange: 0,
    });
    expect(back.length).toBe((SIZE / 2) * (SIZE / 2) * 3);
  });
});

describe('深度エッジの検出', () => {
  it('奥向きの深度ギャップだけを拾う', () => {
    const w = 8;
    const h = 1;
    // 左半分が手前(1000)、右半分が奥(50000)
    const depth = Uint16Array.from([1000, 1000, 1000, 1000, 50000, 50000, 50000, 50000]);
    const alpha = new Uint8ClampedArray(w).fill(255);
    const edges = detectDepthEdges(depth, alpha, w, h, 0.018);

    // 手前側の最後の画素は、隣が奥なのでエッジ
    expect(edges[3] as number).toBeGreaterThan(0.5);
    // 奥側の最初の画素は、隣が手前＝ギャップは負なのでエッジではない
    expect(edges[4]).toBe(0);
    // 平坦なところはエッジではない
    expect(edges[0]).toBe(0);
    expect(edges[7]).toBe(0);
  });

  it('閾値以下のギャップは拾わない', () => {
    const depth = Uint16Array.from([1000, 1100, 1200, 1300]);
    const alpha = new Uint8ClampedArray(4).fill(255);
    expect([...detectDepthEdges(depth, alpha, 4, 1, 0.018)].every((v) => v === 0)).toBe(true);
  });

  it('被写体の外は見ない', () => {
    const depth = Uint16Array.from([1000, 60000, 1000, 1000]);
    const alpha = Uint8ClampedArray.from([0, 0, 255, 255]);
    const edges = detectDepthEdges(depth, alpha, 4, 1, 0.018);
    expect(edges[0]).toBe(0);
    expect(edges[1]).toBe(0);
  });

  it('ギャップの大きさを正規化して返す', () => {
    const depth = Uint16Array.from([0, 65535]);
    const alpha = new Uint8ClampedArray(2).fill(255);
    expect(detectDepthEdges(depth, alpha, 2, 1, 0.018)[0]).toBeCloseTo(1.0, 4);
  });
});
