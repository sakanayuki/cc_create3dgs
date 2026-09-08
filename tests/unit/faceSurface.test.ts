/**
 * 顔の立体を landmark から起こす（docs/03 §3.4.5）。
 *
 * ここで見るのは「顔専用モデルの出力を、深度の細部の帯として正しく
 * 入れられるか」。モデルそのものは動かさない（合成の landmark を使う）。
 */
import { describe, expect, it } from 'vitest';
import {
  applyFaceRelief,
  boxFromLandmarks,
  faceDepthSurface,
  headBoxFromMatte,
  type FaceLandmark,
} from '../../src/pipeline/geometry/faceSurface';

const G = 256;

/**
 * 直立した人物のシルエット。頭（幅 40）→ 首（幅 16）→ 肩から下（幅 100）。
 * 頭の中心は (128, 50)、首は y=72。
 */
function standingAlpha(): Uint8ClampedArray {
  const a = new Uint8ClampedArray(G * G);
  const band = (y0: number, y1: number, half: number): void => {
    for (let y = y0; y < y1; y++) {
      for (let x = 128 - half; x < 128 + half; x++) a[y * G + x] = 255;
    }
  };
  band(30, 70, 20); // 頭
  band(70, 78, 8); // 首
  band(78, 200, 50); // 胴
  return a;
}

/** 顔の landmark を模す。中央が手前（z が小さい）に出た面。 */
function fakeFace(cx: number, cy: number, r: number, n = 300): FaceLandmark[] {
  const pts: FaceLandmark[] = [];
  // 決定的な螺旋配置。乱数は使わない。
  for (let i = 0; i < n; i++) {
    const t = (i / n) * Math.PI * 2 * 8;
    const rr = r * Math.sqrt(i / n);
    const x = cx + rr * Math.cos(t);
    const y = cy + rr * Math.sin(t);
    // 中央ほど手前（鼻）
    const z = 20 * (rr / r) - 20;
    pts.push({ x, y, z });
  }
  return pts;
}

describe('マットから頭の位置を当てる', () => {
  it('首のくびれを見つけて頭を囲む', () => {
    const box = headBoxFromMatte(standingAlpha(), G, G);
    expect(box).not.toBeNull();
    const b = box as { x: number; y: number; width: number; height: number };
    const cx = b.x + b.width / 2;
    const cy = b.y + b.height / 2;
    expect(cx, `中心 x=${cx}`).toBeGreaterThan(118);
    expect(cx, `中心 x=${cx}`).toBeLessThan(138);
    expect(cy, `中心 y=${cy}`).toBeGreaterThan(35);
    expect(cy, `中心 y=${cy}`).toBeLessThan(70);
    // 頭（40px）は入り、胴（幅 100）まで飲み込まない
    expect(b.width).toBeGreaterThan(35);
    expect(b.width).toBeLessThan(90);
  });

  it('くびれが無ければ身長の割合で切る（頭でっかちにならない）', () => {
    // 幅が一定の柱。首が見つからない。
    const a = new Uint8ClampedArray(G * G);
    for (let y = 20; y < 220; y++) for (let x = 108; x < 148; x++) a[y * G + x] = 255;
    const b = headBoxFromMatte(a, G, G);
    expect(b).not.toBeNull();
    const box = b as { y: number; height: number };
    expect(box.y + box.height, '頭の箱が体の半分を超えています').toBeLessThan(20 + 200 * 0.4);
  });

  it('被写体が無ければ null', () => {
    expect(headBoxFromMatte(new Uint8ClampedArray(G * G), G, G)).toBeNull();
  });

  it('箱は画像の外へ出ない', () => {
    // 画像の左上隅に寄せた被写体
    const a = new Uint8ClampedArray(G * G);
    for (let y = 0; y < 40; y++) for (let x = 0; x < 30; x++) a[y * G + x] = 255;
    const b = headBoxFromMatte(a, G, G);
    expect(b).not.toBeNull();
    const box = b as { x: number; y: number; width: number; height: number };
    expect(box.x).toBeGreaterThanOrEqual(0);
    expect(box.y).toBeGreaterThanOrEqual(0);
    expect(box.x + box.width).toBeLessThanOrEqual(G);
    expect(box.y + box.height).toBeLessThanOrEqual(G);
  });
});

describe('landmark から切り直す', () => {
  it('顔を中心に、余裕をつけた正方を返す', () => {
    const pts = fakeFace(100, 120, 30);
    const b = boxFromLandmarks(pts, G, G, 1.5);
    expect(b).not.toBeNull();
    const box = b as { x: number; y: number; width: number; height: number };
    expect(box.width).toBe(box.height);
    expect(box.x + box.width / 2).toBeCloseTo(100, 0);
    expect(box.y + box.height / 2).toBeCloseTo(120, 0);
    // 顔（直径 60）に対して 1.5 倍
    expect(box.width).toBeGreaterThan(80);
    expect(box.width).toBeLessThan(100);
  });

  it('点が無ければ null', () => {
    expect(boxFromLandmarks([], G, G)).toBeNull();
  });
});

describe('landmark から顔の面を起こす', () => {
  const pts = fakeFace(128, 128, 40);
  const rect = { x: 68, y: 68, width: 120, height: 120 };

  it('中央が手前、外周が奥になる', () => {
    const s = faceDepthSurface(pts, rect);
    const at = (gx: number, gy: number): number =>
      s.depth[(gy - rect.y) * rect.width + (gx - rect.x)] as number;
    expect(at(128, 128), '中央').toBeLessThan(at(128, 158));
    expect(at(128, 128), '中央').toBeLessThan(at(158, 128));
  });

  it('重みは顔の楕円の中だけ立つ', () => {
    const s = faceDepthSurface(pts, rect);
    const w = (gx: number, gy: number): number =>
      s.weight[(gy - rect.y) * rect.width + (gx - rect.x)] as number;
    expect(w(128, 128), '中央').toBeCloseTo(1, 3);
    expect(w(70, 70), '四隅').toBe(0);
    expect(s.covered).toBeGreaterThan(1000);
    expect(s.covered, '楕円が四角を埋め尽くしています').toBeLessThan(rect.width * rect.height * 0.9);
  });

  it('点が少なすぎれば何も返さない', () => {
    const s = faceDepthSurface(pts.slice(0, 4), rect);
    expect(s.covered).toBe(0);
  });

  it('点の密度の偏りで楕円がずれない', () => {
    // 右半分にだけ点を増やす。重心±σ で作ると楕円が右へずれる。
    const biased = [...pts, ...fakeFace(150, 128, 12, 200)];
    const s = faceDepthSurface(biased, rect);
    // 重みの重心が、点の外接矩形の中心（x=128 付近）から大きく外れないこと
    let sx = 0;
    let sw = 0;
    for (let y = 0; y < rect.height; y++) {
      for (let x = 0; x < rect.width; x++) {
        const w = s.weight[y * rect.width + x] as number;
        sx += (rect.x + x) * w;
        sw += w;
      }
    }
    const cx = sx / sw;
    expect(cx, `重みの重心 x=${cx.toFixed(1)}`).toBeGreaterThan(120);
    expect(cx, `重みの重心 x=${cx.toFixed(1)}`).toBeLessThan(140);
  });
});

describe('顔の起伏を深度へ入れる', () => {
  const rect = { x: 68, y: 68, width: 120, height: 120 };
  const pts = fakeFace(128, 128, 40);
  const FOCAL = 500;

  /** 平らな顔（深度モデルが返すもの）。 */
  function flatDepth(): Float32Array {
    return new Float32Array(G * G).fill(1);
  }

  it('顔の中に起伏が入る', () => {
    const s = faceDepthSurface(pts, rect);
    const out = applyFaceRelief(flatDepth(), s, rect, G, G, FOCAL);
    const at = (gx: number, gy: number): number => out[gy * G + gx] as number;
    expect(at(128, 128), '鼻先が手前に出ていません').toBeLessThan(at(128, 155));
  });

  it('大きさは幾何から決まる（当てはめに頼らない）', () => {
    // landmark の z 幅は 20px。距離 1、焦点 500 なら 20/500 = 0.04 のはず。
    const s = faceDepthSurface(pts, rect);
    const out = applyFaceRelief(flatDepth(), s, rect, G, G, FOCAL);
    let lo = Infinity;
    let hi = -Infinity;
    for (let y = 0; y < rect.height; y++) {
      for (let x = 0; x < rect.width; x++) {
        if ((s.weight[y * rect.width + x] as number) < 0.5) continue;
        const v = out[(rect.y + y) * G + (rect.x + x)] as number;
        if (v < lo) lo = v;
        if (v > hi) hi = v;
      }
    }
    const span = hi - lo;
    expect(span, `顔の奥行き幅 ${span.toFixed(4)}`).toBeGreaterThan(0.015);
    expect(span, `顔の奥行き幅 ${span.toFixed(4)}`).toBeLessThan(0.06);
  });

  it('顔の外は動かさない', () => {
    const s = faceDepthSurface(pts, rect);
    const base = flatDepth();
    // 顔から離れた所に段差を置く
    for (let y = 200; y < 230; y++) for (let x = 20; x < 60; x++) base[y * G + x] = 1.5;
    const out = applyFaceRelief(base, s, rect, G, G, FOCAL);
    for (let y = 205; y < 225; y++) {
      for (let x = 25; x < 55; x++) {
        expect(out[y * G + x]).toBeCloseTo(base[y * G + x] as number, 4);
      }
    }
  });

  it('大域の位置は深度モデルのまま（顔だけ空間を移動しない）', () => {
    const s = faceDepthSurface(pts, rect);
    const base = flatDepth();
    for (let i = 0; i < base.length; i++) base[i] = 2.5; // 遠くに置いた頭
    const out = applyFaceRelief(base, s, rect, G, G, FOCAL);
    let sum = 0;
    let n = 0;
    for (let y = 0; y < rect.height; y++) {
      for (let x = 0; x < rect.width; x++) {
        if ((s.weight[y * rect.width + x] as number) < 0.5) continue;
        sum += out[(rect.y + y) * G + (rect.x + x)] as number;
        n++;
      }
    }
    expect(sum / n, '顔の平均深度が動いています').toBeCloseTo(2.5, 2);
  });

  it('面が小さすぎれば触らない', () => {
    const s = faceDepthSurface(pts.slice(0, 4), rect);
    const base = flatDepth();
    const out = applyFaceRelief(base, s, rect, G, G, FOCAL);
    for (let i = 0; i < base.length; i += 997) expect(out[i]).toBe(base[i]);
  });
});
