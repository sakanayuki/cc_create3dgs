/**
 * ⓪ 前処理の幾何（docs/03 §3.2, §3.4）。
 * 画像デコードはブラウザ API に任せているので、ここでは座標の計算だけを見る。
 */
import { describe, expect, it } from 'vitest';
import { depthTiles, gridToSource, letterbox, subjectBBox } from '../../src/pipeline/0-preprocess';

describe('レターボックス', () => {
  it('横長の写真を中央に置き、上下に余白を作る', () => {
    const b = letterbox(4000, 3000, 1024);
    expect(b.width).toBe(1024);
    expect(b.height).toBe(768);
    expect(b.offsetX).toBe(0);
    expect(b.offsetY).toBe(128);
  });

  it('正方形の写真は余白なし', () => {
    const b = letterbox(2000, 2000, 1024);
    expect(b.width).toBe(1024);
    expect(b.height).toBe(1024);
    expect(b.offsetX).toBe(0);
    expect(b.offsetY).toBe(0);
  });

  it('小さい写真を引き伸ばさない', () => {
    const b = letterbox(400, 300, 1024);
    expect(b.scale).toBe(1);
    expect(b.width).toBe(400);
    expect(b.height).toBe(300);
    expect(b.offsetX).toBe(312);
  });

  it('グリッド座標から元の座標へ戻せる', () => {
    const b = letterbox(4000, 3000, 1024);
    const [sx, sy] = gridToSource(b, b.offsetX, b.offsetY);
    expect(sx).toBeCloseTo(0, 6);
    expect(sy).toBeCloseTo(0, 6);
    const [ex, ey] = gridToSource(b, b.offsetX + b.width, b.offsetY + b.height);
    expect(ex).toBeCloseTo(4000, 0);
    expect(ey).toBeCloseTo(3000, 0);
  });

  it('大きさが 0 なら拒む', () => {
    expect(() => letterbox(0, 100)).toThrow();
  });
});

describe('被写体の外接矩形', () => {
  const S = 64;
  const alpha = new Uint8ClampedArray(S * S);
  for (let y = 20; y < 40; y++) for (let x = 10; x < 30; x++) alpha[y * S + x] = 255;

  it('余白なしなら被写体にぴったり合う', () => {
    const r = subjectBBox(alpha, S, S, 128, 0);
    expect(r).toEqual({ x: 10, y: 20, width: 20, height: 20 });
  });

  it('余白を足しても画像の外へは出ない', () => {
    const r = subjectBBox(alpha, S, S, 128, 1.0);
    expect(r!.x).toBe(0);
    expect(r!.y).toBe(0);
    expect(r!.x + r!.width).toBeLessThanOrEqual(S);
    expect(r!.y + r!.height).toBeLessThanOrEqual(S);
  });

  it('被写体が無ければ null', () => {
    expect(subjectBBox(new Uint8ClampedArray(S * S), S, S)).toBeNull();
  });
});

describe('深度タイル', () => {
  it('4枚で外接矩形を覆い、隣と重なる', () => {
    const bbox = { x: 100, y: 100, width: 400, height: 400 };
    const tiles = depthTiles(bbox, 1024, 1024, 0.2);
    expect(tiles).toHaveLength(4);

    // 各タイルは 400 × 1.2 / 2 = 240
    for (const t of tiles) {
      expect(t.width).toBe(240);
      expect(t.height).toBe(240);
    }
    // 左上と右上は横に重なる
    const [tl, tr] = tiles as [typeof tiles[0], typeof tiles[0]];
    expect(tl.x + tl.width).toBeGreaterThan(tr.x);

    // 4枚の和が外接矩形を覆う
    const covered = new Set<string>();
    for (const t of tiles) {
      for (let y = t.y; y < t.y + t.height; y += 1) covered.add(`${y}`);
    }
    for (let y = bbox.y; y < bbox.y + bbox.height; y++) expect(covered.has(`${y}`)).toBe(true);
  });

  it('画像の外へはみ出さない', () => {
    const bbox = { x: 0, y: 0, width: 1024, height: 1024 };
    for (const t of depthTiles(bbox, 1024, 1024, 0.2)) {
      expect(t.x).toBeGreaterThanOrEqual(0);
      expect(t.y).toBeGreaterThanOrEqual(0);
      expect(t.x + t.width).toBeLessThanOrEqual(1024);
      expect(t.y + t.height).toBeLessThanOrEqual(1024);
    }
  });
});
