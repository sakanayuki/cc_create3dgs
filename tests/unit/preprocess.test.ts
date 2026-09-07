/**
 * ⓪ 前処理の幾何（docs/03 §3.2, §3.4）。
 * 画像デコードはブラウザ API に任せているので、ここでは座標の計算だけを見る。
 */
import { describe, expect, it } from 'vitest';
import {
  depthTiles,
  expandToSquare,
  gridToSource,
  letterbox,
  subjectBBox,
} from '../../src/pipeline/0-preprocess';

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
  it('タイルは必ず正方形（518² へ歪みなく渡せる）', () => {
    // 立った人物のような縦長の外接矩形でも、正方でなければならない。
    // 縦長のまま 518² へ伸ばすと顔が横に伸びる（実測 1.67 倍）。
    for (const bbox of [
      { x: 201, y: 108, width: 550, height: 916 }, // 実写の全身
      { x: 100, y: 100, width: 400, height: 400 }, // 正方
      { x: 0, y: 300, width: 900, height: 300 },   // 横長
    ]) {
      for (const t of depthTiles(bbox, 1024, 1024, 0.2)) {
        expect(t.width).toBe(t.height);
      }
    }
  });

  it('縦長の被写体では横に割らない（顔を継ぎ目が横切らない）', () => {
    // 実写で顔が割れた条件。外接矩形の左右中央に継ぎ目が来ると、
    // 左右のタイルが別々に尺度を合わせるので顔の中央に段差が入る。
    const bbox = { x: 201, y: 108, width: 550, height: 916 };
    const tiles = depthTiles(bbox, 1024, 1024, 0.2);
    for (const t of tiles) {
      // どのタイルも外接矩形の横幅を丸ごと覆う
      expect(t.x).toBeLessThanOrEqual(bbox.x);
      expect(t.x + t.width).toBeGreaterThanOrEqual(bbox.x + bbox.width);
    }
  });

  it('頭部が丸ごと 1 枚のタイルに収まる', () => {
    // 実写で測った頭部の位置。ここが 2 枚にまたがると顔が割れる。
    const bbox = { x: 201, y: 108, width: 550, height: 916 };
    const head = { x0: 401, y0: 152, x1: 579, y1: 440 };
    const tiles = depthTiles(bbox, 1024, 1024, 0.2);
    const holds = tiles.some(
      (t) => t.x <= head.x0 && t.x + t.width >= head.x1 && t.y <= head.y0 && t.y + t.height >= head.y1,
    );
    expect(holds).toBe(true);
  });

  it('長辺を覆い、隣と重なる', () => {
    const bbox = { x: 201, y: 108, width: 550, height: 916 };
    const tiles = depthTiles(bbox, 1024, 1024, 0.2);
    expect(tiles.length).toBeGreaterThanOrEqual(2);

    const sorted = [...tiles].sort((a, b) => a.y - b.y);
    // 先頭は矩形の上端、末尾は下端に届く
    expect(sorted[0]!.y).toBeLessThanOrEqual(bbox.y);
    expect(sorted[sorted.length - 1]!.y + sorted[sorted.length - 1]!.height).toBeGreaterThanOrEqual(
      bbox.y + bbox.height,
    );
    // 隣り合うタイルは重なる（融合には重なりが要る）
    for (let i = 1; i < sorted.length; i++) {
      expect(sorted[i - 1]!.y + sorted[i - 1]!.height).toBeGreaterThan(sorted[i]!.y);
    }
  });

  it('外接矩形が正方に近ければ 1 枚で済ませる', () => {
    // バストアップの構図。割る理由が無い。
    const tiles = depthTiles({ x: 200, y: 150, width: 600, height: 620 }, 1024, 1024, 0.2);
    expect(tiles).toHaveLength(1);
    expect(tiles[0]!.width).toBe(tiles[0]!.height);
  });

  it('画像の外へはみ出さない', () => {
    for (const bbox of [
      { x: 0, y: 0, width: 1024, height: 1024 },
      { x: 900, y: 900, width: 124, height: 124 },
      { x: 201, y: 108, width: 550, height: 916 },
    ]) {
      for (const t of depthTiles(bbox, 1024, 1024, 0.2)) {
        expect(t.x).toBeGreaterThanOrEqual(0);
        expect(t.y).toBeGreaterThanOrEqual(0);
        expect(t.x + t.width).toBeLessThanOrEqual(1024);
        expect(t.y + t.height).toBeLessThanOrEqual(1024);
      }
    }
  });
});

describe('正方への拡張', () => {
  it('縦長の矩形を、元を含む正方形に広げる', () => {
    const r = expandToSquare({ x: 201, y: 108, width: 330, height: 550 }, 1024, 1024);
    expect(r.width).toBe(r.height);
    expect(r.width).toBe(550);
    expect(r.x).toBeLessThanOrEqual(201);
    expect(r.x + r.width).toBeGreaterThanOrEqual(201 + 330);
    expect(r.y).toBeLessThanOrEqual(108);
    expect(r.y + r.height).toBeGreaterThanOrEqual(108 + 550);
  });

  it('画像の端でも外へはみ出さない', () => {
    for (const r0 of [
      { x: 0, y: 0, width: 100, height: 400 },
      { x: 924, y: 700, width: 100, height: 324 },
    ]) {
      const r = expandToSquare(r0, 1024, 1024);
      expect(r.x).toBeGreaterThanOrEqual(0);
      expect(r.y).toBeGreaterThanOrEqual(0);
      expect(r.x + r.width).toBeLessThanOrEqual(1024);
      expect(r.y + r.height).toBeLessThanOrEqual(1024);
      // 元の矩形を含んでいる
      expect(r.x).toBeLessThanOrEqual(r0.x);
      expect(r.y).toBeLessThanOrEqual(r0.y);
    }
  });

  it('すでに正方形ならそのまま', () => {
    const r = expandToSquare({ x: 10, y: 20, width: 300, height: 300 }, 1024, 1024);
    expect(r).toEqual({ x: 10, y: 20, width: 300, height: 300 });
  });
});
