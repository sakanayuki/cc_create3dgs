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
  headDepthTile,
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

/**
 * 顔の landmark を模す。中央が手前（z が小さい）に出た面。
 *
 * @param bump  中央に置く細部（鼻に当たる）の高さ[px]。0 なら大域だけの面。
 * @param sigma その細部の広がり[px]。
 * @param dome  面ぜんたいの丸み[px]（大域）。
 */
function fakeFace(
  cx: number,
  cy: number,
  r: number,
  n = 300,
  bump = 0,
  sigma = 8,
  dome = 20,
): FaceLandmark[] {
  const pts: FaceLandmark[] = [];
  // 決定的な螺旋配置。乱数は使わない。
  for (let i = 0; i < n; i++) {
    const t = (i / n) * Math.PI * 2 * 8;
    const rr = r * Math.sqrt(i / n);
    const x = cx + rr * Math.cos(t);
    const y = cy + rr * Math.sin(t);
    // 大域は中央ほど手前の、なだらかな丸み。
    const z = dome * (rr / r) - dome - bump * Math.exp(-((rr / sigma) ** 2));
    pts.push({ x, y, z });
  }
  return pts;
}

/** z が一定の面。大域しか持たない landmark 面を作る。 */
function flatFace(cx: number, cy: number, r: number, n = 300): FaceLandmark[] {
  return fakeFace(cx, cy, r, n).map((p) => ({ x: p.x, y: p.y, z: 0 }));
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
  /** 大域の丸み（20px）に、鼻に当たる細部（15px）を載せた顔。 */
  const pts = fakeFace(128, 128, 40, 300, 15);
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
    // 細部（鼻）の高さは landmark で 15px。距離 1、焦点 500 なら
    // 15 / 500 = 0.03 に当たる。平滑化で少し目減りする。
    const s = faceDepthSurface(pts, rect);
    const base = flatDepth();
    const out = applyFaceRelief(base, s, rect, G, G, FOCAL);
    const rise = (base[128 * G + 128] as number) - (out[128 * G + 128] as number);
    expect(rise, `鼻の高さ ${rise.toFixed(4)}`).toBeGreaterThan(0.015);
    expect(rise, `鼻の高さ ${rise.toFixed(4)}`).toBeLessThan(0.035);
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

/**
 * v2.6.1 の回帰（docs/09 §V13）。
 *
 * landmark 面は「カメラを向いた顔の統計的な型」であって、この人の頭が
 * どれだけ丸いかは知らない。以前はラプラシアンブレンドで大域まで
 * landmark 面に置き換わっており、実測（プールの写真）で深度モデルの顔
 * （奥行き 49mm）と landmark 面（17mm）が **15mm** 食い違ったまま楕円の
 * 重みで切られ、左頬が 7mm 手前へ、額が奥へ動く低周波のうねりになった。
 * 顔の輪郭に沿った隆起として見える。
 */
describe('顔の輪郭に沿った隆起（v2.6.1）', () => {
  const rect = { x: 68, y: 68, width: 120, height: 120 };
  const FOCAL = 500;

  function domeDepth(span: number): Float32Array {
    const d = new Float32Array(G * G).fill(1);
    for (let y = 0; y < G; y++) {
      for (let x = 0; x < G; x++) {
        const r = Math.min(1, Math.hypot(x - 128, y - 128) / 60);
        d[y * G + x] = 1 + span * r * r;
      }
    }
    return d;
  }

  it('landmark 面が大域しか持たないなら深度は 1 画素も動かない', () => {
    // z が一定 = 細部が無い面。足すべきものが無いのだから、何も足さない。
    const s = faceDepthSurface(flatFace(128, 128, 40), rect);
    expect(s.covered).toBeGreaterThan(1000);
    const base = domeDepth(0.05);
    const out = applyFaceRelief(base, s, rect, G, G, FOCAL);
    let worst = 0;
    for (let i = 0; i < base.length; i++) {
      worst = Math.max(worst, Math.abs((out[i] as number) - (base[i] as number)));
    }
    expect(worst, `最大の動き ${worst.toFixed(6)}`).toBeLessThan(1e-6);
  });

  it('深度モデルが持つ顔の丸みを landmark 面が上書きしない', () => {
    // 深度モデルは 50mm の丸みを持ち、landmark 面は 10mm ぶんしか持たない
    // （実測でも 49mm 対 17mm だった）。食い違うのは landmark 面のほうが
    // 型でしかないからで、出力に残る丸みは深度モデルのままであること。
    const s = faceDepthSurface(fakeFace(128, 128, 40, 300, 15, 8, 5), rect);
    const base = domeDepth(0.05);
    const out = applyFaceRelief(base, s, rect, G, G, FOCAL);
    // 顔の縁（半径 36px）と中央の差＝丸み。鼻の細部を避けて縁だけで測る。
    const ringOf = (d: ArrayLike<number>, r: number): number => {
      let sum = 0;
      let n = 0;
      for (let a = 0; a < 64; a++) {
        const th = (a / 64) * Math.PI * 2;
        const x = Math.round(128 + r * Math.cos(th));
        const y = Math.round(128 + r * Math.sin(th));
        sum += d[y * G + x] as number;
        n++;
      }
      return sum / n;
    };
    const before = ringOf(base, 36) - ringOf(base, 12);
    const after = ringOf(out, 36) - ringOf(out, 12);
    expect(before).toBeGreaterThan(0.01);
    expect(after / before, `丸みの残り ${(after / before).toFixed(2)} 倍`).toBeGreaterThan(0.8);
  });

  it('楕円の羽根のところに隆起の輪ができない', () => {
    const s = faceDepthSurface(fakeFace(128, 128, 40, 300, 15, 8, 5), rect);
    const base = domeDepth(0.05);
    const out = applyFaceRelief(base, s, rect, G, G, FOCAL);
    // 半径ごとの平均の差。羽根は t=0.75..1.1、つまり半径 28..42px に当たる。
    const prof = (r: number): number => {
      let sum = 0;
      let n = 0;
      for (let a = 0; a < 128; a++) {
        const th = (a / 128) * Math.PI * 2;
        const x = Math.round(128 + r * Math.cos(th));
        const y = Math.round(128 + r * Math.sin(th));
        sum += (out[y * G + x] as number) - (base[y * G + x] as number);
        n++;
      }
      return sum / n;
    };
    // 鼻の高さ（中央の動き）を物差しにする。
    const nose = Math.abs(prof(0));
    expect(nose, `鼻の高さ ${(nose * 1000).toFixed(2)}mm`).toBeGreaterThan(0.005);
    for (let r = 26; r <= 46; r += 2) {
      const v = prof(r);
      expect(Math.abs(v) / nose, `半径 ${r}px の動き ${(v * 1000).toFixed(2)}mm`).toBeLessThan(0.25);
    }
  });
});

/**
 * 顔だけを見る深度タイル（docs/09 §V18）。全身写真では顔がタイルの中の
 * 一部でしかなく、深度モデルが鼻も眼窩も出さない。
 */
describe('頭だけを見る深度タイル', () => {
  it('頭を囲む正方形を、体のタイルより小さく返す', () => {
    const t = headDepthTile(standingAlpha(), G, G, 120);
    expect(t).not.toBeNull();
    const r = t as { x: number; y: number; width: number; height: number };
    expect(r.width).toBe(r.height);
    // 頭（幅 40、余裕込みで 48）の 1.4 倍あたり
    expect(r.width).toBeGreaterThan(60);
    expect(r.width).toBeLessThan(90);
    // 頭の中心（128, 50）を含む
    expect(r.x).toBeLessThan(128);
    expect(r.x + r.width).toBeGreaterThan(128);
    expect(r.y).toBeLessThan(50);
    expect(r.y + r.height).toBeGreaterThan(50);
  });

  it('体のタイルと大差ないなら足さない', () => {
    // 体のタイルが 80px なら、頭のタイル（約 67px）は 0.75 倍を超える
    expect(headDepthTile(standingAlpha(), G, G, 80)).toBeNull();
  });

  it('小さすぎるタイルは足さない', () => {
    // 遠くに写った人物。頭は 10px しかない。
    const a = new Uint8ClampedArray(G * G);
    const band = (y0: number, y1: number, x0: number, x1: number): void => {
      for (let y = y0; y < y1; y++) for (let x = x0; x < x1; x++) a[y * G + x] = 255;
    };
    band(4, 14, 123, 133); // 頭
    band(14, 18, 126, 130); // 首
    band(18, 60, 113, 143); // 胴
    expect(headDepthTile(a, G, G, 200)).toBeNull();
  });

  it('被写体が無ければ null', () => {
    expect(headDepthTile(new Uint8ClampedArray(G * G), G, G, 200)).toBeNull();
  });

  it('タイルは画像の外へ出ない', () => {
    const a = new Uint8ClampedArray(G * G);
    for (let y = 0; y < 40; y++) for (let x = 0; x < 30; x++) a[y * G + x] = 255;
    for (let y = 44; y < 200; y++) for (let x = 0; x < 60; x++) a[y * G + x] = 255;
    const t = headDepthTile(a, G, G, 400);
    if (t) {
      expect(t.x).toBeGreaterThanOrEqual(0);
      expect(t.y).toBeGreaterThanOrEqual(0);
      expect(t.x + t.width).toBeLessThanOrEqual(G);
      expect(t.y + t.height).toBeLessThanOrEqual(G);
    }
  });
});
