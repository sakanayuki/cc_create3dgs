/**
 * ② 深度の2パス融合（docs/03 §3.4）。
 *
 * 見たいのは3つ。
 *   ・タイルごとに違うスケールを、重なりで合わせられるか
 *   ・ラプラシアンブレンドが直流成分を保つか（畳み直しの正しさ）
 *   ・タイル境界に段差を残さないか
 */
import { describe, expect, it } from 'vitest';
import { DEFAULT_FUSE_PARAMS, fitAffine, fuseDepth } from '../../src/pipeline/2-depth';
import {
  blendLaplacian,
  collapse,
  gaussianPyramid,
  laplacianPyramid,
} from '../../src/pipeline/geometry/pyramid';

const S = 64;

/** 左から右へ滑らかに増える深度。 */
function ramp(width = S, height = S): Float32Array {
  const d = new Float32Array(width * height);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) d[y * width + x] = 1 + x / width;
  }
  return d;
}

describe('最小二乗フィット', () => {
  it('倍率と切片を取り戻す', () => {
    const src = ramp();
    const dst = new Float32Array(src.length);
    for (let i = 0; i < src.length; i++) dst[i] = 2.5 * (src[i] as number) + 0.75;
    const mask = new Float32Array(src.length).fill(1);
    const fit = fitAffine(src, dst, mask, true);
    expect(fit.a).toBeCloseTo(2.5, 6);
    expect(fit.b).toBeCloseTo(0.75, 6);
    expect(fit.rmse).toBeLessThan(1e-6);
  });

  it('切片を許さないときは倍率だけを解く', () => {
    const src = ramp();
    const dst = new Float32Array(src.length);
    for (let i = 0; i < src.length; i++) dst[i] = 3 * (src[i] as number);
    const mask = new Float32Array(src.length).fill(1);
    const fit = fitAffine(src, dst, mask, false);
    expect(fit.a).toBeCloseTo(3, 6);
    expect(fit.b).toBe(0);
  });

  it('マスクの外は無視する', () => {
    const src = ramp();
    const dst = new Float32Array(src.length);
    const mask = new Float32Array(src.length);
    for (let i = 0; i < src.length; i++) {
      // マスク内は 2 倍、外はでたらめ
      const inside = i % 2 === 0;
      mask[i] = inside ? 1 : 0;
      dst[i] = inside ? 2 * (src[i] as number) : 99;
    }
    const fit = fitAffine(src, dst, mask, false);
    expect(fit.a).toBeCloseTo(2, 6);
  });

  it('画素が足りなければ恒等を返す', () => {
    const z = new Float32Array(10);
    const fit = fitAffine(z, z, new Float32Array(10), true);
    expect(fit.a).toBe(1);
    expect(fit.b).toBe(0);
    expect(fit.count).toBe(0);
  });
});

describe('ピラミッド', () => {
  it('ラプラシアンを畳み直すと元に戻る', () => {
    const src = ramp();
    const back = collapse(laplacianPyramid(src, S, S, 5));
    for (let i = 0; i < src.length; i++) {
      expect(back[i]).toBeCloseTo(src[i] as number, 4);
    }
  });

  it('段ごとに半分になる', () => {
    const g = gaussianPyramid(ramp(), S, S, 4);
    expect(g.map((l) => l.width)).toEqual([64, 32, 16, 8]);
  });

  it('重みが片方に寄っていればその入力を返す', () => {
    const a = new Float32Array(S * S).fill(1);
    const b = new Float32Array(S * S).fill(5);
    const wa = new Float32Array(S * S).fill(1);
    const wb = new Float32Array(S * S).fill(0);
    const out = blendLaplacian([a, b], [wa, wb], S, S, 4);
    for (let i = 0; i < out.length; i++) expect(out[i]).toBeCloseTo(1, 4);
  });
});

describe('2パス融合', () => {
  /** 全体パスと、その一部を 3 倍のスケールで見ているタイル。 */
  function scenario() {
    const global = ramp();
    const rect = { x: 16, y: 16, width: 32, height: 32 };
    const depth = new Float32Array(rect.width * rect.height);
    for (let y = 0; y < rect.height; y++) {
      for (let x = 0; x < rect.width; x++) {
        const g = global[(rect.y + y) * S + (rect.x + x)] as number;
        depth[y * rect.width + x] = g * 3; // タイルは独自スケール
      }
    }
    return { global, tile: { depth, rect } };
  }

  it('尺度合わせに被写体の画素だけを使う（v2.3、実写で判明）', () => {
    // 背景は「合わせられない」領域である。単眼深度は遠景に安定した値を
    // 返さないので、切り出し方が違えば空やプールの深度は別物になる。
    // 実写では、頭のタイルの 60% が背景で、その背景に引かれて倍率が
    // 1.05 → 1.22 までずれ、被写体の深度が丸ごと押し出されていた。
    const global = ramp();
    const rect = { x: 8, y: 8, width: 48, height: 48 };
    const depth = new Float32Array(rect.width * rect.height);
    const alpha = new Uint8Array(S * S);
    for (let y = 0; y < rect.height; y++) {
      for (let x = 0; x < rect.width; x++) {
        const gi = (rect.y + y) * S + (rect.x + x);
        const g = global[gi] as number;
        // 被写体は素直に 2 倍。背景はまったく無関係な値を返している。
        const isSubject = x >= 16 && x < 32 && y >= 16 && y < 32;
        if (isSubject) alpha[gi] = 255;
        depth[y * rect.width + x] = isSubject ? g * 2 : 0.05 * (x + y);
      }
    }
    const tile = { depth, rect };
    const opts = { ...DEFAULT_FUSE_PARAMS, feather: 2 };

    const withBg = fuseDepth(global, [tile], S, S, opts);
    const subjectOnly = fuseDepth(global, [tile], S, S, { ...opts, subject: alpha });

    // 被写体だけで合わせれば、正解の 1/2 が出る
    expect(subjectOnly.fits[0]!.a).toBeCloseTo(0.5, 2);
    // 背景こみだと正解から外れる
    expect(Math.abs(withBg.fits[0]!.a - 0.5)).toBeGreaterThan(
      Math.abs(subjectOnly.fits[0]!.a - 0.5),
    );
    // 残差も被写体だけのほうが小さい
    expect(subjectOnly.fits[0]!.rmse).toBeLessThan(withBg.fits[0]!.rmse);
  });

  it('タイルのスケールを全体に合わせる', () => {
    const { global, tile } = scenario();
    const { fits } = fuseDepth(global, [tile], S, S, { ...DEFAULT_FUSE_PARAMS, feather: 4 });
    expect(fits).toHaveLength(1);
    // タイルが 3 倍なので、合わせる係数は 1/3
    expect(fits[0]!.a).toBeCloseTo(1 / 3, 3);
  });

  it('継ぎ目に段差を残さない', () => {
    const { global, tile } = scenario();
    const { depth } = fuseDepth(global, [tile], S, S, { ...DEFAULT_FUSE_PARAMS, feather: 4 });

    // 中央の行に沿って、隣接画素の差の最大値を見る。
    // 元の傾斜は 1/64 ≒ 0.0156/px。段差があればこれを大きく超える。
    let maxJump = 0;
    const row = S / 2;
    for (let x = 1; x < S; x++) {
      maxJump = Math.max(maxJump, Math.abs((depth[row * S + x] as number) - (depth[row * S + x - 1] as number)));
    }
    expect(maxJump).toBeLessThan(0.05);
  });

  it('タイルが掛からない場所では全体パスをそのまま返す', () => {
    const { global, tile } = scenario();
    const { depth } = fuseDepth(global, [tile], S, S, { ...DEFAULT_FUSE_PARAMS, feather: 4 });
    // 左上の隅はタイルの外
    expect(depth[0]).toBeCloseTo(global[0] as number, 2);
  });

  it('合わせられないタイルは捨てる', () => {
    const global = ramp();
    const rect = { x: 16, y: 16, width: 4, height: 4 }; // 画素が少なすぎる
    const depth = new Float32Array(16).fill(50);
    const { depth: out, fits } = fuseDepth(global, [{ depth, rect }], S, S, {
      ...DEFAULT_FUSE_PARAMS,
      feather: 0,
    });
    expect(fits[0]!.count).toBeLessThan(64);
    // 捨てられているので全体パスのまま
    expect(out[18 * S + 18]).toBeCloseTo(global[18 * S + 18] as number, 2);
  });
});
