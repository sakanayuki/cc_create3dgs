/**
 * 四肢に円柱の断面を与える（docs/09 §V20）。
 *
 * 深度モデルは腕や脚の横断面をほとんど平らに返す。シルエットの横一線の
 * 区間から円柱を起こして、大域だけ差し替える。
 */
import { describe, expect, it } from 'vitest';
import { applyLimbRoundness, DEFAULT_LIMB_PARAMS } from '../../src/pipeline/geometry/limbRoundness';

const W = 200;
const H = 120;
const FOCAL = 500;
const Z = 1.0;

/**
 * 幅 `legW` の縦棒を 2 本と、幅 `bodyW` の胴を置く。深度は平ら。
 * 実写と同じで、深度モデルは断面の丸みを出さない。
 */
function scene(legW = 24, bodyW = 120): { depth: Float32Array; alpha: Uint8ClampedArray } {
  const depth = new Float32Array(W * H).fill(Z);
  const alpha = new Uint8ClampedArray(W * H);
  const put = (y0: number, y1: number, x0: number, x1: number): void => {
    for (let y = y0; y < y1; y++) for (let x = x0; x < x1; x++) alpha[y * W + x] = 255;
  };
  put(4, 40, 100 - bodyW / 2, 100 + bodyW / 2); // 胴
  put(44, 116, 70, 70 + legW); // 左脚
  put(44, 116, 130 - legW, 130); // 右脚
  return { depth, alpha };
}

/** 区間 [x0,x1) の中央 1/3 と、外側 18% の平均の差。正なら端が奥（＝円柱）。 */
function edgeMinusCenter(depth: ArrayLike<number>, y: number, x0: number, x1: number): number {
  const n = x1 - x0;
  const cut = Math.max(1, Math.round(n * 0.18));
  let edge = 0;
  let ne = 0;
  let mid = 0;
  let nm = 0;
  for (let i = 0; i < n; i++) {
    const v = depth[y * W + x0 + i] as number;
    if (i < cut || i >= n - cut) {
      edge += v;
      ne++;
    } else if (i >= n / 3 && i < (2 * n) / 3) {
      mid += v;
      nm++;
    }
  }
  return edge / ne - mid / nm;
}

describe('四肢の断面を円柱にする', () => {
  it('平らな脚に丸みが入る', () => {
    const { depth, alpha } = scene();
    const before = edgeMinusCenter(depth, 80, 70, 94);
    expect(before, '入力は平らなはず').toBeCloseTo(0, 6);

    const out = applyLimbRoundness(depth, alpha, W, H, FOCAL);
    const after = edgeMinusCenter(out, 80, 70, 94);
    // 真円の柱なら「端の外側 18% と中央 1/3 の差 ÷ 幅」は 0.29 前後になる。
    // 幅 24px、距離 1、焦点 500 → 1画素 = 0.002。24 × 0.002 × 0.29 = 0.0139。
    const ratio = after / (24 * (Z / FOCAL));
    expect(after, '端が奥へ引いていません').toBeGreaterThan(0);
    expect(ratio, `端−中央 ÷ 幅 = ${ratio.toFixed(3)}`).toBeGreaterThan(0.2);
    expect(ratio, `端−中央 ÷ 幅 = ${ratio.toFixed(3)}`).toBeLessThan(0.4);
  });

  it('胴は太いので触らない', () => {
    const { depth, alpha } = scene();
    const out = applyLimbRoundness(depth, alpha, W, H, FOCAL);
    for (let x = 45; x < 155; x++) {
      expect(out[20 * W + x], `胴の x=${x} が動いています`).toBeCloseTo(Z, 6);
    }
  });

  it('区間の平均の深度は動かない（脚が空間を移動しない）', () => {
    const { depth, alpha } = scene();
    const out = applyLimbRoundness(depth, alpha, W, H, FOCAL);
    let sum = 0;
    for (let x = 70; x < 94; x++) sum += out[80 * W + x] as number;
    expect(sum / 24, '脚の平均深度が動いています').toBeCloseTo(Z, 5);
  });

  it('被写体の外は動かさない', () => {
    const { depth, alpha } = scene();
    const out = applyLimbRoundness(depth, alpha, W, H, FOCAL);
    for (let x = 0; x < W; x += 7) {
      if ((alpha[80 * W + x] as number) >= 128) continue;
      expect(out[80 * W + x]).toBe(depth[80 * W + x]);
    }
  });

  it('細部は残る（大域だけ差し替える）', () => {
    const { depth, alpha } = scene();
    // 脚の真ん中に細い溝を掘る
    for (let y = 44; y < 116; y++) depth[y * W + 82] = Z + 0.01;
    const out = applyLimbRoundness(depth, alpha, W, H, FOCAL);
    const groove = (out[80 * W + 82] as number) - (out[80 * W + 80] as number);
    expect(groove, `溝が ${groove.toFixed(5)} しか残っていません`).toBeGreaterThan(0.006);
  });

  it('外す四角の中は触らない（頭）', () => {
    // 頭は胴より細いので、そのままだと「四肢」の網に掛かる。実写でも
    // 顔の 31〜56% の画素が四肢と判定されていた。
    const { depth, alpha } = scene();
    // 胴の上を空けて、そこに頭（幅 24）を置く。胴とつながると 1 区間に
    // 溶けてしまい、頭が細いことを試せない。
    for (let y = 0; y < 18; y++) for (let x = 0; x < W; x++) alpha[y * W + x] = 0;
    for (let y = 0; y < 14; y++) for (let x = 88; x < 112; x++) alpha[y * W + x] = 255;

    const without = applyLimbRoundness(depth, alpha, W, H, FOCAL);
    expect(edgeMinusCenter(without, 10, 88, 112), '頭が四肢扱いされていません').toBeGreaterThan(0);

    const head = { x: 80, y: 0, width: 40, height: 16 };
    const out = applyLimbRoundness(depth, alpha, W, H, FOCAL, DEFAULT_LIMB_PARAMS, head);
    for (let x = 88; x < 112; x++) {
      expect(out[10 * W + x], `外した四角の x=${x} が動いています`).toBeCloseTo(Z, 6);
    }
    // 四角の外（脚）は効いている
    expect(edgeMinusCenter(out, 80, 70, 94)).toBeGreaterThan(0);
  });

  it('強さ 0 なら何もしない', () => {
    const { depth, alpha } = scene();
    const out = applyLimbRoundness(depth, alpha, W, H, FOCAL, {
      ...DEFAULT_LIMB_PARAMS,
      strength: 0,
    });
    for (let i = 0; i < depth.length; i += 13) expect(out[i]).toBe(depth[i]);
  });

  it('短すぎる区間は触らない', () => {
    const depth = new Float32Array(W * H).fill(Z);
    const alpha = new Uint8ClampedArray(W * H);
    for (let y = 40; y < 80; y++) for (let x = 100; x < 106; x++) alpha[y * W + x] = 255; // 幅 6px
    const out = applyLimbRoundness(depth, alpha, W, H, FOCAL);
    for (let x = 100; x < 106; x++) expect(out[60 * W + x]).toBe(Z);
  });
});
