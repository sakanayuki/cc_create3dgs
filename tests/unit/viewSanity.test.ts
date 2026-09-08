/**
 * 「少し斜めから見ても人物として破綻しないか」を自動で確かめる。
 *
 * 正面から見るぶんには、前面シェルは写真そのものなので破綻が見えない。
 * 実写で見つかった壊れ方は、いずれも**回して初めて出る**ものだった。
 *
 *   ・輪郭に沿って最奥に張り付いたサーフェルの膜（体の後ろへ伸びる板）
 *   ・被写体に紛れ込んだ背景（平らな板として突き出す）
 *   ・体が断片に割れる
 *
 * ここでは合成の被写体でその仕組みを再現し、指標で捕まえる。
 */
import { describe, expect, it } from 'vitest';
import { estimateNormals, pullBoundaryDepthInward } from '../../src/pipeline/3-calibrate';
import { thicknessMap } from '../../src/pipeline/4-shell';
import { buildSplats, DEFAULT_BUILD_PARAMS } from '../../src/pipeline/6-splats';
import { adaptiveSample, SAMPLING_PRESETS } from '../../src/pipeline/7-sample';
import { decodeSplats, viewMetrics } from '../helpers/splatView';

const S = 160;
const FOCAL = 200;
const NEAR = 0.8;
const FAR = 1.2;

/**
 * 縦長のカプセル（人物の代わり）。奥行きは滑らかで、段差はどこにも無い。
 *
 * @param rimBleed 輪郭から内側 `rimBand` 画素の深度を、最奥へ引きずる量（0〜1）。
 *                 単眼深度モデルが縁で前景と背景を混ぜる現象を模す。
 */
function capsule(rimBleed = 0, rimBand = 6) {
  const depth = new Float32Array(S * S);
  const color = new Uint8ClampedArray(S * S * 4);
  const alpha = new Uint8ClampedArray(S * S);
  const cx = S / 2;
  const rx = S * 0.22;
  const y0 = S * 0.12;
  const y1 = S * 0.88;

  for (let y = 0; y < S; y++) {
    for (let x = 0; x < S; x++) {
      const i = y * S + x;
      const t = Math.min(1, Math.max(0, (y - y0) / (y1 - y0)));
      // 上下で少し細くなる紡錘形
      const w = rx * (0.55 + 0.45 * Math.sin(Math.PI * t));
      const dx = (x - cx) / w;
      if (y < y0 || y > y1 || Math.abs(dx) > 1) continue;
      alpha[i] = 255;
      // 円柱の表面。中心が手前、縁が奥。
      const bulge = Math.sqrt(Math.max(0, 1 - dx * dx));
      depth[i] = 0.65 - 0.3 * bulge;
      const c = Math.round(120 + 100 * bulge);
      color[i * 4] = c;
      color[i * 4 + 1] = c;
      color[i * 4 + 2] = c;
      color[i * 4 + 3] = 255;
    }
  }

  if (rimBleed > 0) {
    const src = Float32Array.from(depth);
    for (let y = 0; y < S; y++) {
      for (let x = 0; x < S; x++) {
        const i = y * S + x;
        if ((alpha[i] as number) < 128) continue;
        // 左右方向のシルエットまでの距離
        let d = 0;
        while (d < rimBand) {
          const l = x - d - 1;
          const r = x + d + 1;
          const outL = l < 0 || (alpha[y * S + l] as number) < 128;
          const outR = r >= S || (alpha[y * S + r] as number) < 128;
          if (outL || outR) break;
          d++;
        }
        if (d >= rimBand) continue;
        const k = rimBleed * (1 - d / rimBand);
        depth[i] = (src[i] as number) * (1 - k) + 1.0 * k; // 1.0 = 最奥
      }
    }
  }
  return { depth, color, alpha };
}

function build(depth: Float32Array, color: Uint8ClampedArray, alpha: Uint8ClampedArray) {
  const metric = new Float32Array(S * S);
  for (let i = 0; i < metric.length; i++) metric[i] = NEAR + (depth[i] as number) * (FAR - NEAR);
  const normals = estimateNormals(metric, S, S, FOCAL, 0.05);
  const cells = adaptiveSample(depth, color, alpha, S, S, SAMPLING_PRESETS.high);
  const thickness = thicknessMap(alpha, S, S, { maxThickness: 0.35, profile: 'ellipsoid' });
  return buildSplats(
    { cells, normals, width: S, height: S, focalPx: FOCAL, nearZ: NEAR, farZ: FAR },
    color,
    alpha,
    thickness,
    null,
    { ...DEFAULT_BUILD_PARAMS },
  );
}

/** capsule() の戻りを build() の引数順に並べ替える。 */
function capsuleArgs(
  c: ReturnType<typeof capsule>,
): [Float32Array, Uint8ClampedArray, Uint8ClampedArray] {
  return [c.depth, c.color, c.alpha];
}

describe('斜めから見たときの健全性（v2.3、実写での破綻にもとづく）', () => {
  it('正面でも斜めでも、ひとつながりの塊であること', () => {
    const { depth, color, alpha } = capsule();
    const b = build(depth, color, alpha);
    for (const yawDeg of [0, 15, 30, 45]) {
      const m = viewMetrics(b.data, b.count, { yawDeg });
      expect(m.covered, `yaw ${yawDeg} で何も描かれていません`).toBeGreaterThan(1000);
      expect(m.mainRatio, `yaw ${yawDeg} で体が断片に割れています`).toBeGreaterThan(0.97);
    }
  });

  it('斜めにしてもシルエットの内側が穴だらけにならない', () => {
    const { depth, color, alpha } = capsule();
    const b = build(depth, color, alpha);
    for (const yawDeg of [0, 15, 30]) {
      const m = viewMetrics(b.data, b.count, { yawDeg });
      expect(m.holeRatio, `yaw ${yawDeg} の穴率 ${(m.holeRatio * 100).toFixed(1)}%`).toBeLessThan(0.1);
    }
  });

  it('輪郭のにじみを引き込むと、回したときに膜が出てこない', () => {
    // にじみを放置すると、輪郭に沿って最奥に張り付いたサーフェルの膜が
    // でき、斜めから見たときに体の後ろへ板として伸びる。実写では被写体の
    // 6.9% がこれで、シルエットから 1px の画素は 56% が最奥に張り付いて
    // いた。引き込むと 7.2% まで落ちた。
    //
    // 見分け方: **凸な体は回すと見かけの面積が少し減る**。膜があると
    // 逆に増える（膜が横を向いて広がるため）。
    const bleed = capsule(0.9, 6);
    const raw = build(bleed.depth, bleed.color, bleed.alpha);

    const u16 = new Uint16Array(S * S);
    for (let i = 0; i < u16.length; i++) {
      u16[i] = Math.round(Math.max(0, Math.min(1, bleed.depth[i] as number)) * 65535);
    }
    const pulled = pullBoundaryDepthInward(u16, bleed.alpha, S, S, 6);
    const fixedDepth = new Float32Array(S * S);
    for (let i = 0; i < fixedDepth.length; i++) fixedDepth[i] = (pulled[i] as number) / 65535;
    const fixed = build(fixedDepth, bleed.color, bleed.alpha);

    const growth = (b: { data: Uint8Array; count: number }): number =>
      viewMetrics(b.data, b.count, { yawDeg: 40 }).covered /
      viewMetrics(b.data, b.count, { yawDeg: 0 }).covered;

    // にじみを残すと、回したほうが大きく見える（実測 1.15）
    expect(growth(raw)).toBeGreaterThan(1.05);
    // 引き込むと、凸な体らしく少し縮む（実測 0.89）
    expect(growth(fixed)).toBeLessThan(1.0);
  });

  it('輪郭のにじみを引き込むと、点群の奥行き幅が実際の形に戻る', () => {
    const clean = build(...capsuleArgs(capsule()));
    const bleed = capsule(0.9, 6);
    const raw = build(bleed.depth, bleed.color, bleed.alpha);
    const u16 = new Uint16Array(S * S);
    for (let i = 0; i < u16.length; i++) {
      u16[i] = Math.round(Math.max(0, Math.min(1, bleed.depth[i] as number)) * 65535);
    }
    const pulled = pullBoundaryDepthInward(u16, bleed.alpha, S, S, 6);
    const fixedDepth = new Float32Array(S * S);
    for (let i = 0; i < fixedDepth.length; i++) fixedDepth[i] = (pulled[i] as number) / 65535;
    const fixed = build(fixedDepth, bleed.color, bleed.alpha);

    const extent = (b: { data: Uint8Array; count: number }): number => {
      const { z } = decodeSplats(b.data, b.count);
      const zs = Array.from(z).sort((a, c) => a - c);
      const q = (f: number): number => zs[Math.floor(f * (zs.length - 1))] as number;
      return q(0.99) - q(0.01);
    };
    const base = extent(clean);
    // 膜は奥行きを 1.6 倍に膨らませる（実測 0.224 → 0.358）
    expect(extent(raw)).toBeGreaterThan(base * 1.3);
    // 引き込むと本来の形へ戻る（実測 0.234）
    expect(extent(fixed)).toBeLessThan(base * 1.15);
  });

  it('引き込みは輪郭だけに効き、内部の形は変えない', () => {
    const { depth, alpha } = capsule();
    const u16 = new Uint16Array(S * S);
    for (let i = 0; i < u16.length; i++) {
      u16[i] = Math.round(Math.max(0, Math.min(1, depth[i] as number)) * 65535);
    }
    const pulled = pullBoundaryDepthInward(u16, alpha, S, S, 6);
    // 中央の縦線（シルエットから十分内側）は変わらない
    let touched = 0;
    for (let y = 30; y < S - 30; y++) {
      const i = y * S + S / 2;
      if ((alpha[i] as number) < 128) continue;
      if (pulled[i] !== u16[i]) touched++;
    }
    expect(touched).toBe(0);
  });

  it('引き込む先が無いほど細い被写体では何もしない', () => {
    // 幅 3px の棒。芯が残らないので、触ると全部消えてしまう。
    const depth = new Float32Array(S * S);
    const alpha = new Uint8ClampedArray(S * S);
    for (let y = 20; y < S - 20; y++) {
      for (let x = 80; x < 83; x++) {
        alpha[y * S + x] = 255;
        depth[y * S + x] = 0.5;
      }
    }
    const u16 = new Uint16Array(S * S);
    for (let i = 0; i < u16.length; i++) u16[i] = Math.round((depth[i] as number) * 65535);
    const pulled = pullBoundaryDepthInward(u16, alpha, S, S, 6);
    for (let i = 0; i < u16.length; i++) expect(pulled[i]).toBe(u16[i]);
  });
});
