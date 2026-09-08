/**
 * ④⑤ シェル → スプラット（docs/03 §3.6）。
 *
 * ここは座標系の変換が集まる場所で、間違えても「それらしい数」が出てしまう。
 * なので幾何そのものを検算する。合成した半球を通し、復元した点が本当に
 * 半球の上に乗るか、法線が手前を向くかを見る。
 */
import { describe, expect, it } from 'vitest';
import { buildSplats, DEFAULT_BUILD_PARAMS } from '../../src/pipeline/6-splats';
import { estimateNormals } from '../../src/pipeline/3-calibrate';
import { adaptiveSample, SAMPLING_PRESETS } from '../../src/pipeline/7-sample';
import { SPLAT_BYTES } from '../../src/render/SplatRenderer';

const S = 64;
const FOCAL = 80;
const NEAR = 1.0;
const FAR = 1.5;

/** カメラ正面に置いた半球。手前に膨らんでいる。 */
function hemisphere() {
  const depth = new Float32Array(S * S);
  const color = new Uint8ClampedArray(S * S * 4);
  const alpha = new Uint8ClampedArray(S * S);
  const R = 22;
  for (let y = 0; y < S; y++) {
    for (let x = 0; x < S; x++) {
      const i = y * S + x;
      const dx = x - S / 2;
      const dy = y - S / 2;
      const r2 = dx * dx + dy * dy;
      if (r2 > R * R) continue;
      alpha[i] = 255;
      // 手前へ膨らむぶんだけ深度が小さくなる
      const bulge = Math.sqrt(1 - r2 / (R * R));
      depth[i] = 0.8 - 0.5 * bulge;
      const c = 120 + 80 * bulge;
      color[i * 4] = c;
      color[i * 4 + 1] = c * 0.8;
      color[i * 4 + 2] = c * 0.6;
      color[i * 4 + 3] = 255;
    }
  }
  return { depth, color, alpha };
}

function build(backShell: boolean) {
  const { depth, color, alpha } = hemisphere();
  // 法線の推定は実距離の深度に対して行う
  const metric = new Float32Array(S * S);
  for (let i = 0; i < metric.length; i++) metric[i] = NEAR + (depth[i] as number) * (FAR - NEAR);
  const normals = estimateNormals(metric, S, S, FOCAL, 0.1);
  const cells = adaptiveSample(depth, color, alpha, S, S, SAMPLING_PRESETS.high);
  return {
    result: buildSplats(
      { cells, normals, width: S, height: S, focalPx: FOCAL, nearZ: NEAR, farZ: FAR },
      color,
      alpha,
      null,
      null,
      { ...DEFAULT_BUILD_PARAMS, backShell },
    ),
    cells,
  };
}

/** 詰めたバッファから位置を読み出す。 */
function positions(data: Uint8Array, count: number): [number, number, number][] {
  const f = new Float32Array(data.buffer, data.byteOffset, (count * SPLAT_BYTES) / 4);
  const out: [number, number, number][] = [];
  for (let i = 0; i < count; i++) {
    const o = (i * SPLAT_BYTES) / 4;
    out.push([f[o] as number, f[o + 1] as number, f[o + 2] as number]);
  }
  return out;
}

/** 八面体符号化された法線を戻す。WGSL の decodeOct と同じ式。 */
function decodeOct(packed: number): [number, number, number] {
  let ex = ((packed & 0xffff) / 65535) * 2 - 1;
  let ey = ((packed >>> 16) / 65535) * 2 - 1;
  let nz = 1 - Math.abs(ex) - Math.abs(ey);
  if (nz < 0) {
    const t = -nz;
    const sx = ex >= 0 ? -t : t;
    const sy = ey >= 0 ? -t : t;
    ex += sx;
    ey += sy;
  }
  const len = Math.hypot(ex, ey, nz) || 1;
  return [ex / len, ey / len, nz / len];
}

function normals(data: Uint8Array, count: number): [number, number, number][] {
  const u = new Uint32Array(data.buffer, data.byteOffset, (count * SPLAT_BYTES) / 4);
  const out: [number, number, number][] = [];
  for (let i = 0; i < count; i++) out.push(decodeOct(u[(i * SPLAT_BYTES) / 4 + 3] as number));
  return out;
}

describe('前面シェル', () => {
  it('セル1個につきスプラット1個', () => {
    const { result, cells } = build(false);
    expect(result.frontCount).toBe(cells.cellCount);
    expect(result.data.length).toBe(result.count * SPLAT_BYTES);
  });

  it('滑らかな面にはスカートを立てない', () => {
    // 合成した半球は連続な曲面。輪郭付近では勾配が無限大に近づくが、
    // それは「急な斜面」であって不連続ではない。ここを段差と誤ると
    // 輪郭全周にスカートが立つ（勾配の跳ね上がりを見る前は 800 枚立った）。
    const { result } = build(false);
    expect(result.skirtCount, `スカートが ${result.skirtCount} 枚立っています`).toBe(0);
    expect(result.count).toBe(result.frontCount);
  });

  it('単位立方体に収まる', () => {
    const { result } = build(false);
    for (const p of positions(result.data, result.count)) {
      for (const v of p) {
        expect(Math.abs(v)).toBeLessThanOrEqual(0.5 + 1e-5);
      }
    }
  });

  it('法線が手前（+z）を向く', () => {
    // 描画側は +z にカメラがある。手前を向いていなければ背面カリングで全部消える。
    const { result } = build(false);
    const ns = normals(result.data, result.frontCount);
    let facing = 0;
    for (const n of ns) if (n[2] > 0) facing++;
    expect(facing / ns.length).toBeGreaterThan(0.98);
  });

  it('逆投影→再投影で元の画素と深度に戻る', () => {
    // 形を仮定した検算（「球面に乗るか」など）は当てにならない。合成した
    // 半球は深度方向の振幅と画面上の半径が一致しないので、実際には楕円体になる。
    // そこで形に依存しない検算をする: 作った点を元の式で投影し直し、
    // セル中心の (u, v) と深度に戻るかを見る。前面シェルはセル順に出るので
    // k 番目のスプラットは k 番目のセルに対応する。
    const { result, cells } = build(false);
    const pts = positions(result.data, result.frontCount);
    const { center, scale } = result.normalization;

    let maxUvError = 0;
    let maxZError = 0;
    for (let k = 0; k < result.frontCount; k++) {
      const p = pts[k] as [number, number, number];
      // 正規化を戻す
      const wx = p[0] / scale + center[0];
      const wy = p[1] / scale + center[1];
      const wz = p[2] / scale + center[2];
      // ワールド → パイプライン側カメラ空間（y と z を戻す）
      const camX = wx;
      const camY = -wy;
      const camZ = -wz;
      // 投影
      const u = (camX * FOCAL) / camZ + S / 2;
      const v = (camY * FOCAL) / camZ + S / 2;

      const expectedU = (cells.x[k] as number) + (cells.size[k] as number) / 2;
      const expectedV = (cells.y[k] as number) + (cells.size[k] as number) / 2;
      const expectedZ = NEAR + (cells.depth[k] as number) * (FAR - NEAR);

      maxUvError = Math.max(maxUvError, Math.abs(u - expectedU), Math.abs(v - expectedV));
      maxZError = Math.max(maxZError, Math.abs(camZ - expectedZ));
    }
    // float32 に詰めた後なので、丸めぶんの誤差だけを許す
    expect(maxUvError, `画素位置の最大誤差 ${maxUvError}`).toBeLessThan(0.01);
    expect(maxZError, `深度の最大誤差 ${maxZError}`).toBeLessThan(1e-4);
  });

  it('中央の点のほうが手前（z が大きい）', () => {
    const { result } = build(false);
    const pts = positions(result.data, result.frontCount);
    const center = pts.reduce((best, p) =>
      Math.hypot(p[0], p[1]) < Math.hypot(best[0], best[1]) ? p : best,
    );
    const edge = pts.reduce((best, p) =>
      Math.hypot(p[0], p[1]) > Math.hypot(best[0], best[1]) ? p : best,
    );
    expect(center[2]).toBeGreaterThan(edge[2]);
  });
});

describe('背面シェル', () => {
  it('前面より奥（z が小さい）に置かれる', () => {
    const thickness = new Float32Array(S * S);
    const { alpha } = hemisphere();
    for (let i = 0; i < thickness.length; i++) if ((alpha[i] as number) >= 128) thickness[i] = 0.2;

    const { depth, color } = hemisphere();
    const metric = new Float32Array(S * S);
    for (let i = 0; i < metric.length; i++) metric[i] = NEAR + (depth[i] as number) * (FAR - NEAR);
    const cells = adaptiveSample(depth, color, alpha, S, S, SAMPLING_PRESETS.high);
    const result = buildSplats(
      {
        cells,
        normals: estimateNormals(metric, S, S, FOCAL, 0.1),
        width: S,
        height: S,
        focalPx: FOCAL,
        nearZ: NEAR,
        farZ: FAR,
      },
      color,
      alpha,
      thickness,
      null,
    );

    expect(result.count).toBeGreaterThan(result.frontCount);
    const pts = positions(result.data, result.count);
    const front = pts.slice(0, result.frontCount);
    const back = pts.slice(result.frontCount);
    const meanZ = (a: [number, number, number][]) => a.reduce((s, p) => s + p[2], 0) / a.length;
    expect(meanZ(back)).toBeLessThan(meanZ(front));
  });
});

describe('スカート（docs/03 §3.6.3）', () => {
  /** 手前の板が奥の板を部分的に隠している場面。真ん中に深度の不連続がある。 */
  function twoSlabs() {
    const depth = new Float32Array(S * S);
    const color = new Uint8ClampedArray(S * S * 4);
    const alpha = new Uint8ClampedArray(S * S);
    for (let y = 8; y < S - 8; y++) {
      for (let x = 8; x < S - 8; x++) {
        const i = y * S + x;
        alpha[i] = 255;
        const near = x < S / 2;
        depth[i] = near ? 0.3 : 0.7; // 0.4 の段差
        const c = near ? 200 : 90;
        color[i * 4] = c;
        color[i * 4 + 1] = c;
        color[i * 4 + 2] = c;
        color[i * 4 + 3] = 255;
      }
    }
    return { depth, color, alpha };
  }

  function buildStep() {
    const { depth, color, alpha } = twoSlabs();
    const metric = new Float32Array(S * S);
    for (let i = 0; i < metric.length; i++) metric[i] = NEAR + (depth[i] as number) * (FAR - NEAR);
    const normals = estimateNormals(metric, S, S, FOCAL, 0.05);
    const cells = adaptiveSample(depth, color, alpha, S, S, SAMPLING_PRESETS.high);
    return buildSplats(
      { cells, normals, width: S, height: S, focalPx: FOCAL, nearZ: NEAR, farZ: FAR },
      color,
      alpha,
      null,
      null,
      { ...DEFAULT_BUILD_PARAMS, backShell: false },
    );
  }

  it('本当の段差にはスカートを立てる', () => {
    const r = buildStep();
    expect(r.skirtCount, 'スカートが1枚も立っていません').toBeGreaterThan(0);
  });

  it('スカートは段差の手前と奥の間に置かれる', () => {
    const r = buildStep();
    const pts = positions(r.data, r.count);
    const skirt = pts.slice(r.frontCount + r.backCount);
    expect(skirt.length).toBe(r.skirtCount);

    // 前面シェルの z の範囲（ワールド空間、+z が手前）
    const front = pts.slice(0, r.frontCount).map((p) => p[2]);
    const zMin = Math.min(...front);
    const zMax = Math.max(...front);
    for (const p of skirt) {
      expect(p[2]).toBeGreaterThanOrEqual(zMin - 1e-3);
      expect(p[2]).toBeLessThanOrEqual(zMax + 1e-3);
    }
    // 中間の深さにも置かれている（端に張り付いていない）
    const mid = skirt.filter((p) => p[2] > zMin + (zMax - zMin) * 0.25 && p[2] < zMax - (zMax - zMin) * 0.25);
    expect(mid.length, 'スカートが段差の中間に無い').toBeGreaterThan(0);
  });

  it('スカートは奥へ行くほど薄くなる', () => {
    const r = buildStep();
    const u = new Uint32Array(r.data.buffer, r.data.byteOffset, (r.count * SPLAT_BYTES) / 4);
    const pts = positions(r.data, r.count);
    const alphas: { z: number; a: number }[] = [];
    for (let i = r.frontCount + r.backCount; i < r.count; i++) {
      const rgba = u[(i * SPLAT_BYTES) / 4 + 5] as number;
      alphas.push({ z: (pts[i] as [number, number, number])[2], a: (rgba >>> 24) & 0xff });
    }
    // z が小さい（奥）ほど α が小さいこと。相関で見る。
    const n = alphas.length;
    const mz = alphas.reduce((s, v) => s + v.z, 0) / n;
    const ma = alphas.reduce((s, v) => s + v.a, 0) / n;
    let cov = 0;
    for (const v of alphas) cov += (v.z - mz) * (v.a - ma);
    expect(cov, '奥ほど薄い、になっていません').toBeGreaterThan(0);
  });

  it('スカートを切れば枚数はゼロ', () => {
    const { depth, color, alpha } = twoSlabs();
    const metric = new Float32Array(S * S);
    for (let i = 0; i < metric.length; i++) metric[i] = NEAR + (depth[i] as number) * (FAR - NEAR);
    const cells = adaptiveSample(depth, color, alpha, S, S, SAMPLING_PRESETS.high);
    const r = buildSplats(
      {
        cells,
        normals: estimateNormals(metric, S, S, FOCAL, 0.05),
        width: S,
        height: S,
        focalPx: FOCAL,
        nearZ: NEAR,
        farZ: FAR,
      },
      color,
      alpha,
      null,
      null,
      { ...DEFAULT_BUILD_PARAMS, backShell: false, skirt: false },
    );
    expect(r.skirtCount).toBe(0);
  });
});

describe('スカートの閾値は深度の分布から決める（v2.3、実写で判明）', () => {
  /**
   * なだらかな起伏だけの面。段差はどこにも無い。
   *
   * `amplitude` を上げると、較正の仕方で深度の勾配が大きくなった状況に
   * なる。固定閾値（0.02）だと、ここで「不連続」の判定が誤爆し、
   * 面じゅうにスカートが立ってしまう。実写では全スプラットの 63% が
   * スカートになり、顔に縦の帯が走った。
   */
  function bumpy(amplitude: number) {
    const depth = new Float32Array(S * S);
    const color = new Uint8ClampedArray(S * S * 4);
    const alpha = new Uint8ClampedArray(S * S);
    for (let y = 6; y < S - 6; y++) {
      for (let x = 6; x < S - 6; x++) {
        const i = y * S + x;
        alpha[i] = 255;
        depth[i] =
          0.5 + amplitude * Math.sin((x / 5) * Math.PI * 2) * Math.cos((y / 5) * Math.PI * 2);
        color[i * 4] = 150;
        color[i * 4 + 1] = 150;
        color[i * 4 + 2] = 150;
        color[i * 4 + 3] = 255;
      }
    }
    return { depth, color, alpha };
  }

  function build(depth: Float32Array, color: Uint8ClampedArray, alpha: Uint8ClampedArray) {
    const metric = new Float32Array(S * S);
    for (let i = 0; i < metric.length; i++) metric[i] = NEAR + (depth[i] as number) * (FAR - NEAR);
    const normals = estimateNormals(metric, S, S, FOCAL, 0.05);
    const cells = adaptiveSample(depth, color, alpha, S, S, SAMPLING_PRESETS.high);
    return buildSplats(
      { cells, normals, width: S, height: S, focalPx: FOCAL, nearZ: NEAR, farZ: FAR },
      color,
      alpha,
      null,
      null,
      { ...DEFAULT_BUILD_PARAMS, backShell: false },
    );
  }

  it('勾配が大きくなってもスカートが暴走しない', () => {
    // 振幅を 4 倍にしても、スカートの割合は数倍以内に収まること。
    // 固定閾値のときは 21% → 63% と桁で動いていた。
    const small = bumpy(0.02);
    const large = bumpy(0.08);
    const rs = build(small.depth, small.color, small.alpha);
    const rl = build(large.depth, large.color, large.alpha);
    const ratio = (r: { skirtCount: number; count: number }): number => r.skirtCount / r.count;
    expect(ratio(rl)).toBeLessThan(0.25);
    expect(ratio(rl)).toBeLessThan(Math.max(ratio(rs), 0.02) * 6);
  });

  it('分位を上げるほどスカートは減る', () => {
    const { depth, color, alpha } = bumpy(0.06);
    const metric = new Float32Array(S * S);
    for (let i = 0; i < metric.length; i++) metric[i] = NEAR + (depth[i] as number) * (FAR - NEAR);
    const normals = estimateNormals(metric, S, S, FOCAL, 0.05);
    const cells = adaptiveSample(depth, color, alpha, S, S, SAMPLING_PRESETS.high);
    const at = (p: number): number =>
      buildSplats(
        { cells, normals, width: S, height: S, focalPx: FOCAL, nearZ: NEAR, farZ: FAR },
        color,
        alpha,
        null,
        null,
        { ...DEFAULT_BUILD_PARAMS, backShell: false, skirtGapPercentile: p },
      ).skirtCount;
    expect(at(0.99)).toBeLessThanOrEqual(at(0.9));
  });

  it('本当の段差なら分位を上げてもスカートは残る', () => {
    // 段差が「上位数%の外れ値」として立っている場面なら、分位で切っても
    // 残る。分位は小さすぎる段差を捨てる床であって、段差そのものの
    // 判定は skirtStepRatio が受け持つ。
    const depth = new Float32Array(S * S);
    const color = new Uint8ClampedArray(S * S * 4);
    const alpha = new Uint8ClampedArray(S * S);
    for (let y = 8; y < S - 8; y++) {
      for (let x = 8; x < S - 8; x++) {
        const i = y * S + x;
        alpha[i] = 255;
        depth[i] = x < S / 2 ? 0.3 : 0.7;
        const c = x < S / 2 ? 200 : 90;
        color[i * 4] = c;
        color[i * 4 + 1] = c;
        color[i * 4 + 2] = c;
        color[i * 4 + 3] = 255;
      }
    }
    expect(build(depth, color, alpha).skirtCount).toBeGreaterThan(0);
  });
});
