/**
 * 組み立てたスプラットを実際に「見て」確かめるための最小の描画器。
 *
 * 単一の写真から作る 3DGS は、正面から見るぶんには写真そのものなので
 * 破綻が見えない。**少し回したときに初めて出る**壊れ方がある（輪郭に
 * 沿った膜、層に分かれる、体が断片に割れる）。ここはその壊れ方を
 * 数値で捕まえるための道具で、見た目の美しさは測らない。
 *
 * α はほぼ 1 なので、最近接のサーフェルがそのまま見える。z バッファで
 * 十分に近似できる。
 */
import { unpackHalf2 } from '../../src/codec/pack';
import { SPLAT_BYTES } from '../../src/render/SplatRenderer';

export interface ViewMetrics {
  /** 何かが描かれた画素数。 */
  readonly covered: number;
  /** 大きさ `minFragment` 以上の連結成分の数。1 が正常。 */
  readonly fragments: number;
  /** 最大の連結成分が占める割合。1 に近いほど、ひとつながり。 */
  readonly mainRatio: number;
  /** シルエットの内側にできた穴の割合。 */
  readonly holeRatio: number;
}

export interface ViewOptions {
  readonly width?: number;
  readonly height?: number;
  /** Y 軸まわりの回転（度）。 */
  readonly yawDeg?: number;
  readonly distance?: number;
  readonly fovDeg?: number;
  /** これより小さい連結成分は数えない（点の飛びを断片と数えないため）。 */
  readonly minFragment?: number;
}

interface Decoded {
  readonly x: Float32Array;
  readonly y: Float32Array;
  readonly z: Float32Array;
  readonly r: Float32Array;
}

export function decodeSplats(data: Uint8Array, count: number): Decoded {
  const dv = new DataView(data.buffer, data.byteOffset, data.byteLength);
  const x = new Float32Array(count);
  const y = new Float32Array(count);
  const z = new Float32Array(count);
  const r = new Float32Array(count);
  for (let i = 0; i < count; i++) {
    const o = i * SPLAT_BYTES;
    x[i] = dv.getFloat32(o, true);
    y[i] = dv.getFloat32(o + 4, true);
    z[i] = dv.getFloat32(o + 8, true);
    r[i] = unpackHalf2(dv.getUint32(o + 16, true))[0];
  }
  return { x, y, z, r };
}

/** 指定した角度から描いて、被覆マスクを返す。 */
export function renderCoverage(
  data: Uint8Array,
  count: number,
  opts: ViewOptions = {},
): { mask: Uint8Array; width: number; height: number } {
  const W = opts.width ?? 200;
  const H = opts.height ?? 280;
  const dist = opts.distance ?? 1.6;
  const fov = ((opts.fovDeg ?? 42) * Math.PI) / 180;
  const yaw = ((opts.yawDeg ?? 0) * Math.PI) / 180;
  const cy = Math.cos(yaw);
  const sy = Math.sin(yaw);
  const f = H / 2 / Math.tan(fov / 2);
  const { x, y, z, r } = decodeSplats(data, count);
  const mask = new Uint8Array(W * H);

  for (let i = 0; i < count; i++) {
    const xr = (x[i] as number) * cy + (z[i] as number) * sy;
    const zr = -(x[i] as number) * sy + (z[i] as number) * cy;
    // ワールド z は大きいほど手前（6-splats の toWorld が z を反転している）。
    const zc = dist - zr;
    if (!(zc > 0.05)) continue;
    const u = (-xr * f) / zc + W / 2;
    const v = (-(y[i] as number) * f) / zc + H / 2;
    const rad = Math.max(0.5, ((r[i] as number) * f) / zc);
    const k = Math.min(3, Math.max(0, Math.round(rad)));
    const iu = Math.round(u);
    const iv = Math.round(v);
    for (let dy = -k; dy <= k; dy++) {
      for (let dx = -k; dx <= k; dx++) {
        if (dx * dx + dy * dy > k * k) continue;
        const px = iu + dx;
        const py = iv + dy;
        if (px < 0 || py < 0 || px >= W || py >= H) continue;
        mask[py * W + px] = 1;
      }
    }
  }
  return { mask, width: W, height: H };
}

function componentSizes(mask: Uint8Array, W: number, H: number, min: number): number[] {
  const seen = new Uint8Array(mask.length);
  const out: number[] = [];
  const stack: number[] = [];
  for (let s = 0; s < mask.length; s++) {
    if (!mask[s] || seen[s]) continue;
    stack.length = 0;
    stack.push(s);
    seen[s] = 1;
    let n = 0;
    while (stack.length > 0) {
      const i = stack.pop() as number;
      n++;
      const px = i % W;
      const py = (i / W) | 0;
      if (px > 0 && mask[i - 1] && !seen[i - 1]) { seen[i - 1] = 1; stack.push(i - 1); }
      if (px < W - 1 && mask[i + 1] && !seen[i + 1]) { seen[i + 1] = 1; stack.push(i + 1); }
      if (py > 0 && mask[i - W] && !seen[i - W]) { seen[i - W] = 1; stack.push(i - W); }
      if (py < H - 1 && mask[i + W] && !seen[i + W]) { seen[i + W] = 1; stack.push(i + W); }
    }
    if (n >= min) out.push(n);
  }
  return out.sort((a, b) => b - a);
}

/** 外周からたどり着けない背景＝シルエットの内側の穴。 */
function holeCount(mask: Uint8Array, W: number, H: number): number {
  const seen = new Uint8Array(mask.length);
  const stack: number[] = [];
  const push = (i: number): void => {
    if (!mask[i] && !seen[i]) { seen[i] = 1; stack.push(i); }
  };
  for (let x = 0; x < W; x++) { push(x); push((H - 1) * W + x); }
  for (let y = 0; y < H; y++) { push(y * W); push(y * W + W - 1); }
  while (stack.length > 0) {
    const i = stack.pop() as number;
    const px = i % W;
    const py = (i / W) | 0;
    if (px > 0) push(i - 1);
    if (px < W - 1) push(i + 1);
    if (py > 0) push(i - W);
    if (py < H - 1) push(i + W);
  }
  let holes = 0;
  for (let i = 0; i < mask.length; i++) if (!mask[i] && !seen[i]) holes++;
  return holes;
}

export function viewMetrics(data: Uint8Array, count: number, opts: ViewOptions = {}): ViewMetrics {
  const { mask, width, height } = renderCoverage(data, count, opts);
  let covered = 0;
  for (let i = 0; i < mask.length; i++) covered += mask[i] as number;
  const comps = componentSizes(mask, width, height, opts.minFragment ?? 40);
  const main = comps[0] ?? 0;
  return {
    covered,
    fragments: comps.length,
    mainRatio: covered > 0 ? main / covered : 0,
    holeRatio: main > 0 ? holeCount(mask, width, height) / main : 0,
  };
}
