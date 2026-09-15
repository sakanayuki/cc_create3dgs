/**
 * 「人体としてあり得る形か」を数で見る（docs/11 §11.5、docs/13 §13.3）。
 *
 * これまで合格ラインの数字は調査用ハーネスでしか測っていなかった。判定に使う
 * 以上は検査に載せる必要がある。ここに置くのは**純粋な計算だけ**で、
 * スプラットの配列を受けて数を返す。
 *
 * 物差しは 2 つ。どちらも**正面から見た前面の深度**を行ごとに切って測り、
 * **体幅で割る**ので、写真の尺度にも切り取り方にも依らない。
 *
 * - **飛び出し**（`crossSectionBulge`）: 行の「体の形」（区間幅 1/4 の箱平均）
 *   より**手前**へ外れた量。服の縁の行き過ぎ（docs/09 §V21）を捕まえる。
 * - **凹み**（`crossSectionDent`）: 行の手前側の**凸包**より**奥**へ外れた量。
 *   服の開口部の落ち込み（docs/09 §V24）を捕まえる。
 */
import { encodeOct, packHalf2, packRgba8 } from '../../src/codec/pack';
import { SPLAT_BYTES } from '../../src/render/SplatRenderer';
import { decodeSplats } from './splatView';

export interface FrontMap {
  /** 各画素で最も手前の z。被写体が無い所は NaN。 */
  readonly z: Float32Array;
  readonly width: number;
  readonly height: number;
  /**
   * 体幅（**ワールド単位**）。割り算の分母に使う。
   *
   * z はワールド単位のまま入っているので、分母も同じ単位でないと比にならない。
   * 地図の画素数で割ると桁が合わない。
   */
  readonly bodyWidth: number;
}

/**
 * 正面から見た前面の深度地図を作る。
 *
 * z は**大きいほど奥**（`6-splats.ts` の `toWorld` と同じ向き）。
 *
 * @param side 長辺をこの画素数にする。
 */
export function frontDepthMap(data: Uint8Array, count: number, side = 384): FrontMap {
  const { x, y, z } = decodeSplats(data, count);
  const pct = (a: Float32Array, f: number): number => {
    const s = Float32Array.from(a).sort();
    return s[Math.min(s.length - 1, Math.max(0, Math.round(f * (s.length - 1))))] as number;
  };
  const x0 = pct(x, 0.003);
  const x1 = pct(x, 0.997);
  const y0 = pct(y, 0.003);
  const y1 = pct(y, 0.997);
  const scale = side / Math.max(x1 - x0, y1 - y0, 1e-9);
  const width = Math.max(8, Math.round((x1 - x0) * scale) + 1);
  const height = Math.max(8, Math.round((y1 - y0) * scale) + 1);

  const map = new Float32Array(width * height).fill(Number.NaN);
  for (let i = 0; i < count; i++) {
    const ix = Math.min(width - 1, Math.max(0, Math.round(((x[i] as number) - x0) * scale)));
    const iy = Math.min(height - 1, Math.max(0, Math.round(((y[i] as number) - y0) * scale)));
    const j = iy * width + ix;
    const here = map[j] as number;
    const v = z[i] as number;
    if (Number.isNaN(here) || v < here) map[j] = v;
  }
  return { z: map, width, height, bodyWidth: pct(x, 0.99) - pct(x, 0.01) };
}

/** 行の中の、途切れていない区間を返す。 */
function runs(map: FrontMap, row: number, minRun: number): [number, number][] {
  const out: [number, number][] = [];
  let start = -1;
  for (let x = 0; x <= map.width; x++) {
    const on = x < map.width && !Number.isNaN(map.z[row * map.width + x] as number);
    if (on && start < 0) start = x;
    if (!on && start >= 0) {
      if (x - start >= minRun) out.push([start, x - 1]);
      start = -1;
    }
  }
  return out;
}

export interface Percentiles {
  readonly p50: number;
  readonly p90: number;
  readonly p99: number;
  readonly max: number;
  readonly samples: number;
}

function summarise(values: number[]): Percentiles {
  if (values.length === 0) return { p50: 0, p90: 0, p99: 0, max: 0, samples: 0 };
  const s = values.slice().sort((a, b) => a - b);
  const at = (f: number): number => s[Math.min(s.length - 1, Math.floor(f * s.length))] as number;
  return { p50: at(0.5), p90: at(0.9), p99: at(0.99), max: s[s.length - 1] as number, samples: s.length };
}

/**
 * 横断面の「手前への飛び出し」（体幅比）。
 *
 * 行ごとに、区間幅の 1/4 の箱平均を「体の形」とみなし、そこから手前へ外れた
 * 量を体幅で割る。人の皺は体幅の数 % を超えないので、**99 パーセンタイルが
 * 6% を超えたら異常**とみなす（参照実装は 4.15%）。
 */
export function crossSectionBulge(map: FrontMap, minRun = 24): Percentiles {
  const out: number[] = [];
  for (let row = 0; row < map.height; row++) {
    for (const [a, b] of runs(map, row, minRun)) {
      const n = b - a + 1;
      const win = Math.max(2, Math.floor(n / 8));
      for (let i = 0; i < n; i++) {
        let sum = 0;
        let cnt = 0;
        for (let k = -win; k <= win; k++) {
          const j = Math.min(n - 1, Math.max(0, i + k));
          sum += map.z[row * map.width + a + j] as number;
          cnt++;
        }
        const forward = sum / cnt - (map.z[row * map.width + a + i] as number);
        if (forward > 0) out.push(forward / map.bodyWidth);
      }
    }
  }
  return summarise(out);
}

/**
 * 横断面の「奥への凹み」（体幅比）。
 *
 * 行ごとに手前側の凸包（下側凸包）を作り、そこから奥へ外れた量を体幅で割る。
 * 服の開口部の中を深度モデルが大きく奥へ置く破綻（docs/09 §V24）を捕まえる。
 * 参照実装の実測は胸で中央 0.041・最大 0.114。
 */
export function crossSectionDent(map: FrontMap, minRun = 40): Percentiles {
  const out: number[] = [];
  const hull = new Float64Array(map.width);
  const stack = new Int32Array(map.width);
  for (let row = 0; row < map.height; row++) {
    for (const [a, b] of runs(map, row, minRun)) {
      let top = 0;
      for (let i = a; i <= b; i++) {
        const zi = map.z[row * map.width + i] as number;
        while (top >= 2) {
          const p = stack[top - 2] as number;
          const q = stack[top - 1] as number;
          const zp = map.z[row * map.width + p] as number;
          const zq = map.z[row * map.width + q] as number;
          if ((zq - zp) * (i - p) >= (zi - zp) * (q - p)) top--;
          else break;
        }
        stack[top++] = i;
      }
      for (let k = 0; k + 1 < top; k++) {
        const p = stack[k] as number;
        const q = stack[k + 1] as number;
        const zp = map.z[row * map.width + p] as number;
        const zq = map.z[row * map.width + q] as number;
        for (let i = p; i <= q; i++) hull[i] = zp + ((zq - zp) * (i - p)) / (q - p);
      }
      let worst = 0;
      for (let i = a; i <= b; i++) {
        const d = (map.z[row * map.width + i] as number) - (hull[i] as number);
        if (d > worst) worst = d;
      }
      out.push(worst / map.bodyWidth);
    }
  }
  return summarise(out);
}

/**
 * `.splat`（antimatter15 形式、32 バイト）を、内部の 24 バイト列に直す。
 *
 * 書き出したファイルをそのまま測れるようにするため。位置と半径と色だけを
 * 使う。法線は `.splat` に無いので、回転からは起こさず「カメラを向く」で
 * 埋める（この物差しは前面の深度しか見ないので、それで足りる）。
 */
export function fromSplatFile(bytes: Uint8Array): { data: Uint8Array; count: number } {
  const STRIDE = 32;
  const count = Math.floor(bytes.byteLength / STRIDE);
  const src = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const buf = new ArrayBuffer(count * SPLAT_BYTES);
  const f = new Float32Array(buf);
  const u = new Uint32Array(buf);
  const stride = SPLAT_BYTES / 4;
  for (let i = 0; i < count; i++) {
    const o = i * STRIDE;
    const d = i * stride;
    f[d] = src.getFloat32(o, true);
    f[d + 1] = src.getFloat32(o + 4, true);
    f[d + 2] = src.getFloat32(o + 8, true);
    u[d + 3] = encodeOct(0, 0, -1);
    u[d + 4] = packHalf2(src.getFloat32(o + 12, true), src.getFloat32(o + 16, true));
    u[d + 5] = packRgba8(
      bytes[o + 24] as number,
      bytes[o + 25] as number,
      bytes[o + 26] as number,
      bytes[o + 27] as number,
    );
  }
  return { data: new Uint8Array(buf), count };
}
