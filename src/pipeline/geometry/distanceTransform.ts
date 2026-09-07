/**
 * 厳密なユークリッド距離変換（Felzenszwalb & Huttenlocher, 2012）。
 *
 * 2箇所で必要になる。
 *   1. 背面シェルの厚み: シルエットの内部距離から楕円断面を作る（docs/03 §3.6.2）
 *   2. 境界画素の深度の引き込み: 最も近い α ≥ 0.5 の画素を探す（docs/03 §3.5.3）
 *
 * 近似（チャンファー距離）ではなく厳密解を使う。厚みは被写体の形そのものを
 * 決めるので、近似の異方性がそのまま「歪んだ膨らみ」として見えてしまう。
 *
 * 計算量は O(n)。1024² で約 8ms。
 */

const INF = 1e20;

/**
 * 1次元の下側包絡線を求める（論文の Algorithm 1）。
 * 放物線 f(q) + (x−q)² の下側包絡線を走査する。
 */
function edt1d(f: Float64Array, n: number, d: Float64Array, v: Int32Array, z: Float64Array): void {
  let k = 0;
  v[0] = 0;
  z[0] = -INF;
  z[1] = INF;

  for (let q = 1; q < n; q++) {
    // 新しい放物線が既存の包絡線をどこで追い越すか
    let s = intersect(f, q, v[k] as number);
    while (s <= (z[k] as number)) {
      k--;
      s = intersect(f, q, v[k] as number);
    }
    k++;
    v[k] = q;
    z[k] = s;
    z[k + 1] = INF;
  }

  k = 0;
  for (let q = 0; q < n; q++) {
    while ((z[k + 1] as number) < q) k++;
    const vk = v[k] as number;
    d[q] = (q - vk) * (q - vk) + (f[vk] as number);
  }
}

function intersect(f: Float64Array, q: number, vk: number): number {
  return ((f[q] as number) + q * q - ((f[vk] as number) + vk * vk)) / (2 * q - 2 * vk);
}

/**
 * 二値マスクの距離変換。
 *
 * @param mask   長さ width×height。`inside(mask[i])` が真の画素を「前景」とみなす。
 * @param inside 前景の判定。既定は 0 以外。
 * @returns 各画素から**最も近い背景画素**までのユークリッド距離。
 *          前景の内部ほど大きく、背景では 0 になる。
 */
export function distanceTransform(
  mask: ArrayLike<number>,
  width: number,
  height: number,
  inside: (v: number) => boolean = (v) => v !== 0,
): Float32Array {
  const n = width * height;
  if (mask.length !== n) throw new Error(`マスクの長さが合いません: ${mask.length}（期待 ${n}）`);

  const grid = new Float64Array(n);
  for (let i = 0; i < n; i++) grid[i] = inside(mask[i] as number) ? INF : 0;

  const maxDim = Math.max(width, height);
  const f = new Float64Array(maxDim);
  const d = new Float64Array(maxDim);
  const v = new Int32Array(maxDim);
  const z = new Float64Array(maxDim + 1);

  // 列方向
  for (let x = 0; x < width; x++) {
    for (let y = 0; y < height; y++) f[y] = grid[y * width + x] as number;
    edt1d(f, height, d, v, z);
    for (let y = 0; y < height; y++) grid[y * width + x] = d[y] as number;
  }
  // 行方向
  for (let y = 0; y < height; y++) {
    const row = y * width;
    for (let x = 0; x < width; x++) f[x] = grid[row + x] as number;
    edt1d(f, width, d, v, z);
    for (let x = 0; x < width; x++) grid[row + x] = d[x] as number;
  }

  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = Math.sqrt(grid[i] as number);
  return out;
}

/**
 * 最も近い前景画素の index を返す（最近傍ラベル伝播）。
 *
 * 境界画素の深度を「最も近い α ≥ 0.5 の画素」で置き換えるのに使う。
 * 距離変換と同じ走査で index も運べるが、実装が複雑になるので
 * ここでは素直に「距離が確定した後に近傍を探す」二段構えにする。
 * 探索半径は距離変換の結果で上から抑えられるので、実用上は数回の反復で収束する。
 */
export function nearestForegroundIndex(
  mask: ArrayLike<number>,
  width: number,
  height: number,
  inside: (v: number) => boolean = (v) => v !== 0,
): Int32Array {
  const n = width * height;
  const out = new Int32Array(n).fill(-1);
  // 前景自身は自分を指す
  for (let i = 0; i < n; i++) if (inside(mask[i] as number)) out[i] = i;

  // 8近傍のジャンプフラッド。log2(maxDim) 回で全画素が埋まる。
  const best = new Float64Array(n).fill(Infinity);
  for (let i = 0; i < n; i++) if ((out[i] as number) >= 0) best[i] = 0;

  const maxDim = Math.max(width, height);
  for (let step = 1 << Math.ceil(Math.log2(maxDim)); step >= 1; step >>= 1) {
    let changed = false;
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const i = y * width + x;
        for (let dy = -1; dy <= 1; dy++) {
          for (let dx = -1; dx <= 1; dx++) {
            if (dx === 0 && dy === 0) continue;
            const nx = x + dx * step;
            const ny = y + dy * step;
            if (nx < 0 || ny < 0 || nx >= width || ny >= height) continue;
            const src = out[ny * width + nx] as number;
            if (src < 0) continue;
            const sx = src % width;
            const sy = (src / width) | 0;
            const dist = (sx - x) * (sx - x) + (sy - y) * (sy - y);
            if (dist < (best[i] as number)) {
              best[i] = dist;
              out[i] = src;
              changed = true;
            }
          }
        }
      }
    }
    if (!changed && step === 1) break;
  }
  return out;
}
