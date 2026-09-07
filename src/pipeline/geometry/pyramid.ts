/**
 * ガウシアン／ラプラシアンピラミッド（Burt & Adelson, 1983）。
 *
 * 深度の2パス融合（docs/03 §3.4）で使う。全体パスとタイルパスを
 * 単純な重み付き平均で混ぜると、重なりの端に段差が残る。理由は、
 * 低周波（全体の傾き）の食い違いが大きいのに、混ぜる幅が狭いため。
 * 周波数帯ごとに違う幅で混ぜれば、低周波は広く、高周波は狭く混ざり、
 * 段差もぼけも出ない。それがラプラシアンブレンド。
 *
 * カーネルは 5 タップの二項係数 [1,4,6,4,1]/16。分離可能なので O(n)。
 */

const K = [1 / 16, 4 / 16, 6 / 16, 4 / 16, 1 / 16] as const;

export interface Level {
  readonly data: Float32Array;
  readonly width: number;
  readonly height: number;
}

/** 端は折り返す。0 で埋めると縁が暗くなり、そこに偽の段差ができる。 */
function reflect(i: number, n: number): number {
  if (i < 0) return -i - 1 < n ? -i - 1 : 0;
  if (i >= n) return 2 * n - i - 1 >= 0 ? 2 * n - i - 1 : n - 1;
  return i;
}

/** 5タップの分離可能な平滑化。 */
export function smooth(src: ArrayLike<number>, width: number, height: number): Float32Array {
  const tmp = new Float32Array(width * height);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      let s = 0;
      for (let k = -2; k <= 2; k++) s += (K[k + 2] as number) * (src[y * width + reflect(x + k, width)] as number);
      tmp[y * width + x] = s;
    }
  }
  const out = new Float32Array(width * height);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      let s = 0;
      for (let k = -2; k <= 2; k++) s += (K[k + 2] as number) * (tmp[reflect(y + k, height) * width + x] as number);
      out[y * width + x] = s;
    }
  }
  return out;
}

/** 平滑化してから 1/2 に間引く。 */
export function downsample(src: ArrayLike<number>, width: number, height: number): Level {
  const blurred = smooth(src, width, height);
  const w = Math.max(1, Math.ceil(width / 2));
  const h = Math.max(1, Math.ceil(height / 2));
  const out = new Float32Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      out[y * w + x] = blurred[Math.min(height - 1, y * 2) * width + Math.min(width - 1, x * 2)] as number;
    }
  }
  return { data: out, width: w, height: h };
}

/**
 * 2倍に引き伸ばす。目標の大きさを渡すのは、奇数を丸めた分を戻すため。
 *
 * 偶数位置に値を置いてから平滑化し、間引きで落ちたエネルギーぶん 4 倍する。
 */
export function upsample(src: Level, targetWidth: number, targetHeight: number): Float32Array {
  const spread = new Float32Array(targetWidth * targetHeight);
  for (let y = 0; y < src.height; y++) {
    const ty = y * 2;
    if (ty >= targetHeight) break;
    for (let x = 0; x < src.width; x++) {
      const tx = x * 2;
      if (tx >= targetWidth) break;
      spread[ty * targetWidth + tx] = (src.data[y * src.width + x] as number) * 4;
    }
  }
  return smooth(spread, targetWidth, targetHeight);
}

/** ガウシアンピラミッド。levels[0] が原寸。 */
export function gaussianPyramid(
  src: ArrayLike<number>,
  width: number,
  height: number,
  levels: number,
): Level[] {
  const out: Level[] = [{ data: Float32Array.from(src as ArrayLike<number>), width, height }];
  for (let l = 1; l < levels; l++) {
    const prev = out[l - 1] as Level;
    if (prev.width <= 2 || prev.height <= 2) break;
    out.push(downsample(prev.data, prev.width, prev.height));
  }
  return out;
}

/**
 * ラプラシアンピラミッド。最上位（最も粗い層）だけはガウシアンそのもの。
 * これが無いと畳み直したときに直流成分が失われる。
 */
export function laplacianPyramid(
  src: ArrayLike<number>,
  width: number,
  height: number,
  levels: number,
): Level[] {
  const g = gaussianPyramid(src, width, height, levels);
  const out: Level[] = [];
  for (let l = 0; l < g.length - 1; l++) {
    const cur = g[l] as Level;
    const up = upsample(g[l + 1] as Level, cur.width, cur.height);
    const d = new Float32Array(cur.width * cur.height);
    for (let i = 0; i < d.length; i++) d[i] = (cur.data[i] as number) - (up[i] as number);
    out.push({ data: d, width: cur.width, height: cur.height });
  }
  out.push(g[g.length - 1] as Level);
  return out;
}

/** ラプラシアンピラミッドを畳んで1枚に戻す。 */
export function collapse(pyramid: Level[]): Float32Array {
  let acc = (pyramid[pyramid.length - 1] as Level).data;
  for (let l = pyramid.length - 2; l >= 0; l--) {
    const cur = pyramid[l] as Level;
    const prev = pyramid[l + 1] as Level;
    const up = upsample({ data: acc, width: prev.width, height: prev.height }, cur.width, cur.height);
    const next = new Float32Array(cur.width * cur.height);
    for (let i = 0; i < next.length; i++) next[i] = (cur.data[i] as number) + (up[i] as number);
    acc = next;
  }
  return acc;
}

/**
 * 複数の入力を重みに従ってラプラシアンブレンドする。
 *
 * @param sources 同じ大きさの入力たち。
 * @param weights 各入力の重み。負でないこと。全部 0 の画素は 0 を返す。
 * @param levels  段数。1024² なら 6 段（最小 32²）が目安。
 */
export function blendLaplacian(
  sources: ArrayLike<number>[],
  weights: ArrayLike<number>[],
  width: number,
  height: number,
  levels = 6,
): Float32Array {
  if (sources.length === 0) throw new Error('入力がありません');
  if (sources.length !== weights.length) throw new Error('入力と重みの数が違います');

  // 重みは正規化してからピラミッドにする。層ごとに正規化すると
  // 平滑化で合計が 1 からずれ、明るさ（＝深度）が層ごとに変わってしまう。
  const n = width * height;
  const norm: Float32Array[] = weights.map(() => new Float32Array(n));
  for (let i = 0; i < n; i++) {
    let sum = 0;
    for (const w of weights) sum += Math.max(0, w[i] as number);
    if (sum <= 0) continue;
    for (let k = 0; k < weights.length; k++) {
      (norm[k] as Float32Array)[i] = Math.max(0, (weights[k] as ArrayLike<number>)[i] as number) / sum;
    }
  }

  const lap = sources.map((s) => laplacianPyramid(s, width, height, levels));
  const gw = norm.map((w) => gaussianPyramid(w, width, height, levels));
  const depth = Math.min(...lap.map((p) => p.length));

  const blended: Level[] = [];
  for (let l = 0; l < depth; l++) {
    const ref = (lap[0] as Level[])[l] as Level;
    const d = new Float32Array(ref.width * ref.height);
    for (let k = 0; k < lap.length; k++) {
      const src = (lap[k] as Level[])[l] as Level;
      const w = (gw[k] as Level[])[l] as Level;
      for (let i = 0; i < d.length; i++) {
        d[i] = (d[i] as number) + (src.data[i] as number) * (w.data[i] as number);
      }
    }
    blended.push({ data: d, width: ref.width, height: ref.height });
  }
  return collapse(blended);
}
