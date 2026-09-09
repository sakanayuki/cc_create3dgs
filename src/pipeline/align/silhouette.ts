/**
 * シルエットの比較（docs/12 §12.7 の目的関数、§12.13 の M1）。
 *
 * 90° 近い基線では特徴点マッチングが効かない（docs/12 §12.14 R15）。
 * 効くのはシルエットで、しかも**合格の指標（M1 = IoU）をそのまま目的関数に使える**。
 * 最適化の対象と、合格の判定が同じものになるのは都合がよい。
 *
 * ただし IoU は段々の関数で、局所探索の手がかりに乏しい。そこで
 * 「はみ出した点が、被写体までどれだけ遠いか」（距離変換で測る）を足して滑らかにする。
 * IoU が尺度を押さえ、距離が向きを押さえる。
 */
import { distanceTransform } from '../geometry/distanceTransform';

/** 位置合わせの計算に使う、粗い解像度のシルエット。 */
export interface SilhouetteGrid {
  readonly width: number;
  readonly height: number;
  /** 1 が被写体。 */
  readonly mask: Uint8Array;
  /** 被写体の外側では最も近い被写体画素までの距離、内側は 0。 */
  readonly distanceToSubject: Float32Array;
  /** 被写体の画素数。 */
  readonly area: number;
  /** 元画像からこのグリッドへの倍率。 */
  readonly scale: number;
  /** 被写体の外接矩形の幅。はみ出しは横向きに起きるので、これで正規化する。 */
  readonly bodyWidth: number;
  /** 被写体の外接矩形の高さ。ヨーに対して変わらないので、尺度の錨に使う。 */
  readonly bodyHeight: number;
  /**
   * 粗いグリッド上の深度（セル内の最小 z）。被写体の外は 0。
   *
   * 「その view が実際に見た面より手前に、他の view の点が来てはいけない」
   * という自由空間の判定に使う（docs/12 §12.7 の目的関数 2）。
   * 手前に来るなら、その view はそちらを見ていたはずである。
   */
  readonly depth: Float32Array;
}

/**
 * α プレーンから粗いシルエットを作る。
 *
 * 位置合わせは形の大づかみだけを見るので、元の 1024² で回す必要はない。
 * `targetLongSide` は既定 128。1024² から 128² へ落とすと計算は 64 分の 1 になる。
 */
export function buildSilhouette(
  alpha: ArrayLike<number>,
  width: number,
  height: number,
  options: {
    readonly targetLongSide?: number;
    readonly threshold?: number;
    /** 画素ごとの深度。渡すと粗い深度マップも作る。 */
    readonly depth?: ArrayLike<number>;
  } = {},
): SilhouetteGrid {
  const target = options.targetLongSide ?? 128;
  const threshold = options.threshold ?? 128;
  if (alpha.length !== width * height) {
    throw new Error(`α の長さが合いません: ${alpha.length}（期待 ${width * height}）`);
  }

  const scale = Math.min(1, target / Math.max(width, height));
  const w = Math.max(1, Math.round(width * scale));
  const h = Math.max(1, Math.round(height * scale));
  const mask = new Uint8Array(w * h);
  const depthGrid = new Float32Array(w * h);
  let minX = Infinity;
  let maxX = -Infinity;
  let minY = Infinity;
  let maxY = -Infinity;
  const srcDepth = options.depth;

  // 面積の平均で縮める。細い部位が消えないよう、半分以上が被写体なら被写体とする。
  const sx = width / w;
  const sy = height / h;
  let area = 0;
  for (let y = 0; y < h; y++) {
    const y0 = Math.floor(y * sy);
    const y1 = Math.max(y0 + 1, Math.floor((y + 1) * sy));
    for (let x = 0; x < w; x++) {
      const x0 = Math.floor(x * sx);
      const x1 = Math.max(x0 + 1, Math.floor((x + 1) * sx));
      let hit = 0;
      let total = 0;
      let zMin = Infinity;
      for (let yy = y0; yy < y1 && yy < height; yy++) {
        for (let xx = x0; xx < x1 && xx < width; xx++) {
          const si = yy * width + xx;
          total++;
          if ((alpha[si] as number) >= threshold) {
            hit++;
            if (srcDepth) {
              const z = srcDepth[si] as number;
              if (z > 0 && z < zMin) zMin = z;
            }
          }
        }
      }
      const on = total > 0 && hit * 2 >= total ? 1 : 0;
      mask[y * w + x] = on;
      if (on && zMin < Infinity) depthGrid[y * w + x] = zMin;
      if (on) {
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
      }
      area += on;
    }
  }

  // distanceTransform は「inside が真の画素から、偽の画素までの距離」を返す。
  // inside を反転して呼ぶと、背景から被写体までの距離になる（被写体の上では 0）。
  const distanceToSubject = distanceTransform(mask, w, h, (v) => v === 0);

  return {
    width: w,
    height: h,
    mask,
    distanceToSubject,
    area,
    scale,
    bodyWidth: Math.max(1, maxX - minX + 1),
    bodyHeight: Math.max(1, maxY - minY + 1),
    depth: depthGrid,
  };
}

/**
 * 距離変換を小数の座標で読む（双一次補間）。
 *
 * 横向きの被写体は粗いグリッドで 15〜20 画素しか幅がない。整数へ丸めると、
 * 幾何の差より丸めの差のほうが大きくなり、**真の姿勢が最小にならない**。
 * 実際にそれで最適化が真値を追い越した（docs/12 §12.15.2）。
 */
export function sampleBilinear(
  field: Float32Array,
  width: number,
  height: number,
  x: number,
  y: number,
): number {
  const cx = Math.min(width - 1, Math.max(0, x));
  const cy = Math.min(height - 1, Math.max(0, y));
  const x0 = Math.floor(cx);
  const y0 = Math.floor(cy);
  const x1 = Math.min(width - 1, x0 + 1);
  const y1 = Math.min(height - 1, y0 + 1);
  const fx = cx - x0;
  const fy = cy - y0;
  const a = field[y0 * width + x0] as number;
  const b = field[y0 * width + x1] as number;
  const c = field[y1 * width + x0] as number;
  const d = field[y1 * width + x1] as number;
  return a * (1 - fx) * (1 - fy) + b * fx * (1 - fy) + c * (1 - fx) * fy + d * fx * fy;
}

/**
 * 2つのマスクの IoU。M1（docs/12 §12.13）の計算そのもの。
 *
 * 両方とも空なら 1 を返す（比べるものが無いのに 0 を返すと最適化が壊れる）。
 */
export function iou(a: ArrayLike<number>, b: ArrayLike<number>): number {
  if (a.length !== b.length) throw new Error('マスクの長さが違います');
  let inter = 0;
  let union = 0;
  for (let i = 0; i < a.length; i++) {
    const x = (a[i] as number) !== 0;
    const y = (b[i] as number) !== 0;
    if (x && y) inter++;
    if (x || y) union++;
  }
  return union === 0 ? 1 : inter / union;
}
