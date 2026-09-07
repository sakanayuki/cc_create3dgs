/**
 * ⑦ 適応サンプリング（docs/04 §4.3.1 ②）。
 *
 * 四分木でセルを分けて、中身が一様なセルを1ガウシアンに統合する。
 * 平坦な壁や服の面は 8×8 画素を1個で済ませ、顔や輪郭は1画素ごとに残す。
 *
 * **決定的でなければならない。** `.pgs` はセルの情報を保存せず、読込時に
 * depth / color / alpha から四分木を引き直す（docs/05 §5.2.3）。同じ入力から
 * 同じ分割が出ないと、書き出したものと読み込んだものが食い違う。
 * したがってここには乱数も、走査順に依存する判断も入れない。
 *
 * 分割の基準は「セル全体の分散」ではなく **子4つの平均どうしのばらつき**
 * （群間分散）と「子のどれかが分割されるか」にする。設計（docs/05 §5.2.3）は
 * セル全体の分散で判定すると書いていたが、それでは書き出し→読込で分割が
 * 変わってしまう。実際に反例を作って確かめた。
 *
 *   全分散 = 群間分散 + 群内分散。統合はセル内を平均で塗り潰すので、
 *   読み直したときには群内分散が消えている。群内 0.9θ・群間 0.6θ のセルは、
 *   書き出し時は全分散 1.5θ > θ で分割されるのに、読込時は 0.6θ ≤ θ で
 *   統合される。8×8 に対して「書き 4 セル / 読み 1 セル」になった。
 *
 * 群間分散なら、統合しても各ノードの平均は変わらない（葉の平均の重み付き
 * 平均＝親の平均）ので、どの階層の判定も完全に再現される。細部を取りこぼす
 * 心配も要らない。市松模様のように群間分散が 0 でも細かい構造がある場合は、
 * 下の階層で群間分散が立ち、「子が分割される」が上へ伝わる。
 */

/** α がこの値以上の画素を被写体とみなす。 */
const SUBJECT_ALPHA = 128;

export interface SamplingParams {
  /**
   * 深度の群間分散の閾値。深度は 0〜1 に正規化した値で測る。
   * 子4つの平均のばらつきがこれを超えたら分割する。
   */
  readonly depthVar: number;
  /** 色の群間分散の閾値。3チャンネルの分散の和、各チャンネル 0〜1 で測る。 */
  readonly colorVar: number;
  /** 最大セル（画素）。これより大きいセルは中身によらず分割する。 */
  readonly maxCell: number;
  /** 最小セル（画素）。1 なら1画素まで分割できる。 */
  readonly minCell: number;
}

/**
 * 閾値の基準値。実際に使う閾値はここから倍率で作る（`solveSamplingParams`）。
 *
 * 閾値を直接決め打ちにしない理由は、統合率が写真によって大きく変わるため。
 * この基準値のままだと、合成した人物風の画像で 70% 統合された。設計
 * （docs/04 §4.7）が求めているのは 30% である。**閾値ではなく統合率のほうが
 * 設計の意図**なので、統合率から閾値を逆算する。
 */
const BASE_THRESHOLD: SamplingParams = { depthVar: 1e-5, colorVar: 1e-3, maxCell: 8, minCell: 1 };

/** docs/04 §4.7 の品質プリセットが狙う統合率。 */
export const REDUCTION_TARGETS = {
  light: 0.45,
  standard: 0.3,
  high: 0,
} as const;

/**
 * 統合しない設定（高品質プリセット）。閾値の逆算も要らないので定数で持つ。
 */
export const NO_SAMPLING: SamplingParams = { depthVar: 0, colorVar: 0, maxCell: 1, minCell: 1 };

/** 後方互換と単体テスト用。統合率ではなく閾値で指定したいとき。 */
export const SAMPLING_PRESETS = {
  light: { depthVar: 4e-5, colorVar: 3e-3, maxCell: 8, minCell: 1 },
  standard: BASE_THRESHOLD,
  high: NO_SAMPLING,
} as const satisfies Record<string, SamplingParams>;

export interface CellMap {
  /** 画素 → セル番号。被写体外は −1。 */
  readonly cellId: Int32Array;
  /** セルの左上 x。 */
  readonly x: Int32Array;
  readonly y: Int32Array;
  /** セルの一辺（画素）。 */
  readonly size: Int32Array;
  /** セル内の被写体画素数。 */
  readonly count: Int32Array;
  /** セル内の平均深度・平均色・平均 α。 */
  readonly depth: Float32Array;
  readonly color: Float32Array; // 3 要素 × セル数
  readonly alpha: Float32Array;
  /** セル数。 */
  readonly cellCount: number;
  /** 被写体画素数。統合率はこれとの比で測る。 */
  readonly subjectPixels: number;
}

interface Stats {
  count: number;
  /** 被写体でない画素の数。0 でも全画素でもなければ、セルはシルエットをまたいでいる。 */
  outside: number;
  depthMean: number;
  colorMean: [number, number, number];
  alphaMean: number;
}

/**
 * 最小セルの統計を直接数える。
 *
 * 積分画像を作れば O(1) で引けるが、最小セルは 1〜数画素なので直接数えても
 * 全体で O(n) にしかならない。積分画像は 1024² で 9 枚 × 8 バイト = 72 MB
 * 必要になり、スマートフォンでは持ちたくない。
 */
function leafStats(
  depth: ArrayLike<number>,
  color: ArrayLike<number>,
  alpha: ArrayLike<number>,
  width: number,
  height: number,
  cx: number,
  cy: number,
  size: number,
): Stats {
  let n = 0;
  let outside = 0;
  let sd = 0;
  let sr = 0;
  let sg = 0;
  let sb = 0;
  let sa = 0;

  const x1 = Math.min(width, cx + size);
  const y1 = Math.min(height, cy + size);
  for (let y = cy; y < y1; y++) {
    for (let x = cx; x < x1; x++) {
      const i = y * width + x;
      const a = alpha[i] as number;
      if (a < SUBJECT_ALPHA) {
        outside++;
        continue;
      }
      n++;
      sd += depth[i] as number;
      sr += (color[i * 4] as number) / 255;
      sg += (color[i * 4 + 1] as number) / 255;
      sb += (color[i * 4 + 2] as number) / 255;
      sa += a / 255;
    }
  }
  if (n === 0) {
    return { count: 0, outside, depthMean: 0, colorMean: [0, 0, 0], alphaMean: 0 };
  }
  return {
    count: n,
    outside,
    depthMean: sd / n,
    colorMean: [sr / n, sg / n, sb / n],
    alphaMean: sa / n,
  };
}

/** 木の1階層ぶんのノード。1つの根タイルの中だけを持つ。 */
interface LevelArrays {
  /** 一辺のノード数。 */
  readonly grid: number;
  /** ノードが覆う画素の一辺。 */
  readonly size: number;
  readonly count: Int32Array;
  readonly outside: Int32Array;
  readonly depth: Float64Array;
  readonly color: Float64Array; // 3 要素 × ノード数
  readonly alpha: Float64Array;
  readonly split: Uint8Array;
}

function makeLevel(grid: number, size: number): LevelArrays {
  const n = grid * grid;
  return {
    grid,
    size,
    count: new Int32Array(n),
    outside: new Int32Array(n),
    depth: new Float64Array(n),
    color: new Float64Array(n * 3),
    alpha: new Float64Array(n),
    split: new Uint8Array(n),
  };
}

/** 重み付き平均のまわりの群間分散。重み 0 のノードは数に入れない。 */
function betweenVariance(values: number[], weights: number[]): number {
  let w = 0;
  let m = 0;
  for (let i = 0; i < values.length; i++) {
    w += weights[i] as number;
    m += (values[i] as number) * (weights[i] as number);
  }
  if (w <= 0) return 0;
  m /= w;
  let v = 0;
  for (let i = 0; i < values.length; i++) {
    const d = (values[i] as number) - m;
    v += (weights[i] as number) * d * d;
  }
  return Math.max(0, v / w);
}

/**
 * 四分木で適応サンプリングし、セルマップを返す。
 *
 * @param depth 0〜1 に正規化した深度。長さ width×height。
 * @param color RGBA8。長さ width×height×4。
 * @param alpha 0〜255 の α。長さ width×height。
 */
function runQuadtree(
  depth: ArrayLike<number>,
  color: ArrayLike<number>,
  alpha: ArrayLike<number>,
  width: number,
  height: number,
  params: SamplingParams,
  /** 根タイルを何枚おきに見るか。1 以外は統合率の見積り専用。 */
  tileStride: number,
  /** セルマップを組み立てるか。見積りでは数だけ数えれば足りる。 */
  collect: boolean,
): CellMap {
  const minCell = Math.max(1, 1 << Math.round(Math.log2(Math.max(1, params.minCell))));
  const root = Math.max(minCell, 1 << Math.ceil(Math.log2(Math.max(params.maxCell, minCell))));
  const levelCount = Math.round(Math.log2(root / minCell)) + 1;

  // 根タイル1枚ぶんの作業領域。タイルごとに作り直さない（1024² で 16,384 枚ある）。
  const levels: LevelArrays[] = [];
  for (let l = 0; l < levelCount; l++) {
    const size = minCell << l;
    levels.push(makeLevel(root / size, size));
  }

  const n = width * height;
  const cellId = new Int32Array(n).fill(-1);
  const xs: number[] = [];
  const ys: number[] = [];
  const sizes: number[] = [];
  const counts: number[] = [];
  const depths: number[] = [];
  const colors: number[] = [];
  const alphas: number[] = [];
  let subjectPixels = 0;

  let cellCount = 0;
  const emit = (cx: number, cy: number, size: number, lv: LevelArrays, k: number): void => {
    cellCount++;
    subjectPixels += lv.count[k] as number;
    if (!collect) return;
    const id = xs.length;
    xs.push(cx);
    ys.push(cy);
    sizes.push(size);
    counts.push(lv.count[k] as number);
    depths.push(lv.depth[k] as number);
    colors.push(lv.color[k * 3] as number, lv.color[k * 3 + 1] as number, lv.color[k * 3 + 2] as number);
    alphas.push(lv.alpha[k] as number);

    const x1 = Math.min(width, cx + size);
    const y1 = Math.min(height, cy + size);
    for (let y = cy; y < y1; y++) {
      for (let x = cx; x < x1; x++) {
        const i = y * width + x;
        if ((alpha[i] as number) >= SUBJECT_ALPHA) cellId[i] = id;
      }
    }
  };

  const step = root * tileStride;
  for (let ty = 0; ty < height; ty += step) {
    for (let tx = 0; tx < width; tx += step) {
      // --- 最下層: 実データから直接数える
      const l0 = levels[0] as LevelArrays;
      for (let gy = 0; gy < l0.grid; gy++) {
        for (let gx = 0; gx < l0.grid; gx++) {
          const k = gy * l0.grid + gx;
          const s = leafStats(
            depth, color, alpha, width, height,
            tx + gx * minCell, ty + gy * minCell, minCell,
          );
          l0.count[k] = s.count;
          l0.outside[k] = s.outside;
          l0.depth[k] = s.depthMean;
          l0.color[k * 3] = s.colorMean[0];
          l0.color[k * 3 + 1] = s.colorMean[1];
          l0.color[k * 3 + 2] = s.colorMean[2];
          l0.alpha[k] = s.alphaMean;
          l0.split[k] = 0; // 最小セルは分割できない
        }
      }

      // --- 上の層: 子4つから積み上げる
      for (let l = 1; l < levelCount; l++) {
        const cur = levels[l] as LevelArrays;
        const ch = levels[l - 1] as LevelArrays;
        for (let gy = 0; gy < cur.grid; gy++) {
          for (let gx = 0; gx < cur.grid; gx++) {
            const k = gy * cur.grid + gx;
            const kids = [
              (gy * 2) * ch.grid + gx * 2,
              (gy * 2) * ch.grid + gx * 2 + 1,
              (gy * 2 + 1) * ch.grid + gx * 2,
              (gy * 2 + 1) * ch.grid + gx * 2 + 1,
            ];

            let count = 0;
            let outside = 0;
            let anySplit = false;
            let sd = 0;
            let sr = 0;
            let sg = 0;
            let sb = 0;
            let sa = 0;
            for (const c of kids) {
              const w = ch.count[c] as number;
              count += w;
              outside += ch.outside[c] as number;
              if (ch.split[c] === 1) anySplit = true;
              sd += (ch.depth[c] as number) * w;
              sr += (ch.color[c * 3] as number) * w;
              sg += (ch.color[c * 3 + 1] as number) * w;
              sb += (ch.color[c * 3 + 2] as number) * w;
              sa += (ch.alpha[c] as number) * w;
            }

            cur.count[k] = count;
            cur.outside[k] = outside;
            if (count > 0) {
              cur.depth[k] = sd / count;
              cur.color[k * 3] = sr / count;
              cur.color[k * 3 + 1] = sg / count;
              cur.color[k * 3 + 2] = sb / count;
              cur.alpha[k] = sa / count;
            }

            const w = kids.map((c) => ch.count[c] as number);
            const vDepth = betweenVariance(kids.map((c) => ch.depth[c] as number), w);
            const vColor =
              betweenVariance(kids.map((c) => ch.color[c * 3] as number), w) +
              betweenVariance(kids.map((c) => ch.color[c * 3 + 1] as number), w) +
              betweenVariance(kids.map((c) => ch.color[c * 3 + 2] as number), w);

            // シルエットをまたぐセルは必ず割る。またいだまま統合すると、
            // 被写体の外の画素まで同じ色・同じ深度のガウシアンに含まれ、
            // 輪郭が四角く膨らむ。
            const straddles = count > 0 && outside > 0;
            cur.split[k] =
              straddles ||
              cur.size > params.maxCell ||
              anySplit ||
              vDepth > params.depthVar ||
              vColor > params.colorVar
                ? 1
                : 0;
          }
        }
      }

      // --- 上から降りて、分割しないノードを1セルとして出す
      const walk = (l: number, gx: number, gy: number): void => {
        const lv = levels[l] as LevelArrays;
        const k = gy * lv.grid + gx;
        if ((lv.count[k] as number) === 0) return; // 被写体がまったく無い
        if (l > 0 && lv.split[k] === 1) {
          walk(l - 1, gx * 2, gy * 2);
          walk(l - 1, gx * 2 + 1, gy * 2);
          walk(l - 1, gx * 2, gy * 2 + 1);
          walk(l - 1, gx * 2 + 1, gy * 2 + 1);
          return;
        }
        emit(tx + gx * lv.size, ty + gy * lv.size, lv.size, lv, k);
      };
      walk(levelCount - 1, 0, 0);
    }
  }

  return {
    cellId,
    x: Int32Array.from(xs),
    y: Int32Array.from(ys),
    size: Int32Array.from(sizes),
    count: Int32Array.from(counts),
    depth: Float32Array.from(depths),
    color: Float32Array.from(colors),
    alpha: Float32Array.from(alphas),
    cellCount,
    subjectPixels,
  };
}

/**
 * 四分木で適応サンプリングし、セルマップを返す。
 *
 * @param depth 0〜1 に正規化した深度。長さ width×height。
 * @param color RGBA8。長さ width×height×4。
 * @param alpha 0〜255 の α。長さ width×height。
 */
export function adaptiveSample(
  depth: ArrayLike<number>,
  color: ArrayLike<number>,
  alpha: ArrayLike<number>,
  width: number,
  height: number,
  params: SamplingParams = SAMPLING_PRESETS.standard,
): CellMap {
  return runQuadtree(depth, color, alpha, width, height, params, 1, true);
}

/**
 * 統合率。0 なら1画素1ガウシアン、0.3 なら 30% 減。
 *
 * docs/04 §4.3.1 は標準で −30% を見込んでいる。見込みが実際に出るかは
 * 写真によるので、生成時にこの値を計測して記録する。
 */
export function samplingReduction(map: CellMap): number {
  if (map.subjectPixels === 0) return 0;
  return 1 - map.cellCount / map.subjectPixels;
}

/**
 * セルの値を全画素へ書き戻す（docs/05 §5.2.3 の「継ぎ目補正との整合」）。
 *
 * 統合したセルの中は一様にする。こうしておくと、書き出した `.pgs` を
 * 読み直したときに分散 0 のセルとして同じ分割が再現される。
 */
export function flattenToPlanes(
  map: CellMap,
  color: Uint8ClampedArray,
  alpha: Uint8ClampedArray,
  depth: Float32Array,
): void {
  for (let i = 0; i < map.cellId.length; i++) {
    const id = map.cellId[i] as number;
    if (id < 0) continue;
    depth[i] = map.depth[id] as number;
    color[i * 4] = Math.round((map.color[id * 3] as number) * 255);
    color[i * 4 + 1] = Math.round((map.color[id * 3 + 1] as number) * 255);
    color[i * 4 + 2] = Math.round((map.color[id * 3 + 2] as number) * 255);
    alpha[i] = Math.round((map.alpha[id] as number) * 255);
  }
}

/**
 * 統合率を見積もる。根タイルを間引いて見るので全走査より速い。
 *
 * @param tileStride 何枚おきに見るか。4 なら 1/16 の仕事量。
 */
export function estimateReduction(
  depth: ArrayLike<number>,
  color: ArrayLike<number>,
  alpha: ArrayLike<number>,
  width: number,
  height: number,
  params: SamplingParams,
  tileStride = 4,
): number {
  return samplingReduction(
    runQuadtree(depth, color, alpha, width, height, params, Math.max(1, tileStride), false),
  );
}

/**
 * 目標の統合率になる閾値を求める（docs/04 §4.7 のプリセット値を実現する）。
 *
 * 求めた閾値は `.pgs` のマニフェストに書く。読込側は同じ閾値で四分木を
 * 引き直すので、書いたものと同じ分割が再現される（docs/05 §5.2.3）。
 * 閾値を保存せず「標準プリセット」とだけ書いたのでは、統合率が写真ごとに
 * 違うため復元できない。
 *
 * 倍率について二分探索する。統合率は倍率について単調非減少なので、
 * 対数スケールで 12 回も回せば十分に収束する。見積りはタイルを間引いて
 * 行うので、全走査 1.5 回ぶん程度の費用で済む。
 */
export function solveSamplingParams(
  depth: ArrayLike<number>,
  color: ArrayLike<number>,
  alpha: ArrayLike<number>,
  width: number,
  height: number,
  targetReduction: number,
  base: SamplingParams = BASE_THRESHOLD,
  tileStride = 4,
): SamplingParams {
  if (targetReduction <= 0) return NO_SAMPLING;

  const scaled = (k: number): SamplingParams => ({
    depthVar: base.depthVar * k,
    colorVar: base.colorVar * k,
    maxCell: base.maxCell,
    minCell: base.minCell,
  });
  const at = (k: number): number =>
    estimateReduction(depth, color, alpha, width, height, scaled(k), tileStride);

  // 上限が目標に届かないなら、それ以上探しても無駄。最大まで統合する。
  let hi = 1e3;
  if (at(hi) < targetReduction) return scaled(hi);
  let lo = 1e-6;
  if (at(lo) > targetReduction) return scaled(lo);

  for (let i = 0; i < 12; i++) {
    const mid = Math.sqrt(lo * hi); // 対数スケールの中点
    if (at(mid) < targetReduction) lo = mid;
    else hi = mid;
  }
  return scaled(Math.sqrt(lo * hi));
}
