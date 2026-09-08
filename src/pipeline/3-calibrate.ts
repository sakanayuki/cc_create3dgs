/**
 * 深度較正（docs/03 §3.5）。
 *
 * 単眼深度モデルの出力をそのまま3Dにはできない。
 *   ・DA3 系: 深度を直接予測する。スケールだけが不定
 *   ・V2 系:  affine-invariant な**逆深度**。スケールとシフトの両方が不定で、
 *             未知のシフト b が被写体の膨らみ・平坦さ、つまり形そのものを歪める
 *
 * ここでは尺度 a を「被写体の奥行き ≈ 短辺の 0.65 倍」から決め、
 * V2 経路のシフト b は**シルエット法線の事前分布**から推定する。
 */
import { distanceTransform, nearestForegroundIndex } from './geometry/distanceTransform';

export type DepthOutputKind = 'depth' | 'inverse-depth';

export interface CalibrationInput {
  /** モデルの生出力。値域は正規化されていなくてよい。 */
  readonly raw: Float32Array;
  readonly width: number;
  readonly height: number;
  /** 被写体マット 0..255。128 以上を被写体とみなす。 */
  readonly alpha: Uint8ClampedArray;
  readonly kind: DepthOutputKind;
  readonly focalPx: number;
  /**
   * 奥行き ÷ 短辺を**明示的に**指定する（UI の「立体感」スライダ）。
   *
   * 省略した場合、深度を直接出すモデル（DA3）では**実寸をそのまま使う**。
   * 逆深度しか出さないモデル（V2）は絶対スケールを持たないので、
   * 省略時は既定値 0.65 を使う。
   */
  readonly depthToWidthRatio?: number;

  /**
   * 帯ごとの「奥行き ÷ 幅」の上限。既定は PLAUSIBLE_BAND_DEPTH_TO_WIDTH。
   * 実寸経路（depthToWidthRatio 未指定）でのみ効く。
   */
  readonly bandMaxRatio?: number;
  /**
   * 局所的な起伏の強調倍率（1 で無効）。
   *
   * 実寸のままだと、人物は「奥行き ÷ 高さ ≒ 0.6」の薄い物体なので、
   * 顔の凹凸が見かけの大きさに対して小さく、平らな面に見える。
   * 大域の形（＝実寸の比率）は保ったまま、細かい起伏だけを持ち上げる。
   */
  readonly reliefBoost?: number;
  /** 局所強調の平滑化半径。被写体の短辺に対する割合。 */
  readonly reliefRadius?: number;
}

export interface CalibrationResult {
  /** 0..65535 に正規化した深度。手前が小さい。 */
  readonly depth: Uint16Array;
  readonly nearZ: number;
  readonly farZ: number;
  /** V2 経路で解いたシフト b。DA3 経路では 0。 */
  readonly shift: number;
  /** 互換のため残す。solveAffine は最適化しないので常に 0。 */
  readonly silhouetteSamples: number;
  /** 実寸をそのまま使ったか（DA3 経路で ratio 未指定のとき true）。 */
  readonly metric: boolean;
  /** 被写体の 奥行き ÷ 高さ。構図で変わるので診断用。 */
  readonly depthToHeight: number;
  /** 被写体の 奥行き ÷ 幅（短辺）。人物なら 0.9 前後が自然。妥当性はこれで見る。 */
  readonly depthToWidth: number;
}

const SUBJECT_THRESHOLD = 128;
const DEFAULT_RATIO = 0.65;

/** 局所強調の既定値。実写の人物で測って決めた（docs/03 §3.5.4）。 */
const DEFAULT_RELIEF_BOOST = 3;
const DEFAULT_RELIEF_RADIUS = 0.04;

/**
 * 被写体の「奥行き ÷ 幅（短辺）」として許す範囲（v2.4、他実装との比較で改訂）。
 *
 * **高さではなく幅で見る。** 奥行き ÷ 高さは構図で大きく変わる（全身なら
 * 0.35 前後、バストアップなら 1 に近づく）ので、一つの範囲では縛れない。
 * 一方、人はどこを切っても「幅とおおよそ同じだけ奥行きがある」。実測でも
 * 頭 0.69 / 胸 0.77 / 腰 0.91 と安定していた。
 *
 * 理想的な出力（別実装の結果）を解析すると、全体で 奥行き ÷ 幅 = 0.895
 * だった。私たちの出力は 1.90 で、体が視線方向に 2 倍伸びていた。少し
 * 回すだけで串のように崩れるのはこれが原因である。
 *
 * DA3 の実寸は**絶対値としては当てにならない**。この画像では、こちらの
 * 加工を一切かけない生の実寸ですら 奥行き ÷ 身長 が 0.506（理想 0.356）
 * と 1.4 倍あった。形の相対関係は使い、全体の伸びだけをここで抑える。
 *
 * 上限を 0.85 に置くのは、この後に背面シェルが厚みを上乗せするため。
 * 厚み 0.12 で 0.04 ほど足されるので、点群としては 0.9 前後に着地する。
 * 実測ではどの写真でも上限に張り付くので、事実上「妥当な奥行きへ
 * 正規化する」処理になっている。実寸を信じきれない以上それが正しい。
 */
const PLAUSIBLE_DEPTH_TO_WIDTH = { min: 0.40, max: 0.85 } as const;

/**
 * 帯ごとの「奥行き ÷ 幅」の上限（flattenImplausibleBands 用）。
 *
 * 実写 2 枚（座位・立位）で振って、理想側との**帯ごとの差の合計**で選んだ。
 * 数字が小さいほど平らになる。
 *
 * | 上限 | 座位の差 | 立位の差 | 合計 |
 * |---|---|---|---|
 * | なし | 1.92 | 0.75 | 2.67 |
 * | 1.00 | 1.47 | 0.16 | 1.63 |
 * | 0.85 | 1.19 | 0.21 | 1.40 |
 * | **0.70** | **0.97** | **0.68** | **1.65** |
 * | 0.50 | 0.82 | 1.38 | 2.20 |
 *
 * 合計だけ見れば 0.85 が最小だが、それは立位がほぼ一致するからで、座位の
 * 胸は 0.92（理想 0.14）と深いままである。今回直したいのは**回すと胴が
 * 横に裂ける**ことなので、35° から描いて見比べ、裂け目がいちばん小さい
 * 0.70 を採った。立位はこの値で理想より 2 割ほど浅くなるが、破綻はしない。
 *
 * 0.50 まで絞ると座位はさらに良くなるが、立位を**上限なしより悪くする**。
 * 立位は体が細く帯ごとの比が元から 0.81〜1.05 あるので、そこを 0.50 で
 * 切ると人が板になる。
 */
const PLAUSIBLE_BAND_DEPTH_TO_WIDTH = 0.70;

/**
 * 高さ方向の帯ごとに「奥行き ÷ 幅」を見て、人体としてあり得ない帯だけ潰す
 * （v2.5、他実装との比較で追加）。
 *
 * 全体でひとつの比を見るだけでは足りない。座った人物は伸ばした脚で幅が
 * 決まるので、**全体の比が正しくても胴だけが極端に深い**という壊れ方が
 * 起きる。実測（他実装 対 こちら）:
 *
 * | 部位 | 理想 | こちら | 比 |
 * |---|---|---|---|
 * | 頭 | 0.51 | 1.34 | 2.6 倍 |
 * | 胸 | 0.14 | 1.18 | 8.4 倍 |
 * | 腰/膝 | 0.68 | 0.66 | 1.0 倍 |
 * | 脚 | 0.73 | 0.70 | 1.0 倍 |
 * | **全体** | **0.76** | **0.81** | **1.1 倍** |
 *
 * 脚が幅を決めるので全体の比は合ってしまい、胴の 8 倍が隠れる。斜め
 * 35° から見ると、この胴が横方向の裂け目になって現れる。
 *
 * 帯ごとの中央値 m(y) は動かさず、その周りの広がりだけを k(y) 倍する。
 * 帯どうしの前後関係は保たれるので、体が分断されない。m も k も y 方向に
 * 平滑化してから使う。
 *
 * 上限の選び方は PLAUSIBLE_BAND_DEPTH_TO_WIDTH の表を見よ。人体の帯ごとの
 * 比は姿勢と部位で 0.14〜1.05 と 7 倍も違うので、絞りすぎると本物の形まで
 * 潰れる。
 */
export function flattenImplausibleBands(
  z: Float32Array,
  alpha: ArrayLike<number>,
  width: number,
  height: number,
  focalPx: number,
  maxRatio: number,
  bandCount = 12,
): Float32Array {
  // 被写体の y 範囲
  let y0 = height;
  let y1 = -1;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      if ((alpha[y * width + x] as number) >= SUBJECT_THRESHOLD) {
        if (y < y0) y0 = y;
        if (y > y1) y1 = y;
        break;
      }
    }
  }
  if (y1 < y0) return z;
  const bandH = Math.max(1, Math.ceil((y1 - y0 + 1) / bandCount));

  // 行ごとの幅。比の分母は**帯ではなく行**の幅で取る。
  //
  // 帯の外接幅（xhi - xlo）で割ると、帯の中で最も広い行が分母を決める。
  // 胴と脚の境目にかかる帯では、脚の幅で胴の奥行きを割ることになり、比が
  // 小さく出て「問題なし」と判定されてしまう（自分のテストで、境目に接する
  // 帯が素通りし、胴の下半分が 1.55 のまま残るのを見つけた）。
  //
  // ならすのは**5 行の中央値**まで。マットの毛羽立ちは消えるが、シルエットが
  // 本当に広がる所は広がったままになる。ここを移動平均にすると、幅の階段が
  // 前後 ±(帯の高さ/2) 行へにじみ、境目の手前が素通りして同じ穴が残る
  // （胴の下半分が 1.46 のまま残った）。
  const rowWidth = new Float32Array(height);
  for (let y = y0; y <= y1; y++) {
    let xlo = width;
    let xhi = -1;
    for (let x = 0; x < width; x++) {
      if ((alpha[y * width + x] as number) < SUBJECT_THRESHOLD) continue;
      if (x < xlo) xlo = x;
      if (x > xhi) xhi = x;
    }
    rowWidth[y] = xhi >= xlo ? xhi - xlo + 1 : 0;
  }
  const rowWidthS = new Float32Array(height);
  for (let y = y0; y <= y1; y++) {
    const w: number[] = [];
    for (let d = -2; d <= 2; d++) {
      const yy = y + d;
      if (yy < y0 || yy > y1) continue;
      const v = rowWidth[yy] as number;
      if (v > 0) w.push(v);
    }
    w.sort((a, b) => a - b);
    rowWidthS[y] = w.length > 0 ? (w[w.length >> 1] as number) : (rowWidth[y] as number);
  }

  // 帯ごとの中央値と広がり。帯は半分ずつ重ねて、境目を作らない。
  const centers: number[] = [];
  const meds: number[] = [];
  const spreads: number[] = [];
  for (let b = 0; ; b++) {
    const bs = y0 + Math.round((b * bandH) / 2);
    const be = Math.min(y1, bs + bandH - 1);
    if (bs > y1) break;
    const zs: number[] = [];
    for (let y = bs; y <= be; y++) {
      for (let x = 0; x < width; x++) {
        const i = y * width + x;
        if ((alpha[i] as number) < SUBJECT_THRESHOLD) continue;
        const v = z[i] as number;
        if (Number.isFinite(v)) zs.push(v);
      }
    }
    centers.push((bs + be) / 2);
    // 分位を取るのに必要な最低限。細い被写体（棒・腕だけ）でも帯あたり
    // 十数画素は残るので、ここを大きくすると帯が丸ごと素通りする。
    if (zs.length < 16) {
      meds.push(NaN);
      spreads.push(NaN);
      if (be >= y1) break;
      continue;
    }
    zs.sort((a, c) => a - c);
    const q = (f: number): number => zs[Math.floor(f * (zs.length - 1))] as number;
    meds.push(q(0.5));
    spreads.push(q(0.98) - q(0.02));
    if (be >= y1) break;
  }

  // 平滑化（3点移動平均）。急に値が変わると帯の境目が段差になる。
  const smooth = (a: number[]): number[] =>
    a.map((_, i) => {
      let sum = 0;
      let n = 0;
      for (let d = -1; d <= 1; d++) {
        const v = a[i + d];
        if (v !== undefined && Number.isFinite(v)) {
          sum += v;
          n++;
        }
      }
      return n > 0 ? sum / n : (a[i] as number);
    });
  const sS = smooth(spreads);
  const mS = smooth(meds);

  /** 行 y での値を、帯の中心どうしで線形に補間する。 */
  const at = (arr: number[], y: number, fallback: number): number => {
    if (arr.length === 0) return fallback;
    if (y <= (centers[0] as number)) return Number.isFinite(arr[0] as number) ? (arr[0] as number) : fallback;
    const last = arr.length - 1;
    if (y >= (centers[last] as number)) {
      return Number.isFinite(arr[last] as number) ? (arr[last] as number) : fallback;
    }
    for (let i = 1; i < arr.length; i++) {
      const c0 = centers[i - 1] as number;
      const c1 = centers[i] as number;
      if (y <= c1) {
        const a0 = arr[i - 1] as number;
        const a1 = arr[i] as number;
        if (!Number.isFinite(a0) || !Number.isFinite(a1)) return fallback;
        const t = c1 > c0 ? (y - c0) / (c1 - c0) : 0;
        return a0 + (a1 - a0) * t;
      }
    }
    return fallback;
  };

  // 行ごとの倍率。分子（広がり）は帯から、分母（幅）はその行から取る。
  //
  // ここで k を y 方向に平滑化してはいけない。分子はすでに帯で均されていて
  // 滑らかなので、k が急に変わるのは**シルエットの幅が本当に段になる所**
  // だけである。そこを均すと、上の rowWidthS を移動平均にしたのと同じで、
  // 段の手前が素通りする。
  const ks = new Float32Array(height).fill(1);
  let touched = false;
  for (let y = y0; y <= y1; y++) {
    const m = at(mS, y, NaN);
    const spread = at(sS, y, NaN);
    const w = rowWidthS[y] as number;
    if (!Number.isFinite(m) || !Number.isFinite(spread) || !(w > 0)) continue;
    const worldWidth = (w / focalPx) * Math.max(m, 1e-6);
    const ratio = worldWidth > 0 ? spread / worldWidth : 0;
    if (ratio > maxRatio) {
      ks[y] = maxRatio / ratio;
      touched = true;
    }
  }
  if (!touched) return z;

  const out = new Float32Array(z);
  for (let y = y0; y <= y1; y++) {
    const k = ks[y] as number;
    if (k >= 0.999) continue;
    const m = at(mS, y, NaN);
    if (!Number.isFinite(m)) continue;
    for (let x = 0; x < width; x++) {
      const i = y * width + x;
      if ((alpha[i] as number) < SUBJECT_THRESHOLD) continue;
      out[i] = m + ((z[i] as number) - m) * k;
    }
  }
  return out;
}

/** 被写体内の頑健な奥行き幅（p1..p99）。強調の前後で比べるのに使う。 */
function subjectExtent(values: ArrayLike<number>, alpha: ArrayLike<number>): number {
  const [lo, hi] = percentileOfSubject(values, alpha, [0.01, 0.99]) as [number, number];
  return hi - lo;
}

/** 被写体領域のパーセンタイル値を返す。マットの縁の外れ値を避けるため。 */
function percentileOfSubject(
  values: ArrayLike<number>,
  alpha: ArrayLike<number>,
  qs: readonly number[],
): number[] {
  const picked: number[] = [];
  for (let i = 0; i < values.length; i++) {
    if ((alpha[i] as number) >= SUBJECT_THRESHOLD) {
      const v = values[i] as number;
      if (Number.isFinite(v)) picked.push(v);
    }
  }
  if (picked.length === 0) return qs.map(() => 0);
  picked.sort((a, b) => a - b);
  return qs.map((q) => picked[Math.min(picked.length - 1, Math.max(0, Math.round(q * (picked.length - 1))))] as number);
}

/** 被写体のバウンディングボックスから短辺の画素数を得る。 */
export function subjectShortSide(alpha: ArrayLike<number>, width: number, height: number): number {
  let minX = width;
  let minY = height;
  let maxX = -1;
  let maxY = -1;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      if ((alpha[y * width + x] as number) >= SUBJECT_THRESHOLD) {
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
      }
    }
  }
  if (maxX < 0) return 0;
  return Math.min(maxX - minX + 1, maxY - minY + 1);
}

/**
 * 深度マップから**外向き**法線を求める（5×5 の平面フィット）。
 *
 * 単純な中央差分はノイズに弱い。また深度の不連続をまたぐ画素を混ぜると
 * 物体の縁で法線が横倒しになり、リムが不自然に光る。
 *
 * 向きの規約: カメラを原点、視線を +z としたとき、**手前を向いた面の法線は
 * −z 成分を持つ**（＝視点の側を向く）。描画の背面カリングが
 * `dot(n, eye − pos) > 閾値` で判定するので、この向きでなければならない。
 *
 * 導出: 表面点 P(x,y) = (u·z, v·z, z)（u=(x−cx)/f, v=(y−cy)/f）の
 * 接ベクトルの外積から n ∝ (z_x·f, z_y·f, −(z + (x−cx)z_x + (y−cy)z_y))。
 */
export function estimateNormals(
  depth: Float32Array,
  width: number,
  height: number,
  focalPx: number,
  discontinuity: number,
): Float32Array {
  const out = new Float32Array(width * height * 3);
  const R = 2;

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const i = y * width + x;
      const z0 = depth[i] as number;

      // 局所的に z = a·x + b·y + c をフィットする（重み付き最小二乗の正規方程式）
      let sxx = 0, sxy = 0, syy = 0, sxz = 0, syz = 0, sx = 0, sy = 0, sz = 0, sw = 0;
      for (let dy = -R; dy <= R; dy++) {
        const yy = y + dy;
        if (yy < 0 || yy >= height) continue;
        for (let dx = -R; dx <= R; dx++) {
          const xx = x + dx;
          if (xx < 0 || xx >= width) continue;
          const z = depth[yy * width + xx] as number;
          // 深度の不連続をまたぐ画素はフィットから外す
          if (Math.abs(z - z0) > discontinuity) continue;
          sxx += dx * dx; sxy += dx * dy; syy += dy * dy;
          sxz += dx * z; syz += dy * z;
          sx += dx; sy += dy; sz += z; sw += 1;
        }
      }
      if (sw < 4) {
        out[i * 3] = 0; out[i * 3 + 1] = 0; out[i * 3 + 2] = -1;
        continue;
      }
      // 中心化して 2×2 の連立方程式を解く
      const mxx = sxx - (sx * sx) / sw;
      const mxy = sxy - (sx * sy) / sw;
      const myy = syy - (sy * sy) / sw;
      const mxz = sxz - (sx * sz) / sw;
      const myz = syz - (sy * sz) / sw;
      const det = mxx * myy - mxy * mxy;
      let dzdx = 0;
      let dzdy = 0;
      if (Math.abs(det) > 1e-12) {
        dzdx = (myy * mxz - mxy * myz) / det;
        dzdy = (mxx * myz - mxy * mxz) / det;
      }
      // 画面上の勾配を3D法線に直す。透視投影なので焦点距離が要る。
      const nx = dzdx * focalPx;
      const ny = dzdy * focalPx;
      const nz = -(z0 + (x - width / 2) * dzdx + (y - height / 2) * dzdy);
      const len = Math.hypot(nx, ny, nz) || 1;
      out[i * 3] = nx / len;
      out[i * 3 + 1] = ny / len;
      out[i * 3 + 2] = nz / len;
    }
  }
  return out;
}

/**
 * シルエット帯（α 境界から内側 inner〜outer px）の画素インデックスを集める。
 *
 * v2 の設計ではシフト推定に使う想定だったが、その推定は不要になった（solveAffine）。
 * いまはリム処理（docs/03 §3.6.4）で境界付近の α とスケールを調整するのに使う。
 */
export function silhouetteBand(
  alpha: ArrayLike<number>,
  width: number,
  height: number,
  inner = 3,
  outer = 8,
): Int32Array {
  const dist = distanceTransform(alpha, width, height, (v) => v >= SUBJECT_THRESHOLD);
  const picked: number[] = [];
  for (let i = 0; i < dist.length; i++) {
    const d = dist[i] as number;
    if (d >= inner && d <= outer) picked.push(i);
  }
  return Int32Array.from(picked);
}

/**
 * 逆深度 → 深度の変換係数 (a, b) を**閉形式で解く**（z = 1/(a·d + b)）。
 *
 * ## 設計（docs/03 §3.5.2）からの変更
 *
 * v2 の設計は「シルエット帯で Σ(n·v)² を最小化して b を推定する」としていたが、
 * 実装して検算したところ **この目的関数は b を同定できない**。
 * コストは b が小さい（＝被写体が膨らむ）ほど単調に下がり続け、内点最小値を持たない。
 * 奥行き幅を固定しても単調性は解消しなかった。
 *
 * さらに測ってみると、**奥行き幅を固定した時点で b は見た目をほとんど変えない**。
 * 実行可能な b の全域で、形の差は奥行き幅の 5% 以内に収まる。
 * つまり設計が心配していた「b が形そのものを歪める」という問題は、
 * 幅（＝UI の「立体感」スライダ）を決めた時点で解消している。
 *
 * そこで最適化をやめ、次の2つの制約から a と b を直接解く。
 *
 *   (1) 被写体の深度の中央値を 1.0 に置く       →  a·d50 + b = 1
 *   (2) 被写体の深度幅を目標値 S にする         →  z(d5) − z(d95) = S
 *
 * これは a についての2次方程式になる。
 *
 *   S·k·a² + (S·m − Δ)·a + S = 0
 *     m = d5 + d95 − 2·d50,  k = (d5 − d50)(d95 − d50),  Δ = d95 − d5
 *
 * 合成球で検算すると真の (a, b) を誤差 0% で復元する。
 * 最適化（法線マップを20回作り直す、約60ms）が丸ごと不要になった。
 */
export function solveAffine(
  d5: number,
  d50: number,
  d95: number,
  targetSpan: number,
): { a: number; b: number } | null {
  const delta = d95 - d5;
  if (!(delta > 0) || !(targetSpan > 0)) return null;

  const m = d5 + d95 - 2 * d50;
  const k = (d5 - d50) * (d95 - d50);
  const A2 = targetSpan * k;
  const A1 = targetSpan * m - delta;
  const A0 = targetSpan;

  const candidates: number[] = [];
  if (Math.abs(A2) < 1e-15) {
    if (Math.abs(A1) > 1e-15) candidates.push(-A0 / A1);
  } else {
    const disc = A1 * A1 - 4 * A2 * A0;
    if (disc < 0) return null;
    const root = Math.sqrt(disc);
    candidates.push((-A1 + root) / (2 * A2), (-A1 - root) / (2 * A2));
  }

  for (const a of candidates) {
    if (!(a > 0) || !Number.isFinite(a)) continue;
    const b = 1 - a * d50;
    // 被写体の全域で a·d + b > 0（＝深度が正）でなければならない
    if (a * d5 + b > 1e-6 && a * d95 + b > 1e-6) return { a, b };
  }
  return null;
}

/**
 * 被写体マスクの内側だけで箱平均を取る。
 *
 * マスクの外（背景）は平均に入れない。入れると輪郭付近で背景の深度に
 * 引っ張られ、そこだけ起伏が消える。積分画像なので半径によらず O(n)。
 */
function maskedBoxMean(
  values: ArrayLike<number>,
  mask: ArrayLike<number>,
  width: number,
  height: number,
  radius: number,
): Float32Array {
  const w1 = width + 1;
  const sumA = new Float64Array(w1 * (height + 1));
  const sumM = new Float64Array(w1 * (height + 1));
  for (let y = 0; y < height; y++) {
    let rowA = 0;
    let rowM = 0;
    for (let x = 0; x < width; x++) {
      const i = y * width + x;
      const inside = (mask[i] as number) >= SUBJECT_THRESHOLD ? 1 : 0;
      rowA += inside ? (values[i] as number) : 0;
      rowM += inside;
      sumA[(y + 1) * w1 + x + 1] = (sumA[y * w1 + x + 1] as number) + rowA;
      sumM[(y + 1) * w1 + x + 1] = (sumM[y * w1 + x + 1] as number) + rowM;
    }
  }

  const out = new Float32Array(width * height);
  for (let y = 0; y < height; y++) {
    const y0 = Math.max(0, y - radius);
    const y1 = Math.min(height, y + radius + 1);
    for (let x = 0; x < width; x++) {
      const x0 = Math.max(0, x - radius);
      const x1 = Math.min(width, x + radius + 1);
      const a =
        (sumA[y1 * w1 + x1] as number) - (sumA[y0 * w1 + x1] as number) -
        (sumA[y1 * w1 + x0] as number) + (sumA[y0 * w1 + x0] as number);
      const m =
        (sumM[y1 * w1 + x1] as number) - (sumM[y0 * w1 + x1] as number) -
        (sumM[y1 * w1 + x0] as number) + (sumM[y0 * w1 + x0] as number);
      out[y * width + x] = m > 0 ? a / m : (values[y * width + x] as number);
    }
  }
  return out;
}

/**
 * 中央値絶対偏差（MAD）から外れ値に強い範囲を求める。
 *
 * 分位（p1..p99）だけでは足りない。外れ値が全体の 1% を超えると
 * 分位そのものが外れ値の中に入ってしまうためで、実際に髪や
 * マットの染み出しは被写体の数%を占める。MAD は外れ値が半数に
 * 達するまで中央値がずれないので、こちらのほうが素直に効く。
 *
 * 正規分布なら 1.4826·MAD が標準偏差に一致するので、その k 倍を取る。
 */
export function robustRange(
  values: ArrayLike<number>,
  alpha: ArrayLike<number>,
  k = 3,
): [number, number] {
  const picked: number[] = [];
  for (let i = 0; i < values.length; i++) {
    if ((alpha[i] as number) < SUBJECT_THRESHOLD) continue;
    const v = values[i] as number;
    if (Number.isFinite(v)) picked.push(v);
  }
  if (picked.length === 0) return [0, 1];
  picked.sort((a, b) => a - b);
  const med = picked[Math.floor(picked.length / 2)] as number;

  const dev: number[] = new Array(picked.length);
  for (let i = 0; i < picked.length; i++) dev[i] = Math.abs((picked[i] as number) - med);
  dev.sort((a, b) => a - b);
  const mad = dev[Math.floor(dev.length / 2)] as number;
  const sigma = mad * 1.4826;

  // MAD が 0（値がほぼ一定）のときは分位に任せる
  const lo = picked[0] as number;
  const hi = picked[picked.length - 1] as number;
  if (!(sigma > 0)) return [lo, hi];
  return [Math.max(lo, med - k * sigma), Math.min(hi, med + k * sigma)];
}

/** 被写体の外接矩形の高さ（画素）。奥行きとの比を見るのに使う。 */
export function subjectBoxHeight(alpha: ArrayLike<number>, width: number, height: number): number {
  let y0 = height;
  let y1 = -1;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      if ((alpha[y * width + x] as number) >= SUBJECT_THRESHOLD) {
        if (y < y0) y0 = y;
        if (y > y1) y1 = y;
        break;
      }
    }
  }
  return y1 < 0 ? 0 : y1 - y0 + 1;
}

/**
 * 局所的な起伏を強調する（アンシャープマスク）。
 *
 * 大域の形は平滑化した成分がそのまま持つので、比率は変わらない。
 * 持ち上げるのは「平滑化からのずれ」＝顔の凹凸や服のしわだけ。
 */
export function enhanceRelief(
  z: Float32Array,
  alpha: ArrayLike<number>,
  width: number,
  height: number,
  radius: number,
  boost: number,
): Float32Array {
  if (boost <= 1 || radius < 1) return z;

  // 帯域を絞って持ち上げる。
  //
  // 素朴なアンシャープマスク（z − 大半径の平滑化）だと、**画素ごとの
  // ノイズまで一緒に boost 倍される**。深度は 518² から 1024² へ引き伸ばして
  // いるうえ、量子化と融合のむらも乗るので、最細部はほとんどノイズである。
  // 実写で測ると、boost=3 は隣接画素の差（p90）を 0.024 → 0.046 と倍にし、
  // その結果スカートの判定閾値（0.02）を普通の顔の傾斜が超えて、
  // スカートが顔じゅうに林立した（全体の 21% → 48%）。
  //
  // そこで 3 つの帯に分ける。
  //   ・大半径より粗い成分 = 大域の形    → そのまま（比率を保つ）
  //   ・小半径〜大半径の帯 = 顔の凹凸    → boost 倍する
  //   ・小半径より細かい成分 = ほぼノイズ → そのまま（増幅しない）
  const outer = Math.max(1, Math.round(radius));
  // 内側の半径には**絶対的な下限**を置く。ノイズは画素ごとに乗るので、
  // その尺度は被写体の大きさに依らない。半径に比例させるだけだと、
  // 被写体が小さいときに内側が 1〜2px になってノイズを均せず、
  // 帯にノイズが残ったまま boost 倍されてしまう。
  const inner = Math.min(outer, Math.max(3, Math.round(radius / 8)));
  const base = maskedBoxMean(z, alpha, width, height, outer);
  const fine = inner < outer ? maskedBoxMean(z, alpha, width, height, inner) : z;

  const out = new Float32Array(z.length);
  for (let i = 0; i < z.length; i++) {
    if ((alpha[i] as number) < SUBJECT_THRESHOLD) {
      out[i] = z[i] as number;
      continue;
    }
    const b = base[i] as number;
    const f = fine[i] as number;
    out[i] = b + (f - b) * boost + ((z[i] as number) - f);
  }
  return out;
}

/**
 * 深度を較正して 0..65535 の正規化深度にする。
 *
 * 半球カバー（決定 D2）では絶対スケールは意味を持たない。重要なのは
 * 「被写体の奥行きが幅に対してどの程度あるか」という比だけで、それを直接指定する。
 */
export function calibrate(input: CalibrationInput): CalibrationResult {
  const { raw, width, height, alpha, kind, focalPx } = input;
  const boost = input.reliefBoost ?? DEFAULT_RELIEF_BOOST;
  const reliefRadius = input.reliefRadius ?? DEFAULT_RELIEF_RADIUS;

  const shortSide = subjectShortSide(alpha, width, height);
  if (shortSide === 0) {
    throw new Error('被写体が見つかりません（α が閾値を超える画素がない）');
  }
  const boxHeight = subjectBoxHeight(alpha, width, height) || shortSide;

  // --- ① 外れ値を先に切る
  //
  // 髪やマットの染み出しは、被写体の 5% ほどの画素で深度の裾を長く伸ばす。
  // その裾が奥行きの範囲を決めてしまうと、被写体の本体はそのごく一部に
  // 押し込められ、平らに見える。実測では p95→p99.5 の裾だけで奥行きの
  // 37% を占めていた。範囲を p1..p99 で取り、外はそこへ丸める。
  const [q1, q50, q99] = percentileOfSubject(raw, alpha, [0.01, 0.5, 0.99]) as [
    number, number, number,
  ];
  const [m1, m99] = robustRange(raw, alpha, 3);
  // 分位と MAD の**厳しいほう**を採る。外れ値が 1% を超えると分位は
  // 外れ値の中に入ってしまい、そこだけでは切りきれない。
  const clipLo = Math.max(q1, m1);
  const clipHi = Math.min(q99, m99);
  const clipped = new Float32Array(raw.length);
  for (let i = 0; i < raw.length; i++) {
    const v = raw[i] as number;
    clipped[i] = Number.isFinite(v) ? Math.min(clipHi, Math.max(clipLo, v)) : q50;
  }

  // --- ② 深度に直す
  let z: Float32Array;
  let shift = 0;
  let metric = false;

  if (kind === 'depth') {
    if (input.depthToWidthRatio === undefined) {
      // DA3 は実寸の深度を返す（intrinsics も一緒に返る）。素直にそれを使う。
      // 比を決め打ちで引き伸ばすと、大域の形が人間の比率から外れる。
      // 実測では 奥行き ÷ 高さ が 1.47 になり、髪が後ろへ長く尾を引いていた。
      z = clipped;
      metric = true;
    } else {
      const target = (shortSide / focalPx) * input.depthToWidthRatio;
      const span = Math.max(q99 - q1, 1e-9);
      const scale = target / span;
      z = new Float32Array(raw.length);
      for (let i = 0; i < raw.length; i++) z[i] = ((clipped[i] as number) - q50) * scale + 1.0;
    }
  } else {
    // V2 系は逆深度で絶対スケールを持たない。比を指定するしかない。
    const ratio = input.depthToWidthRatio ?? DEFAULT_RATIO;
    const targetSpan = (shortSide / focalPx) * ratio;
    const [p5, p50, p95] = percentileOfSubject(clipped, alpha, [0.05, 0.5, 0.95]) as [
      number, number, number,
    ];
    const solved = solveAffine(p5, p50, p95, targetSpan);
    if (!solved) {
      throw new Error(
        `逆深度の較正に失敗しました（分位 ${p5.toFixed(4)}/${p50.toFixed(4)}/${p95.toFixed(4)}、` +
          `目標幅 ${targetSpan.toFixed(4)}）。立体感を下げると解ける場合があります。`,
      );
    }
    shift = solved.b;
    z = new Float32Array(raw.length);
    for (let i = 0; i < raw.length; i++) {
      const den = solved.a * (clipped[i] as number) + solved.b;
      z[i] = den > 1e-6 ? 1 / den : 0;
    }
  }

  // --- ③ 局所の起伏を持ち上げる
  //
  // 上がるのは顔の凹凸や服のしわだけ。大域の形（＝体の奥行きの比率）は
  // 変えない。
  const radius = Math.max(1, Math.round(shortSide * reliefRadius));
  const beforeBoost = subjectExtent(z, alpha);
  z = enhanceRelief(z, alpha, width, height, radius, boost);

  // 強調は「平滑化からのずれ」を倍にするので、切り残した外れ値も倍になる。
  // クリップの縁に張り付いた画素が、強調後に大きく飛び出す。もう一度
  // 頑健な範囲で抑える（自分のテストで、外れ値が奥行きを 12.7 倍に
  // 広げているのを見つけた）。
  if (boost > 1) {
    const [bLo, bHi] = robustRange(z, alpha, 3);
    for (let i = 0; i < z.length; i++) {
      const v = z[i] as number;
      z[i] = v < bLo ? bLo : v > bHi ? bHi : v;
    }

    // 強調のぶん広がった奥行きを、元の幅へ戻す。
    //
    // 「大域は平滑化成分が持つので比率は変わらない」と書いていたが、実測
    // では変わっていた。持ち上げた帯は極値の側にも足されるので、全体の幅が
    // 広がる。実写では 奥行き÷身長 が 0.539 → 0.721（+34%）になっていた。
    // 体が奥へ伸びると、少し回しただけで串のように崩れる。
    //
    // 幅を戻すと局所の起伏も同じ率で縮むが、**まわりに対する比**は上がった
    // ままである。欲しいのはその比であって、絶対の奥行きではない。
    const afterBoost = subjectExtent(z, alpha);
    if (afterBoost > 1e-9 && beforeBoost > 1e-9) {
      const k = beforeBoost / afterBoost;
      if (k < 1) {
        const mid = percentileOfSubject(z, alpha, [0.5])[0] as number;
        for (let i = 0; i < z.length; i++) z[i] = ((z[i] as number) - mid) * k + mid;
      }
    }
  }

  // --- ④ 実寸が当てにならないときだけ引き戻す
  //
  // 局所強調の**後**に行う。強調は奥行きの幅も少し広げるので、先に
  // クランプしても最後には範囲を超えてしまう（自分のテストで見つけた）。
  //
  // 効かせるのは**実寸経路だけ**。depthToWidthRatio が渡されている
  // ときは、呼び出し側が奥行きを明示していて、②で目標幅ぴったりに
  // 引き伸ばし済み。そこへこのクランプを重ねると、比を上げても
  // 上限 0.9 で頭打ちになり、立体感スライダが効かなくなる。
  const medianZ = percentileOfSubject(z, alpha, [0.5])[0] as number;
  const worldHeight = (boxHeight / focalPx) * Math.max(medianZ, 1e-6);
  const worldWidth = (shortSide / focalPx) * Math.max(medianZ, 1e-6);
  // 外れ値で範囲が決まらないよう、頑健な幅で見る。
  let [lo, hi] = percentileOfSubject(z, alpha, [0.01, 0.99]) as [number, number];
  const widthRatioNow = worldWidth > 0 ? (hi - lo) / worldWidth : 0;

  if (metric && widthRatioNow > 0) {
    const clamped = Math.min(
      PLAUSIBLE_DEPTH_TO_WIDTH.max,
      Math.max(PLAUSIBLE_DEPTH_TO_WIDTH.min, widthRatioNow),
    );
    if (clamped !== widthRatioNow) {
      const k = clamped / widthRatioNow;
      for (let i = 0; i < z.length; i++) z[i] = ((z[i] as number) - medianZ) * k + medianZ;
      [lo, hi] = percentileOfSubject(z, alpha, [0.01, 0.99]) as [number, number];
    }
  }

  // --- ④.5 人体としてあり得ない帯だけ潰す
  //
  // ④の全体クランプでは捕まらない壊れ方がある。座位では脚が幅を決めるので
  // 全体の比は合ってしまい、胴だけが 8 倍深いまま通る。
  //
  // **④より後**に置くこと。先に潰すと全体の比が下限 0.40 を割り、④が
  // 全体を拡大し直して帯の圧縮を打ち消す（自分のテストで、胴の比が
  // 2.40 → 3.06 と**悪化**しているのを見つけた）。帯を潰した結果として
  // 全体が薄くなるのは意図どおりなので、下限に引っかけてはいけない。
  if (metric) {
    z = flattenImplausibleBands(
      z,
      alpha,
      width,
      height,
      focalPx,
      input.bandMaxRatio ?? PLAUSIBLE_BAND_DEPTH_TO_WIDTH,
    );
    [lo, hi] = percentileOfSubject(z, alpha, [0.01, 0.99]) as [number, number];
  }

  // --- ⑤ 0..65535 に写す
  const [zLo, zHi] = percentileOfSubject(z, alpha, [0.0, 1.0]) as [number, number];
  const nearZ = Math.min(zLo, zHi);
  const farZ = Math.max(zLo, zHi);
  const span = Math.max(farZ - nearZ, 1e-9);

  const depth = new Uint16Array(raw.length);
  for (let i = 0; i < raw.length; i++) {
    const t = ((z[i] as number) - nearZ) / span;
    depth[i] = Math.max(0, Math.min(65535, Math.round(t * 65535)));
  }
  return {
    depth,
    nearZ,
    farZ,
    shift,
    silhouetteSamples: 0,
    metric,
    depthToHeight: worldHeight > 0 ? (hi - lo) / worldHeight : 0,
    depthToWidth: worldWidth > 0 ? (hi - lo) / worldWidth : 0,
  };
}

/**
 * シルエット近傍の画素の深度を、内側の値で置き換える（docs/03 §3.5.3(c)）。
 *
 * 単眼深度モデルは物体の縁で前景と背景を混ぜた値を返す。にじみは
 * α の軟化部分（0 < α < 0.5）だけでなく、**α が 1 の内側にも数画素
 * 続く**。モデルは 518² で推論して 1024² へ引き伸ばすので、そのぶん
 * 帯も広がる。
 *
 * 実写で測ったにじみ（深度が最奥に張り付いた画素の割合）:
 *
 * | シルエットからの距離 | 1px | 2px | 4px | 6px | 8px | 12px |
 * |---|---|---|---|---|---|---|
 * | 最奥に張り付き | 56% | 45% | 28% | 14% | 6.6% | 3.0% |
 *
 * この帯を放置すると、輪郭に沿って**最奥に張り付いたサーフェルの膜**が
 * できる。正面からは体の陰に隠れて見えないが、少し回すと体の後ろへ
 * 伸びる平らな板として現れる。実写では被写体の 6.9% がこれだった。
 *
 * @param coreDistance この距離より内側を「信用できる芯」とみなす。
 *                     0 なら軟化 α の置き換えだけを行う（従来の挙動）。
 */
export function pullBoundaryDepthInward(
  depth: Uint16Array,
  alpha: ArrayLike<number>,
  width: number,
  height: number,
  coreDistance = 0,
): Uint16Array {
  const out = new Uint16Array(depth);

  // (a) 軟化境界（0 < α < 0.5）は、最も近い「α が立っている」画素から採る。
  const nearestSubject = nearestForegroundIndex(alpha, width, height, (v) => v >= SUBJECT_THRESHOLD);
  for (let i = 0; i < depth.length; i++) {
    const a = alpha[i] as number;
    if (a > 0 && a < SUBJECT_THRESHOLD) {
      const src = nearestSubject[i] as number;
      if (src >= 0) out[i] = depth[src] as number;
    }
  }
  if (coreDistance <= 0) return out;

  // (b) α が立っていても、シルエットから coreDistance 以内はにじみの帯。
  //     そこより内側の「芯」の値で置き換える。
  const dist = distanceTransform(alpha, width, height, (v) => v >= SUBJECT_THRESHOLD);
  const core = new Uint8Array(width * height);
  let coreCount = 0;
  for (let i = 0; i < core.length; i++) {
    if ((alpha[i] as number) >= SUBJECT_THRESHOLD && (dist[i] as number) >= coreDistance) {
      core[i] = 1;
      coreCount++;
    }
  }
  // 芯が残らないほど細い被写体では、置き換える先が無い。触らない。
  if (coreCount === 0) return out;

  const nearestCore = nearestForegroundIndex(core, width, height, (v) => v !== 0);
  const pulled = new Uint16Array(out);
  for (let i = 0; i < depth.length; i++) {
    if ((alpha[i] as number) < SUBJECT_THRESHOLD) continue;
    if ((dist[i] as number) >= coreDistance) continue;
    const src = nearestCore[i] as number;
    if (src >= 0) pulled[i] = out[src] as number;
  }

  // (c) 軟化境界は、引き込んだあとの値でもう一度採り直す。
  //     (a) で拾った先が帯の中だと、にじんだ値のままになる。
  for (let i = 0; i < depth.length; i++) {
    const a = alpha[i] as number;
    if (a > 0 && a < SUBJECT_THRESHOLD) {
      const src = nearestSubject[i] as number;
      if (src >= 0) pulled[i] = pulled[src] as number;
    }
  }
  return pulled;
}
