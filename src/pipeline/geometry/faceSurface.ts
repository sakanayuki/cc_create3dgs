/**
 * 顔の立体を landmark から起こす（docs/03 §3.4.5、v2.6）。
 *
 * **なぜ要るのか。** 一般の単眼深度モデルは顔をほぼ平らな楕円として返す。
 * 実測（同じ顔の切り出しを 518² で推論）:
 *
 * | モデル | 相対起伏 | 鼻・眼窩・唇 |
 * |---|---|---|
 * | DA3-Small (0.08B) | 0.153 | 出ない |
 * | DA3-Base (0.12B) | 0.190 | 出ない |
 *
 * 容量を 1.5 倍にしても構造は出ない。入力解像度も効かない（顔だけを
 * 64 / 113 / 227px で渡して 0.163 / 0.226 / 0.167 と単調ですらない）。
 * シーン規模の学習目標に対して顔の 2cm の凹凸が小さすぎるためで、
 * **一般の深度モデルを大きくしても解けない**。
 *
 * そこで顔専用の landmark モデル（478 点の 3D 座標）から面を起こし、
 * 深度の**細部の帯だけ**を差し替える。大域（頭がどこにあるか）は
 * これまでどおり深度モデルが持つ。実測で鼻先は頬より 32px 手前、
 * 顔の奥行き幅は顔幅の 0.85 倍と、人の顔として妥当な値が出る。
 *
 * ここに置くのは純粋な計算だけ。モデルの実行は generate.ts が持つ。
 */
import type { Rect } from '../0-preprocess';

/** 画像座標での landmark（z は同じ尺度の相対値。小さいほど手前）。 */
export interface FaceLandmark {
  readonly x: number;
  readonly y: number;
  readonly z: number;
}

/** α がこの値以上を被写体とみなす。 */
const SUBJECT_THRESHOLD = 128;

/** 重みが 1 のままの楕円の内側（楕円半径に対する比）。 */
const WEIGHT_CORE = 0.75;
/** 重みが 0 になる楕円（同）。CORE との差が羽根の幅になる。 */
const WEIGHT_EDGE = 1.1;
/**
 * 「大域」と「細部」の境目を、顔の楕円半径の何倍の平均で切るか。
 *
 * 0.5 なら顔の半径ほどの箱平均。鼻（長さは顔の高さの 1/3 ほど）・眉・
 * 眼窩・唇はこれより細かいので細部に残り、顔全体の丸みだけが落ちる。
 */
const TREND_RADIUS_RATIO = 0.5;

/** Shepard 補間で近傍とみなす半径（点間隔の何倍か）。 */
const CUTOFF_SPACINGS = 2.5;
/** これだけ点が入るまでは半径を広げる。 */
const MIN_NEIGHBOURS = 6;
/**
 * 顔の傾きを直す上限。**四角の幅いっぱいで動かせる量 ÷ 四角の幅（実寸）**。
 *
 * 0.5 なら角度にして ±26.6°。実写で直したい量は 0.30（±16.7°）だったので、
 * 6 割ほど余裕がある。landmark が壊れたときに顔がひっくり返らない程度に
 * 抑えつつ、実際に要る量は通す。
 */
const MAX_TILT_RATIO = 0.5;

/**
 * マットから顔の初期位置を当てる（頭の外接四角、正方）。
 *
 * 顔検出モデルは載せない。landmark モデルは「だいたい顔が入っていれば」
 * 動き、その出力で切り直して 2 回目を回せば収束する（実測でモデルの
 * 自己申告スコアが 9.7 → 13.7）。MediaPipe が動画でやっている
 * 「検出器 → landmark → 次フレームは landmark から切る」の、
 * 検出器を**マットで代用**した形である。
 *
 * 直立した人物を仮定する。仰向けや逆さでは外れるが、そのときは
 * landmark モデルのスコアが低く出るので呼び出し側が捨てられる。
 *
 * @returns 頭を含む正方形。被写体が無ければ null。
 */
export function headBoxFromMatte(
  alpha: ArrayLike<number>,
  width: number,
  height: number,
): Rect | null {
  let top = -1;
  let bottom = -1;
  const rowLo = new Int32Array(height).fill(-1);
  const rowHi = new Int32Array(height).fill(-1);
  for (let y = 0; y < height; y++) {
    let lo = -1;
    let hi = -1;
    for (let x = 0; x < width; x++) {
      if ((alpha[y * width + x] as number) < SUBJECT_THRESHOLD) continue;
      if (lo < 0) lo = x;
      hi = x;
    }
    rowLo[y] = lo;
    rowHi[y] = hi;
    if (lo >= 0) {
      if (top < 0) top = y;
      bottom = y;
    }
  }
  if (top < 0) return null;

  // 頭の下端 = 首。上から見て幅がいったん狭まり、その下で広がる行を探す。
  const subjectHeight = bottom - top + 1;
  const searchTo = Math.min(bottom, top + Math.round(subjectHeight * 0.45));
  const widthAt = (y: number): number =>
    (rowLo[y] as number) < 0 ? 0 : (rowHi[y] as number) - (rowLo[y] as number) + 1;
  const smooth = Math.max(1, Math.round(subjectHeight * 0.01));
  const w = (y: number): number => {
    let sum = 0;
    let n = 0;
    for (let d = -smooth; d <= smooth; d++) {
      const yy = y + d;
      if (yy < top || yy > bottom) continue;
      sum += widthAt(yy);
      n++;
    }
    return n > 0 ? sum / n : 0;
  };

  let neck = -1;
  const from = top + Math.round(subjectHeight * 0.05);
  const step = Math.max(2, Math.round(subjectHeight * 0.02));
  for (let y = from; y < searchTo - step; y++) {
    const here = w(y);
    if (here <= 0) continue;
    // 前後より狭く、少し下で明らかに広がる = 首
    if (here <= w(y - step) && here <= w(y + step) && w(y + 2 * step) > here * 1.25) {
      neck = y;
      break;
    }
  }
  // 見つからなければ身長の 18%（直立の人物の頭のおよその割合）で切る。
  if (neck < 0) neck = top + Math.round(subjectHeight * 0.18);

  let xlo = width;
  let xhi = -1;
  for (let y = top; y <= neck; y++) {
    const lo = rowLo[y] as number;
    if (lo < 0) continue;
    if (lo < xlo) xlo = lo;
    if ((rowHi[y] as number) > xhi) xhi = rowHi[y] as number;
  }
  if (xhi < xlo) return null;

  // 髪や耳まで含めた頭を、少し余裕をつけた正方で囲む。
  const cx = (xlo + xhi) / 2;
  const cy = (top + neck) / 2;
  const side = Math.round(Math.max(xhi - xlo + 1, neck - top + 1) * 1.2);
  return squareAt(cx, cy, side, width, height);
}

/** landmark の外接四角から、次の推論に渡す正方形を作る。 */
export function boxFromLandmarks(
  points: readonly FaceLandmark[],
  width: number,
  height: number,
  /** 顔の大きさに対する余裕。1.5 で顔の 1.5 倍の辺を取る。 */
  margin = 1.5,
): Rect | null {
  if (points.length === 0) return null;
  let xlo = Infinity;
  let xhi = -Infinity;
  let ylo = Infinity;
  let yhi = -Infinity;
  for (const p of points) {
    if (!Number.isFinite(p.x) || !Number.isFinite(p.y)) continue;
    if (p.x < xlo) xlo = p.x;
    if (p.x > xhi) xhi = p.x;
    if (p.y < ylo) ylo = p.y;
    if (p.y > yhi) yhi = p.y;
  }
  if (!Number.isFinite(xlo) || xhi <= xlo) return null;
  const side = Math.round(Math.max(xhi - xlo, yhi - ylo) * margin);
  return squareAt((xlo + xhi) / 2, (ylo + yhi) / 2, side, width, height);
}


/** 頭のタイルを頭の箱の何倍に広げるか。首と肩を少し含める。 */
const HEAD_TILE_MARGIN = 1.4;

/**
 * 頭だけを見る深度タイルの四角を返す（v2.6.4、docs/09 §V18）。
 *
 * 体を数枚に割るタイル（`depthTiles`）では、全身写真の顔はタイルの中の
 * 一部でしかない。実写（1116×2000、長辺を 1024 に落とすので ×0.512）で
 * 顔は作業グリッドで 140px、495px のタイルを 518² に伸ばしても
 * **146px 相当**にしかならず、DA3 は鼻も眼窩も出さない。実測では額が
 * 鼻より 4〜6mm 手前という、前後の逆転した顔になっていた。
 *
 * 頭の箱を少し広げたタイルを 1 枚足すと、同じ顔が 518² の中で
 * **260px 相当**になる。倍率にして 1.8 倍。実測で「顔の中でいちばん手前の
 * 点」が額（箱の上から 24%）から鼻（同 59%）へ移った。
 *
 * 体のタイルと同じくらいの大きさになるなら足さない。同じ絵をもう一度
 * 推論しても解像度は上がらず、融合の継ぎ目が増えるだけである。
 *
 * @param bodyTileSide 体のタイルの一辺。これに対して十分小さいときだけ足す。
 * @returns タイルの四角。頭が見つからない・意味が無いときは null。
 */
export function headDepthTile(
  alpha: ArrayLike<number>,
  width: number,
  height: number,
  bodyTileSide: number,
  margin = HEAD_TILE_MARGIN,
): Rect | null {
  const head = headBoxFromMatte(alpha, width, height);
  if (!head) return null;
  const side = Math.round(head.width * margin);
  if (side < 64) return null;
  if (side > bodyTileSide * 0.75) return null;
  return squareAt(head.x + head.width / 2, head.y + head.height / 2, side, width, height);
}

/** 中心と辺から、画像に収まる正方形を作る。 */
function squareAt(cx: number, cy: number, side: number, width: number, height: number): Rect {
  const s = Math.max(8, Math.min(width, height, side));
  const x = Math.round(Math.max(0, Math.min(width - s, cx - s / 2)));
  const y = Math.round(Math.max(0, Math.min(height - s, cy - s / 2)));
  return { x, y, width: s, height: s };
}

export interface FaceSurface {
  /** rect の大きさの深度板。小さいほど手前（深度モデルと同じ向き）。 */
  readonly depth: Float32Array;
  /** 0..1 の重み。顔の内側が 1、外へ滑らかに 0 へ落ちる。 */
  readonly weight: Float32Array;
  /** 重みが 0 でない画素の数。少なすぎるときは呼び出し側が捨てる。 */
  readonly covered: number;
  /**
   * 「大域」と「細部」を分ける半径（rect の画素）。
   *
   * この半径の平均より粗い成分は landmark 面から捨て、深度モデルのものを
   * 使う（`applyFaceRelief`）。顔の大きさから決めるので、寄りでも引きでも
   * 同じ構造（鼻・眉・眼窩・唇）が細部側に残る。
   */
  readonly trendRadius: number;
}

/**
 * 点を格子に振り分けて、半径の中の点だけを返す入れ物。
 *
 * 478 点 × 227² 画素を素直に総当たりすると 2,460 万回になる。近傍だけを
 * 見るなら格子で足りる。
 */
class PointGrid {
  private readonly cell: number;
  private readonly x0: number;
  private readonly y0: number;
  private readonly cols: number;
  private readonly rows: number;
  private readonly buckets: FaceLandmark[][];

  constructor(points: readonly FaceLandmark[], cell: number) {
    this.cell = cell;
    let xlo = Infinity;
    let ylo = Infinity;
    let xhi = -Infinity;
    let yhi = -Infinity;
    for (const p of points) {
      if (p.x < xlo) xlo = p.x;
      if (p.x > xhi) xhi = p.x;
      if (p.y < ylo) ylo = p.y;
      if (p.y > yhi) yhi = p.y;
    }
    this.x0 = xlo;
    this.y0 = ylo;
    this.cols = Math.max(1, Math.ceil((xhi - xlo) / cell) + 1);
    this.rows = Math.max(1, Math.ceil((yhi - ylo) / cell) + 1);
    this.buckets = Array.from({ length: this.cols * this.rows }, () => [] as FaceLandmark[]);
    for (const p of points) {
      const cx = Math.min(this.cols - 1, Math.max(0, Math.floor((p.x - this.x0) / cell)));
      const cy = Math.min(this.rows - 1, Math.max(0, Math.floor((p.y - this.y0) / cell)));
      (this.buckets[cy * this.cols + cx] as FaceLandmark[]).push(p);
    }
  }

  /** (x, y) から半径 r 以内に**入りうる**点を渡す。半径の確認は呼び出し側。 */
  forEach(x: number, y: number, r: number, fn: (p: FaceLandmark) => void): void {
    const lo = (v: number, o: number, n: number): number =>
      Math.min(n - 1, Math.max(0, Math.floor((v - o) / this.cell)));
    const cx0 = lo(x - r, this.x0, this.cols);
    const cx1 = lo(x + r, this.x0, this.cols);
    const cy0 = lo(y - r, this.y0, this.rows);
    const cy1 = lo(y + r, this.y0, this.rows);
    for (let cy = cy0; cy <= cy1; cy++) {
      for (let cx = cx0; cx <= cx1; cx++) {
        for (const p of this.buckets[cy * this.cols + cx] as FaceLandmark[]) fn(p);
      }
    }
  }
}

/**
 * 478 点から顔の連続面を起こす。
 *
 * 三角形分割は使わない。**必要なのは細部の帯だけ**で、大域は深度モデルが
 * 持つので、多少うねっていても最後には効かない。距離の逆数で重みをつけた
 * 散布データ補間（Shepard 法）に、点間隔ぶんの平滑化を掛ければ足りる。
 * 既定のトポロジ表（2,500 三角形）を持ち込まずに済むぶん、配信も軽い。
 *
 * **近傍だけを見る（v2.6.8、docs/09 §V23）。** 全点を見る素の Shepard 法は
 * 面を平らにしてしまう。2 次元では距離 d の点の数が d に比例して増えるので、
 * 重み 1/d² の総和は遠方が対数で効き続け、**遠くの点の平均**へ引っ張られる。
 * 実測（立ち姿、478 点、顔 227px）:
 *
 * | | z の幅 |
 * |---|---|
 * | landmark そのもの | 94.3 px |
 * | 全点を見る Shepard | **27.3 px（29%）** |
 * | 近傍だけ見る Shepard | 後述 |
 *
 * 起伏として足される量は顔の奥行きの 1.7% しかなく、**鼻も眼窩も出て
 * いなかった**。半径 `cutoff` の内側だけを見れば、点間隔の尺度の構造が残る。
 *
 * 重みは landmark の広がりから作る楕円で、外へ羽根で落とす。顔の外
 * （髪・耳・首・背景）には触らない。
 *
 * @param rect  面を置く四角（landmark と同じ画像座標）。
 * @param power 距離の逆数の指数。大きいほど各点に張り付く。
 */
export function faceDepthSurface(
  points: readonly FaceLandmark[],
  rect: Rect,
  power = 2,
): FaceSurface {
  const n = rect.width * rect.height;
  const depth = new Float32Array(n);
  const weight = new Float32Array(n);
  if (points.length < 16) return { depth, weight, covered: 0, trendRadius: 1 };

  // 顔の楕円は landmark の**外接矩形**から作る。
  //
  // 重心±σ で作ると、点の密度が偏っている側へ楕円がずれる。実測で
  // 20px ずれ、髪と背景を巻き込んだ。外接矩形なら点の密度に依らない。
  let xlo = Infinity;
  let xhi = -Infinity;
  let ylo = Infinity;
  let yhi = -Infinity;
  for (const p of points) {
    if (p.x < xlo) xlo = p.x;
    if (p.x > xhi) xhi = p.x;
    if (p.y < ylo) ylo = p.y;
    if (p.y > yhi) yhi = p.y;
  }
  const cx = (xlo + xhi) / 2;
  const cy = (ylo + yhi) / 2;
  // 楕円は外接矩形の内側に収める。landmark は顔の縁（髪の生え際・耳の前）
  // まで届くので、そこまで信用すると髪を巻き込む。
  const rx = Math.max(2, ((xhi - xlo) / 2) * 0.95);
  const ry = Math.max(2, ((yhi - ylo) / 2) * 0.95);

  // 点の平均間隔。これより細かい構造は landmark には無い。
  const spacing = Math.max(2, Math.sqrt((Math.PI * rx * ry) / points.length));
  const eps = spacing * spacing * 0.25;
  // 近傍とみなす半径。点間隔の 2.5 倍あれば、どの向きにも数点入る。
  const cutoff = spacing * CUTOFF_SPACINGS;
  const maxReach = Math.max(rx, ry) * 2;
  const grid = new PointGrid(points, Math.max(2, spacing));

  let covered = 0;
  for (let y = 0; y < rect.height; y++) {
    const gy = rect.y + y;
    for (let x = 0; x < rect.width; x++) {
      const gx = rect.x + x;
      const i = y * rect.width + x;

      const nx = (gx - cx) / rx;
      const ny = (gy - cy) / ry;
      const t = Math.sqrt(nx * nx + ny * ny);
      // 楕円の内側 CORE までは 1、EDGE で 0。境目を滑らかにする。
      const w =
        t <= WEIGHT_CORE
          ? 1
          : t >= WEIGHT_EDGE
            ? 0
            : smoothstep((WEIGHT_EDGE - t) / (WEIGHT_EDGE - WEIGHT_CORE));
      weight[i] = w;
      if (w <= 0) continue;
      covered++;

      // 近傍だけを見る。半径を広げながら、最低 MIN_NEIGHBOURS 点そろうまで探す。
      let acc = 0;
      let wsum = 0;
      let found = 0;
      for (let r = cutoff; ; r *= 2) {
        acc = 0;
        wsum = 0;
        found = 0;
        const r2 = r * r;
        grid.forEach(gx, gy, r, (p) => {
          const dx = gx - p.x;
          const dy = gy - p.y;
          const dd = dx * dx + dy * dy;
          if (dd > r2) return;
          const d2 = dd + eps;
          const k = power === 2 ? 1 / d2 : 1 / Math.pow(d2, power / 2);
          acc += k * p.z;
          wsum += k;
          found++;
        });
        if (found >= MIN_NEIGHBOURS || r > maxReach) break;
      }
      depth[i] = wsum > 0 ? acc / wsum : 0;
    }
  }

  // 点に張り付いた凸凹を、点間隔ぶんだけ均す。
  const radius = Math.max(1, Math.round(spacing * 0.5));
  const trendRadius = Math.max(2, Math.round(Math.max(rx, ry) * TREND_RADIUS_RATIO));
  return {
    depth: boxBlur(depth, rect.width, rect.height, radius),
    weight,
    covered,
    trendRadius,
  };
}

function smoothstep(t: number): number {
  const s = Math.max(0, Math.min(1, t));
  return s * s * (3 - 2 * s);
}

/** 分離可能な箱平均。積分画像を持たずに済む大きさなので素直に 2 パス。 */
function boxBlur(src: Float32Array, width: number, height: number, radius: number): Float32Array {
  const tmp = new Float32Array(src.length);
  const out = new Float32Array(src.length);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      let sum = 0;
      let n = 0;
      for (let d = -radius; d <= radius; d++) {
        const xx = x + d;
        if (xx < 0 || xx >= width) continue;
        sum += src[y * width + xx] as number;
        n++;
      }
      tmp[y * width + x] = n > 0 ? sum / n : (src[y * width + x] as number);
    }
  }
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      let sum = 0;
      let n = 0;
      for (let d = -radius; d <= radius; d++) {
        const yy = y + d;
        if (yy < 0 || yy >= height) continue;
        sum += tmp[yy * width + x] as number;
        n++;
      }
      out[y * width + x] = n > 0 ? sum / n : (tmp[y * width + x] as number);
    }
  }
  return out;
}

/**
 * 顔の起伏を深度に入れる（細部の帯だけ）。
 *
 * **尺度は当てはめで決めない。** `fuseDepth` のアフィン当てはめは、タイルを
 * 全体パスに合わせるために「合わせる相手に同じ構造がある」ことを前提に
 * している。顔ではそれが成り立たない。深度モデルの顔は平らなので、
 * 合わせる相手が存在せず、当てはめは雑音を拾う。実測では傾き a が
 * **−0.0005**（符号が逆）で返ってきた。
 *
 * landmark の z は crop の画素と同じ尺度（弱透視）なので、尺度は幾何から
 * 出せる。距離 Z にある面では 1 画素が Z / f の実寸に当たるから、
 *
 *     起伏 = (z − その場の大域の z) × Z / f
 *
 * とすればよい。当てはめる必要がそもそも無い。
 *
 * **大域は足さない。** ここが v2.6.1 で直したところ（docs/09 §V13）。
 * 以前はラプラシアンブレンドで landmark 面と深度モデルを混ぜ、粗い側の
 * 1 層だけを深度モデルに固定していた。1024² の 6 段では、固定されるのは
 * 波長 64px より粗い成分だけで、顔（幅 145px）の丸みはほぼ landmark 面に
 * 置き換わる。ところが landmark 面はカメラを向いた面の統計的な型でしか
 * なく、実測でこの二つは顔の中で **15mm** 食い違っていた（深度モデルの
 * 顔の奥行きは 49mm、landmark 面は 17mm）。その食い違いが楕円の重みで
 * 切られて、左頬が 7mm 手前へ、額が奥へ動く低周波のうねりになった。
 * 顔の輪郭に沿った隆起として見える。
 *
 * landmark モデルが知っているのは**細部だけ**である。だから細部だけを
 * 取り出して足す。半径 `trendRadius` の（重み付き）箱平均を大域とみなし、
 * そこからの差だけを深度へ加える。こうすると足す量は作りからして
 * 大域成分を持たないので、重みの羽根がどこを通っても輪ができない。
 * 頭が空間のどこにあるか、顔がどれだけ丸いかは、これまでどおり深度
 * モデルのものが残る。
 *
 * @param depth   融合済みの実寸深度（大きいほど奥）。破壊しない。
 * @param focalPx 作業グリッドでの焦点距離（画素）。
 */
export function applyFaceRelief(
  depth: ArrayLike<number>,
  surface: FaceSurface,
  rect: Rect,
  width: number,
  height: number,
  focalPx: number,
  /**
   * 顔の傾き（1 次の項）も landmark に合わせるか。
   *
   * 既定で入れる。切れるようにしてあるのは、検査で細部だけを見たいときの
   * ためである。
   */
  correctTilt = true,
): Float32Array {
  const out = Float32Array.from(depth);
  if (surface.covered < 256 || focalPx <= 0) return out;

  // 顔がどれだけ遠いか。1 画素の実寸（Z / f）を出すためだけに要る。
  // 外れ値に振られないよう中央値で取る。
  const bodyVals: number[] = [];
  for (let y = 0; y < rect.height; y++) {
    const gy = rect.y + y;
    if (gy < 0 || gy >= height) continue;
    for (let x = 0; x < rect.width; x++) {
      const gx = rect.x + x;
      if (gx < 0 || gx >= width) continue;
      if ((surface.weight[y * rect.width + x] as number) < 0.5) continue;
      const v = depth[gy * width + gx] as number;
      if (Number.isFinite(v)) bodyVals.push(v);
    }
  }
  if (bodyVals.length < 256) return out;
  bodyVals.sort((a, b) => a - b);
  const perPixel = (bodyVals[bodyVals.length >> 1] as number) / focalPx;

  // landmark 面の大域。顔の外（重み 0）は混ぜない。
  const trend = maskedBoxBlur(
    surface.depth,
    surface.weight,
    rect.width,
    rect.height,
    surface.trendRadius,
  );

  // 顔の傾き（1 次の項）を landmark に合わせる（v2.6.8、docs/09 §V23）。
  //
  // **なぜ要るのか。** 深度モデルは顔の向きを取り違える。実写（正面を向いた
  // 顔）で、深度モデルの面は顔幅 227px にわたって **31mm** 傾いていた。顔幅は
  // 実寸で 103mm なので、**16° ほど横を向いた顔**になっていたことになる。
  // 見る側には「片目だけ奥に沈んでいる」「顔の片側が後ろすぎる」と映る。
  // landmark の面は同じ場所で傾き 0、つまり**正面**だと言っている。
  //
  // 位置（0 次）は深度モデルのものを使う。landmark の z は絶対値としては
  // 当てにならないし、顔がどこにあるかは深度モデルのほうが正しい。
  // 直すのは向きだけである。
  const tilt = correctTilt ? tiltFix(depth, surface, rect, width, height, perPixel) : null;

  for (let y = 0; y < rect.height; y++) {
    const gy = rect.y + y;
    if (gy < 0 || gy >= height) continue;
    for (let x = 0; x < rect.width; x++) {
      const gx = rect.x + x;
      if (gx < 0 || gx >= width) continue;
      const ti = y * rect.width + x;
      const w = surface.weight[ti] as number;
      if (w <= 0) continue;
      const relief = ((surface.depth[ti] as number) - (trend[ti] as number)) * perPixel;
      const gi = gy * width + gx;
      const fix = tilt ? tilt(x, y) : 0;
      out[gi] = (depth[gi] as number) + w * (relief + fix);
    }
  }
  return out;
}

/** 顔の傾きを landmark に合わせるのに足す量（実寸）。行き過ぎは上限で止める。 */
function tiltFix(
  depth: ArrayLike<number>,
  surface: FaceSurface,
  rect: Rect,
  width: number,
  height: number,
  perPixel: number,
): ((x: number, y: number) => number) | null {
  const plate = fitPlane(rect, surface.weight, (x, y) => {
    const v = surface.depth[y * rect.width + x] as number;
    return Number.isFinite(v) ? v * perPixel : Number.NaN;
  });
  const model = fitPlane(rect, surface.weight, (x, y) => {
    const gy = rect.y + y;
    const gx = rect.x + x;
    if (gy < 0 || gy >= height || gx < 0 || gx >= width) return Number.NaN;
    return depth[gy * width + gx] as number;
  });
  if (!plate || !model) return null;

  // 傾き [実寸/画素] の上限。四角の幅にわたる変化が
  // `MAX_TILT_RATIO × 四角の幅（実寸）` を超えないようにする。
  const limit = MAX_TILT_RATIO * perPixel;
  const bx = clamp(plate.bx - model.bx, -limit, limit);
  const by = clamp(plate.by - model.by, -limit, limit);
  if (bx === 0 && by === 0) return null;
  // 回す中心は**重みの重心**にする。四角の中心にすると、顔の楕円が四角の
  // 真ん中に無いぶんだけ顔ぜんたいが前後にずれる。
  return (x, y) => bx * (x - model.cx) + by * (y - model.cy);
}

/** 重み付きの平面あてはめ。傾きと、重みの重心を返す。 */
function fitPlane(
  rect: Rect,
  weight: ArrayLike<number>,
  get: (x: number, y: number) => number,
): { bx: number; by: number; cx: number; cy: number } | null {
  let sw = 0;
  let sx = 0;
  let sy = 0;
  let sz = 0;
  let sxx = 0;
  let sxy = 0;
  let syy = 0;
  let sxz = 0;
  let syz = 0;
  for (let y = 0; y < rect.height; y++) {
    for (let x = 0; x < rect.width; x++) {
      const w = weight[y * rect.width + x] as number;
      if (w <= 0) continue;
      const z = get(x, y);
      if (!Number.isFinite(z)) continue;
      sw += w;
      sx += w * x;
      sy += w * y;
      sz += w * z;
      sxx += w * x * x;
      sxy += w * x * y;
      syy += w * y * y;
      sxz += w * x * z;
      syz += w * y * z;
    }
  }
  if (sw < 64) return null;
  const mx = sx / sw;
  const my = sy / sw;
  const mz = sz / sw;
  const cxx = sxx - sw * mx * mx;
  const cxy = sxy - sw * mx * my;
  const cyy = syy - sw * my * my;
  const cxz = sxz - sw * mx * mz;
  const cyz = syz - sw * my * mz;
  const det = cxx * cyy - cxy * cxy;
  if (!(Math.abs(det) > 1e-12)) return null;
  return {
    bx: (cxz * cyy - cyz * cxy) / det,
    by: (cyz * cxx - cxz * cxy) / det,
    cx: mx,
    cy: my,
  };
}

function clamp(v: number, lo: number, hi: number): number {
  return v < lo ? lo : v > hi ? hi : v;
}

/**
 * 重みのある画素だけを見る箱平均。窓を滑らせるので半径に依らず O(n)。
 *
 * 顔の外を 0 として混ぜると、楕円の縁で大域が 0 へ引っ張られ、そこに
 * 偽の起伏ができる。数えるのは重みのある画素だけにする。
 */
function maskedBoxBlur(
  src: Float32Array,
  mask: Float32Array,
  width: number,
  height: number,
  radius: number,
): Float32Array {
  const n = width * height;
  const rowSum = new Float64Array(n);
  const rowCount = new Float64Array(n);
  for (let y = 0; y < height; y++) {
    const row = y * width;
    let sum = 0;
    let count = 0;
    for (let x = 0; x <= radius && x < width; x++) {
      if ((mask[row + x] as number) <= 0) continue;
      sum += src[row + x] as number;
      count++;
    }
    for (let x = 0; x < width; x++) {
      rowSum[row + x] = sum;
      rowCount[row + x] = count;
      const add = x + radius + 1;
      if (add < width && (mask[row + add] as number) > 0) {
        sum += src[row + add] as number;
        count++;
      }
      const drop = x - radius;
      if (drop >= 0 && (mask[row + drop] as number) > 0) {
        sum -= src[row + drop] as number;
        count--;
      }
    }
  }

  const out = new Float32Array(n);
  for (let x = 0; x < width; x++) {
    let sum = 0;
    let count = 0;
    for (let y = 0; y <= radius && y < height; y++) {
      sum += rowSum[y * width + x] as number;
      count += rowCount[y * width + x] as number;
    }
    for (let y = 0; y < height; y++) {
      const i = y * width + x;
      out[i] = count > 0 ? sum / count : (src[i] as number);
      const add = y + radius + 1;
      if (add < height) {
        sum += rowSum[add * width + x] as number;
        count += rowCount[add * width + x] as number;
      }
      const drop = y - radius;
      if (drop >= 0) {
        sum -= rowSum[drop * width + x] as number;
        count -= rowCount[drop * width + x] as number;
      }
    }
  }
  return out;
}
