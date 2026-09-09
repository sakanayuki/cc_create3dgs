/**
 * 四肢に円柱の断面を与える（docs/09 §V20、v2.6.5）。
 *
 * **なぜ要るのか。** 単眼深度モデルは腕や脚の横断面をほとんど平らに返す。
 * 実写（立ち姿）で「端が中央より奥へ引く量 ÷ 脚の幅」を測ると:
 *
 * | | 腿 | 腿下 | 脛 |
 * |---|---|---|---|
 * | 私たち（DA3） | 0.08 | 0.05 | 0.09 |
 * | 参照実装（SHARP） | **0.25** | **0.33** | **0.27** |
 *
 * **約 3 倍平ら。** 生の DA3 の出力の時点でそうなっていて、較正でも強調でも
 * 作っていない。深度モデルは「脚がどこにあるか」は当てるが「脚が丸い」ことは
 * 知らない。
 *
 * **姿勢モデルは要らない。** 立ち姿の腕や脚では、**横一線に切ったときの
 * シルエットの区間がそのまま断面**である。区間の幅から円柱の半径が出るので、
 * 関節位置を別のモデルで当てなくてよい。胴のように太い区間は円柱ではないので、
 * 被写体の幅に対する割合で除く。
 *
 * **入れるのは大域だけ。** 顔（`applyFaceRelief`）と同じ構造にする。区間の中の
 * なだらかな形だけを円柱に差し替え、細部（膝の皺、筋の凹凸）は深度モデルの
 * ものを残す。こうすると足す量が作りからして大域成分しか持たないので、
 * 区間の端に段差ができない。
 */
import { SUBJECT_ALPHA } from '../1-matte';
import type { Rect } from '../0-preprocess';

export interface LimbRoundnessParams {
  /**
   * 四肢とみなす区間の幅（被写体の外接幅に対する割合）。
   *
   * 立ち姿では脚が 0.10〜0.20、胴が 0.55〜0.75 に出る。間を取る。
   */
  readonly maxWidthRatio: number;
  /**
   * 円柱の扁平率。1 で真円（端が半径ぶん奥）、0.5 なら半分に潰した楕円。
   *
   * **1 が正しい。** 「端の外側 18% の平均」と「中央 1/3 の平均」の差を
   * 幅で割る測り方では、真円の柱でも 0.5 ではなく **0.29** になる（端の
   * 平均は t≈0.91 のあたりで、そこは半径の 0.59 倍しか下がらないため）。
   * 参照実装の実測値 0.25〜0.33 はこの 0.29 とほぼ一致する。つまり
   * **参照実装はほぼ真円の柱を出している**。人の四肢の断面もおよそ円なので、
   * 既定は 1 にする。実測でも 1.0 で 0.17〜0.25 と参照実装の範囲に入り、
   * 1.2 では行き過ぎる。
   */
  readonly flatness: number;
  /** 区間の当てはめを縦に均す半径（画素）。行ごとの揺れが縞にならないように。 */
  readonly smoothRows: number;
  /** 効かせる強さ 0..1。 */
  readonly strength: number;
  /** これより短い区間は触らない（指や髪の房）。 */
  readonly minRunPx: number;
}

/**
 * 頭は円柱ではないので外す。
 *
 * 頭の幅は被写体の幅の 2〜4 割で、`maxWidthRatio` の網に掛かってしまう。
 * 実測では顔の 31〜56% の画素が「四肢」と判定された。顔には専用の
 * 起伏（`applyFaceRelief`）が入るので、そちらと取り合いになる。
 */
export interface LimbExclusion {
  /** 触らない四角（ふつうは頭の箱）。 */
  readonly rect: Rect;
}

export const DEFAULT_LIMB_PARAMS: LimbRoundnessParams = {
  maxWidthRatio: 0.32,
  flatness: 1,
  smoothRows: 6,
  strength: 1,
  minRunPx: 12,
};

/**
 * 四肢の断面を円柱に寄せる。
 *
 * @param depth   実寸深度（大きいほど奥）。破壊しない。
 * @param focalPx 作業グリッドでの焦点距離（画素）。
 * @returns 直した深度。触らない画素は入力のまま。
 */
export function applyLimbRoundness(
  depth: ArrayLike<number>,
  alpha: ArrayLike<number>,
  width: number,
  height: number,
  focalPx: number,
  params: LimbRoundnessParams = DEFAULT_LIMB_PARAMS,
  /** 触らない四角（頭の箱）。null なら全身に効かせる。 */
  exclude: Rect | null = null,
): Float32Array {
  const out = Float32Array.from(depth as ArrayLike<number>);
  if (focalPx <= 0 || params.strength <= 0) return out;

  // 被写体の外接幅。四肢かどうかの物差しにする。
  let xlo = width;
  let xhi = -1;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      if ((alpha[y * width + x] as number) < SUBJECT_ALPHA) continue;
      if (x < xlo) xlo = x;
      if (x > xhi) xhi = x;
    }
  }
  if (xhi < xlo) return out;
  const subjectWidth = xhi - xlo + 1;
  const maxRun = subjectWidth * params.maxWidthRatio;

  // 行ごとに補正量を作り、あとで縦に均す。
  const fix = new Float32Array(width * height);
  const touched = new Uint8Array(width * height);

  for (let y = 0; y < height; y++) {
    if (exclude && y >= exclude.y && y < exclude.y + exclude.height) continue;
    const row = y * width;
    let x = 0;
    while (x < width) {
      if ((alpha[row + x] as number) < SUBJECT_ALPHA) {
        x++;
        continue;
      }
      let end = x;
      while (end + 1 < width && (alpha[row + end + 1] as number) >= SUBJECT_ALPHA) end++;
      const run = end - x + 1;
      if (run >= params.minRunPx && run <= maxRun) {
        addCylinder(depth, fix, touched, row, x, end, focalPx, params);
      }
      x = end + 1;
    }
  }

  // 縦に均す。行ごとに独立して当てはめているので、そのままだと横縞になる。
  const smooth = smoothColumns(fix, touched, width, height, params.smoothRows);
  for (let i = 0; i < out.length; i++) {
    if (touched[i] === 1) out[i] = (depth[i] as number) + (smooth[i] as number);
  }
  return out;
}

/**
 * 区間ひとつぶんの補正を求める。
 *
 * 区間の中のなだらかな断面（半径ぶんの箱平均）を円柱に差し替える。
 * 細部（平均からの残り）はそのまま残す。
 */
function addCylinder(
  depth: ArrayLike<number>,
  fix: Float32Array,
  touched: Uint8Array,
  row: number,
  x0: number,
  x1: number,
  focalPx: number,
  params: LimbRoundnessParams,
): void {
  const n = x1 - x0 + 1;
  const half = n / 2;
  const cx = (x0 + x1) / 2;

  // 区間の代表の深度。1 画素の実寸を出すのに要る。
  let sum = 0;
  for (let x = x0; x <= x1; x++) sum += depth[row + x] as number;
  const mean = sum / n;
  if (!(mean > 0)) return;
  // 円柱の半径（実寸）。扁平率を掛ける。
  const radius = half * (mean / focalPx) * params.flatness;

  // いまの断面のなだらかな形。半径の半分の箱平均で取る。
  const win = Math.max(1, Math.round(half * 0.5));
  const smoothNow = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let s = 0;
    let c = 0;
    for (let k = -win; k <= win; k++) {
      const j = i + k;
      if (j < 0 || j >= n) continue;
      s += depth[row + x0 + j] as number;
      c++;
    }
    smoothNow[i] = s / c;
  }

  // 円柱の断面。中央が手前（値が小さい）。
  const cyl = new Float64Array(n);
  let cylSum = 0;
  let nowSum = 0;
  for (let i = 0; i < n; i++) {
    const t = Math.min(1, Math.abs(x0 + i - cx) / half);
    cyl[i] = radius * (1 - Math.sqrt(Math.max(0, 1 - t * t)));
    cylSum += cyl[i] as number;
    nowSum += smoothNow[i] as number;
  }
  const cylMean = cylSum / n;
  const nowMean = nowSum / n;

  for (let i = 0; i < n; i++) {
    const want = (cyl[i] as number) - cylMean;
    const have = (smoothNow[i] as number) - nowMean;
    fix[row + x0 + i] = params.strength * (want - have);
    touched[row + x0 + i] = 1;
  }
}

/** 補正量を縦に均す。触っていない画素は混ぜない。 */
function smoothColumns(
  fix: Float32Array,
  touched: Uint8Array,
  width: number,
  height: number,
  radius: number,
): Float32Array {
  if (radius < 1) return fix;
  const out = new Float32Array(fix.length);
  for (let x = 0; x < width; x++) {
    for (let y = 0; y < height; y++) {
      const i = y * width + x;
      if (touched[i] === 0) continue;
      let s = 0;
      let c = 0;
      for (let k = -radius; k <= radius; k++) {
        const yy = y + k;
        if (yy < 0 || yy >= height) continue;
        const j = yy * width + x;
        if (touched[j] === 0) continue;
        s += fix[j] as number;
        c++;
      }
      out[i] = c > 0 ? s / c : (fix[i] as number);
    }
  }
  return out;
}
