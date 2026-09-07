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
  /** 奥行き ÷ 短辺。UI の「立体感」スライダ（0.3〜1.2、既定 0.65）。 */
  readonly depthToWidthRatio?: number;
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
}

const SUBJECT_THRESHOLD = 128;
const DEFAULT_RATIO = 0.65;

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
 * 深度を較正して 0..65535 の正規化深度にする。
 *
 * 半球カバー（決定 D2）では絶対スケールは意味を持たない。重要なのは
 * 「被写体の奥行きが幅に対してどの程度あるか」という比だけで、それを直接指定する。
 */
export function calibrate(input: CalibrationInput): CalibrationResult {
  const { raw, width, height, alpha, kind, focalPx } = input;
  const ratio = input.depthToWidthRatio ?? DEFAULT_RATIO;

  const shortSide = subjectShortSide(alpha, width, height);
  if (shortSide === 0) {
    throw new Error('被写体が見つかりません（α が閾値を超える画素がない）');
  }

  // 半球カバー（決定 D2）では絶対スケールは意味を持たない。重要なのは
  // 「被写体の奥行きが幅に対してどの程度あるか」という比だけで、それを直接指定する。
  // 距離 1.0 にある被写体の実サイズは shortSide/focalPx。
  const targetSpan = (shortSide / focalPx) * ratio;

  const [p5, p50, p95] = percentileOfSubject(raw, alpha, [0.05, 0.5, 0.95]) as [number, number, number];

  let z: Float32Array;
  let shift = 0;

  if (kind === 'depth') {
    // DA3 系。スケールだけが不定なので線形に合わせる。
    const span = Math.max(p95 - p5, 1e-9);
    const scale = targetSpan / span;
    z = new Float32Array(raw.length);
    for (let i = 0; i < raw.length; i++) z[i] = ((raw[i] as number) - p50) * scale + 1.0;
  } else {
    // V2 系。z = 1/(a·d + b) を2つの制約から閉形式で解く。
    // 注意: 逆深度は「大きいほど手前」なので、深度としての順序は反転する。
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
      const den = solved.a * (raw[i] as number) + solved.b;
      z[i] = den > 1e-6 ? 1 / den : 0;
    }
  }

  // 被写体領域の実際の値域を測って 0..65535 に写す。
  const [zLo, zHi] = percentileOfSubject(z, alpha, [0.005, 0.995]) as [number, number];
  const nearZ = Math.min(zLo, zHi);
  const farZ = Math.max(zLo, zHi);
  const span = Math.max(farZ - nearZ, 1e-9);

  const depth = new Uint16Array(raw.length);
  for (let i = 0; i < raw.length; i++) {
    const t = ((z[i] as number) - nearZ) / span;
    depth[i] = Math.max(0, Math.min(65535, Math.round(t * 65535)));
  }
  return { depth, nearZ, farZ, shift, silhouetteSamples: 0 };
}

/**
 * α の軟化境界にある画素の深度を、最も近い被写体内部の深度で置き換える（docs/03 §3.5.3(c)）。
 *
 * 0 < α < 0.5 の画素は、深度モデルには背景が混ざって見えている。
 * その深度をそのまま使うと、縁の半透明ガウシアンが背景側へ飛び出す。
 */
export function pullBoundaryDepthInward(
  depth: Uint16Array,
  alpha: ArrayLike<number>,
  width: number,
  height: number,
): Uint16Array {
  const nearest = nearestForegroundIndex(alpha, width, height, (v) => v >= SUBJECT_THRESHOLD);
  const out = new Uint16Array(depth);
  for (let i = 0; i < depth.length; i++) {
    const a = alpha[i] as number;
    if (a > 0 && a < SUBJECT_THRESHOLD) {
      const src = nearest[i] as number;
      if (src >= 0) out[i] = depth[src] as number;
    }
  }
  return out;
}
