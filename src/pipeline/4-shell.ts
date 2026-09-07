/**
 * シェル構築（docs/03 §3.6）— 前面・背面・スカート。
 *
 * 半球カバー（決定 D2）の中核。前面シェルだけでは、視点を横に回した瞬間に
 * 「紙のように薄い板」に見えてしまう。閉じた殻を作ることで ±60° 近くまで破綻しない。
 */
import { distanceTransform } from './geometry/distanceTransform';

const SUBJECT_THRESHOLD = 128;

export interface ThicknessParams {
  /** 最大厚み。被写体の奥行きの半分。 */
  readonly maxThickness: number;
  /** 断面プロファイル。円柱状にすると縁が角張って見える。 */
  readonly profile: 'ellipsoid' | 'cylinder';
}

/**
 * シルエットの内部距離変換から厚みマップを作る。
 *
 * `t(u,v) = T · sqrt(1 − (1 − D/Dmax)²)`
 *
 * この形は円柱ではなく**楕円断面**を与える。シルエット境界（α=0.5 の等高線）で厚み0、
 * 中心では最大の厚み、という滑らかな立体になる。厚み一定（円柱状）にすると縁が角張って見える。
 *
 * 距離変換は「最も近い背景画素までの距離」を返すので、境界の画素でも 1 になる。
 * 実際の輪郭は画素の中間にあるので 0.5 を引いてから正規化する。
 *
 * なお sqrt プロファイルは r=0 で傾きが無限大になるため、**輪郭の1画素内側でも
 * 厚みは最大の 3〜4割に達する**。これは誤差ではなく球の幾何そのもの
 * （半径 R の球は輪郭から 1px 内側で厚み 2R·sqrt(2/R)）。
 * リムの継ぎ目は α の減衰（docs/03 §3.6.4）とスカートで隠す。
 */
export function thicknessMap(
  alpha: ArrayLike<number>,
  width: number,
  height: number,
  params: ThicknessParams,
): Float32Array {
  const dist = distanceTransform(alpha, width, height, (v) => v >= SUBJECT_THRESHOLD);
  let dMax = 0;
  for (let i = 0; i < dist.length; i++) if ((dist[i] as number) > dMax) dMax = dist[i] as number;

  const out = new Float32Array(dist.length);
  if (dMax <= 0.5) return out;

  // 輪郭は画素の中間にあるので 0.5 を引く
  const span = dMax - 0.5;
  for (let i = 0; i < dist.length; i++) {
    const d = (dist[i] as number) - 0.5;
    if (d <= 0) continue;
    const r = Math.min(1, d / span); // 0（輪郭）〜 1（最も内側）
    const shape = params.profile === 'cylinder' ? 1 : Math.sqrt(Math.max(0, 1 - (1 - r) * (1 - r)));
    out[i] = params.maxThickness * shape;
  }
  return out;
}

/**
 * 背面色を作る（docs/03 §3.6.2）。
 *
 * v1 は一律に「前面色の水平反転」だったが、人物では**顔の鏡像が後頭部に現れて**
 * 不気味になる。v2 で被写体モード別に変えた。
 *
 * - 人物 (`edge-extend`): 各行について、シルエット境界から内側 6px の色を左右それぞれ
 *   取り、行方向に線形補間する。後ろから見えるのは「髪の色」「服の色」であり、
 *   それは輪郭付近の色そのもの。顔のパーツは一切出ない
 * - 物体 (`mirror-h`): 前面色の水平反転。多くの物体は左右対称に近い
 *
 * どちらも厚みに応じて減光し、裏側が奥まって見える簡易的な陰影を与える。
 * 出力は半解像度（背面は 1/4 密度なので前面と同じ解像度は要らない）。
 */
export interface BackColorParams {
  readonly mode: 'edge-extend' | 'mirror-h';
  readonly shadeBase: number;
  readonly shadeRange: number;
  /** 縁色を採る内側への距離（画素）。 */
  readonly edgeInset?: number;
}

export function backColor(
  frontColor: ArrayLike<number>,
  alpha: ArrayLike<number>,
  thickness: Float32Array,
  width: number,
  height: number,
  params: BackColorParams,
): Uint8ClampedArray {
  const hw = Math.floor(width / 2);
  const hh = Math.floor(height / 2);
  const out = new Uint8ClampedArray(hw * hh * 3);

  let tMax = 0;
  for (let i = 0; i < thickness.length; i++) if ((thickness[i] as number) > tMax) tMax = thickness[i] as number;

  // 人物モード: 行ごとに左右の縁色を先に集める
  const inset = params.edgeInset ?? 6;
  const leftColor = new Float32Array(height * 3);
  const rightColor = new Float32Array(height * 3);
  const rowSpan = new Int32Array(height * 2).fill(-1);

  if (params.mode === 'edge-extend') {
    for (let y = 0; y < height; y++) {
      let x0 = -1;
      let x1 = -1;
      for (let x = 0; x < width; x++) {
        if ((alpha[y * width + x] as number) >= SUBJECT_THRESHOLD) {
          if (x0 < 0) x0 = x;
          x1 = x;
        }
      }
      rowSpan[y * 2] = x0;
      rowSpan[y * 2 + 1] = x1;
      if (x0 < 0) continue;
      const lx = Math.min(x1, x0 + inset);
      const rx = Math.max(x0, x1 - inset);
      for (let c = 0; c < 3; c++) {
        leftColor[y * 3 + c] = frontColor[(y * width + lx) * 3 + c] as number;
        rightColor[y * 3 + c] = frontColor[(y * width + rx) * 3 + c] as number;
      }
    }
  }

  for (let y = 0; y < hh; y++) {
    const sy = y * 2;
    for (let x = 0; x < hw; x++) {
      const sx = x * 2;
      const si = sy * width + sx;
      const di = (y * hw + x) * 3;

      const t = thickness[si] as number;
      const shade = params.shadeBase + params.shadeRange * (1 - (tMax > 0 ? t / tMax : 0));

      if (params.mode === 'mirror-h') {
        const mx = width - 1 - sx;
        const mi = sy * width + mx;
        for (let c = 0; c < 3; c++) out[di + c] = (frontColor[mi * 3 + c] as number) * shade;
      } else {
        const x0 = rowSpan[sy * 2] as number;
        const x1 = rowSpan[sy * 2 + 1] as number;
        if (x0 < 0 || x1 <= x0) {
          for (let c = 0; c < 3; c++) out[di + c] = 0;
          continue;
        }
        const u = Math.max(0, Math.min(1, (sx - x0) / (x1 - x0)));
        for (let c = 0; c < 3; c++) {
          const v = (leftColor[sy * 3 + c] as number) * (1 - u) + (rightColor[sy * 3 + c] as number) * u;
          out[di + c] = v * shade;
        }
      }
    }
  }

  // 人物モードは縦方向にぼかして、行ごとの縁色のばらつきを均す
  if (params.mode === 'edge-extend') return blurVertical(out, hw, hh, 6);
  return out;
}

/** 縦方向のボックスぼかしを3回かけてガウシアン近似にする。 */
function blurVertical(
  rgb: Uint8ClampedArray,
  width: number,
  height: number,
  radius: number,
): Uint8ClampedArray {
  let src = new Uint8ClampedArray(rgb.length);
  src.set(rgb);
  let dst = new Uint8ClampedArray(rgb.length);

  for (let pass = 0; pass < 3; pass++) {
    for (let x = 0; x < width; x++) {
      for (let c = 0; c < 3; c++) {
        for (let y = 0; y < height; y++) {
          let sum = 0;
          let n = 0;
          for (let dy = -radius; dy <= radius; dy++) {
            const yy = y + dy;
            if (yy < 0 || yy >= height) continue;
            sum += src[(yy * width + x) * 3 + c] as number;
            n++;
          }
          dst[(y * width + x) * 3 + c] = sum / Math.max(n, 1);
        }
      }
    }
    const tmp = src;
    src = dst;
    dst = tmp;
  }
  return src;
}

/**
 * 深度エッジを検出する（docs/03 §3.6.3）。
 *
 * 視点を動かすと、深度が不連続な縁の**奥側**に、それまで隠れていた領域が露出する。
 * そこにスカートを張って穴を塞ぐ。
 *
 * @returns 各画素について、深度ギャップの大きさ（0 ならエッジでない）。
 */
export function detectDepthEdges(
  depth: Uint16Array,
  alpha: ArrayLike<number>,
  width: number,
  height: number,
  threshold: number,
): Float32Array {
  const out = new Float32Array(width * height);
  const limit = threshold * 65535;

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const i = y * width + x;
      if ((alpha[i] as number) < SUBJECT_THRESHOLD) continue;
      const z = depth[i] as number;

      // 4近傍で最も大きい「奥向きの」ギャップ。手前向きは別の画素側で拾う。
      let gap = 0;
      if (x > 0) gap = Math.max(gap, (depth[i - 1] as number) - z);
      if (x + 1 < width) gap = Math.max(gap, (depth[i + 1] as number) - z);
      if (y > 0) gap = Math.max(gap, (depth[i - width] as number) - z);
      if (y + 1 < height) gap = Math.max(gap, (depth[i + width] as number) - z);

      if (gap > limit) out[i] = gap / 65535;
    }
  }
  return out;
}
