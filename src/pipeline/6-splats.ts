/**
 * ④⑤ シェルからスプラットを組み立てる（docs/03 §3.6）。
 *
 * ここがパイプラインと描画層の接点。深度・法線・色・セルマップを受け取り、
 * レンダラがそのまま食える 24 バイト × N のバッファを返す（docs/04 §4.8.1）。
 *
 * 座標系の変換を1箇所に閉じ込めるのがこのモジュールの役目。
 *
 *   パイプライン側（カメラ空間）: 原点にカメラ。x は右、**y は下**（画像と同じ）、
 *     z は奥に向かって正。法線は手前（カメラ）を向くので z 成分が負。
 *   描画側（ワールド空間）: 被写体は原点中心の単位立方体。y は**上**、
 *     カメラは +z 側から原点を見る。手前を向く法線は z 成分が正。
 *
 * よって (x, y, z) → (x, −y, −z)。法線も同じ回転で移す。これを間違えると
 * 背面カリングが表裏逆に効き、被写体が消える。
 */
import { encodeOct, packHalf2, packRgba8 } from '../codec/pack';
import { SPLAT_BYTES } from '../render/SplatRenderer';
import { distanceTransform } from './geometry/distanceTransform';
import { exposedBandPx } from './8-inpaint';
import type { CellMap } from './7-sample';

/** 被写体とみなす α。 */
const SUBJECT_ALPHA = 128;

export interface BuildInput {
  /** 適応サンプリングの結果。前面シェルはセル1個につき1スプラット。 */
  readonly cells: CellMap;
  /** 画素ごとの法線（パイプライン側カメラ空間）。長さ width×height×3。 */
  readonly normals: Float32Array;
  readonly width: number;
  readonly height: number;
  /** 焦点距離（画素）。DA3 の intrinsics から得る。 */
  readonly focalPx: number;
  /** 主点。既定は画像中心。 */
  readonly cx?: number;
  readonly cy?: number;
  /** 正規化深度 0..1 が対応する実距離。 */
  readonly nearZ: number;
  readonly farZ: number;
}

export interface BuildParams {
  /**
   * サーフェルの広がり。1画素ぶんの footprint に対する半径の倍率。
   *
   * **1.3（v2.6.7）。** docs/03 §3.6.1 は 1.4 だった。小さいと隙間が空き、
   * 大きいとぼける。参照実装（SHARP）の「半径 ÷ 点間隔」は 0.70 で、私たちは
   * 1.4 で 0.80、1.3 で 0.75 になる。1.2（0.69）まで落とすと、胴を 0.12 単位に
   * 切った至近で穴が 0.11% → 0.46% に増えたので、1.3 で止める。docs/11 §11.6 S5。
   */
  readonly spread: number;
  /** 傾斜面での引き伸ばしの上限。`1/max(|n·v|, cosLimit)`。 */
  readonly cosLimit: number;
  /**
   * 背面シェルを作るか。**既定は false**（v2.6）。
   *
   * 実測すると、私たちのビューアでは背面シェルは **±60° で 1 枚も描かれない**。
   * サーフェルの法線で表裏を落としているので、全部カリングされる
   * （実測: 背面ありとなしで、0°/20°/40°/60° の描画枚数が完全に一致した）。
   * つまり自分のビューアでは費用だけがかかっている（枚数の 19%）。
   *
   * さらに悪いのは**他のビューアでの見え方**である。面カリングは通常の 3DGS
   * には無い最適化なので、書き出した `.splat` を一般のビューアで開くと
   * 背面シェルがそのまま描かれ、`backShade` で暗くした殻が前面越しに透けて
   * 斑に見える。参照実装（他ソフト）が背面を持たないのはこのためだろう。
   *
   * 穴が増えるのは 60° で 1.4% → 1.8% と僅かで、連結性は 100% のまま。
   * 引き換えに枚数が 23% 減る。
   */
  readonly backShell: boolean;
  /** 背面シェルの密度（前面に対する間引き。2 なら 2×2 統合＝1/4）。 */
  readonly backStride: number;
  /** 背面の減光。 */
  readonly backShade: number;
  /**
   * リム処理の幅（画素）。docs/03 §3.6.1 は 2px。
   * シルエットからこの距離までのサーフェルは α を落とし、スケールを広げ、
   * 法線を視線方向へ寄せる。0 で無効。
   */
  readonly rimWidth: number;
  /** リムでのスケール倍率。 */
  readonly rimScale: number;
  /** リムで法線を視線方向へ寄せる強さ（0〜1）。 */
  readonly rimNormalBlend: number;
  /** スカートを作るか（docs/03 §3.6.3）。 */
  readonly skirt: boolean;
  /** スカートを立てる深度ギャップの下限（正規化深度）。 */
  readonly skirtThreshold: number;
  /**
   * 段差と判定する「勾配の跳ね上がり」の倍率。
   *
   * 単に差が大きいだけでは、急な斜面と本当の不連続を区別できない。
   * 球の輪郭付近は勾配が無限大に近づくので、閾値だけだとそこ全部に
   * スカートが立ってしまう（実際に合成した半球で 800 枚立った）。
   * 反対側の隣との差の何倍かを見て、勾配が跳ねている所だけを採る。
   */
  readonly skirtStepRatio: number;
  /** スカートの1枚あたりの奥行き刻み（画素相当）。 */
  readonly skirtStep: number;
  /** スカート1本あたりの最大枚数。長い帯が際限なく増えるのを防ぐ。 */
  readonly skirtMaxSteps: number;
  /**
   * 段差とみなす深度ギャップの分位（0〜1）。
   *
   * `skirtThreshold` の固定値と、この分位で決まる値の**大きいほう**を使う。
   * 較正の仕方で深度の勾配の大きさは変わるので、固定値だけだと枚数が
   * 桁で動く（実写で 21% → 63% になった）。
   */
  readonly skirtGapPercentile: number;
}

export const DEFAULT_BUILD_PARAMS: BuildParams = {
  spread: 1.3,
  cosLimit: 0.3,
  backShell: false,
  backStride: 2,
  backShade: 0.55,
  rimWidth: 2,
  rimScale: 1.5,
  rimNormalBlend: 0.6,
  skirt: true,
  skirtThreshold: 0.02,
  skirtStepRatio: 3,
  skirtStep: 1.5,
  skirtMaxSteps: 12,
  skirtGapPercentile: 0.98,
};

export interface Normalization {
  /** ワールド座標 = (カメラ座標を回した値 − center) × scale。 */
  readonly center: readonly [number, number, number];
  readonly scale: number;
}

export interface SplatBuild {
  /** レンダラに渡す 24 バイト × count。 */
  readonly data: Uint8Array;
  readonly count: number;
  /** 前面シェルの個数。 */
  readonly frontCount: number;
  /** 背面シェルの個数。 */
  readonly backCount: number;
  /** スカートの個数。 */
  readonly skirtCount: number;
  /** 正規化に使った変換。背面シェルやスカートを後から足すのに要る。 */
  readonly normalization: Normalization;
  /** ワールド空間での深度レンジ。レンダラの setDepthRange に渡す。 */
  readonly nearZ: number;
  readonly farZ: number;
}

/** 正規化深度 → 実距離。 */
function metricZ(d: number, nearZ: number, farZ: number): number {
  return nearZ + d * (farZ - nearZ);
}

/**
 * セル中心を逆投影する。
 *
 * セルの代表点はセルの中心（左上ではない）。左上にすると、統合したセルほど
 * 被写体が左上へずれる。
 */
function unproject(
  u: number,
  v: number,
  z: number,
  focalPx: number,
  cx: number,
  cy: number,
): [number, number, number] {
  return [((u - cx) * z) / focalPx, ((v - cy) * z) / focalPx, z];
}

/**
 * パイプライン側カメラ空間 → ワールド空間（v2.6 から**恒等変換**）。
 *
 * **上は −Y、奥は +Z**。逆投影がそのまま返す COLMAP 系のカメラ座標
 * （X 右・Y 下・Z 前）をそのまま world として出す。3DGS の一般的な出力が
 * この向きで、他の実装の `.splat` と上下も前後も揃う（実測: 参照実装の
 * z が +0.845〜+1.044 と全部正）。プレビューも保存ファイルも同じ向きである。
 *
 * v2.5 までは `[x, -y, -z]`（X 軸まわり 180°）で +Y を上にしていた。
 * そこから −Y 上へ移すのに `[x, y, -z]` としてはいけない。**行列式が −1 で
 * 鏡像**になり、左右が入れ替わる。回転で移せるのは軸を偶数個반転する
 * ときだけで、ここでは何も反転しないのが正解である。
 */
function toWorld(p: readonly [number, number, number]): [number, number, number] {
  return [p[0], p[1], p[2]];
}

/**
 * 前面シェルと背面シェルを組み立てる。
 *
 * 背面シェルの厚みは呼び出し側が `thicknessMap()` で作って渡す。
 * ここで距離変換までやると、責任範囲が広がりすぎる。
 */
export function buildSplats(
  input: BuildInput,
  color: ArrayLike<number>,
  alpha: ArrayLike<number>,
  thickness: Float32Array | null,
  backColorPlane: ArrayLike<number> | null,
  params: BuildParams = DEFAULT_BUILD_PARAMS,
  /**
   * スカートの色を採るプレーン（⑧ のインペイント結果、RGBA8）。
   * 無ければ奥側の色を伸ばす（v1 方式の縮退、docs/03 §3.7）。
   */
  skirtColorPlane: ArrayLike<number> | null = null,
): SplatBuild {
  const { cells, normals, width, height, focalPx, nearZ, farZ } = input;
  const cx = input.cx ?? width / 2;
  const cy = input.cy ?? height / 2;

  // --- ① 前面シェルの点を作る（正規化前）
  interface Point {
    pos: [number, number, number];
    nrm: [number, number, number];
    /** 画面上の footprint（画素） */
    footprint: number;
    z: number;
    rgba: [number, number, number, number];
  }
  const points: Point[] = [];

  // シルエットからの距離。リム処理で使う。
  const silhouetteDist =
    params.rimWidth > 0 ? distanceTransform(alpha, width, height, (v) => v >= SUBJECT_ALPHA) : null;

  for (let c = 0; c < cells.cellCount; c++) {
    const size = cells.size[c] as number;
    const u = (cells.x[c] as number) + size / 2;
    const v = (cells.y[c] as number) + size / 2;
    const z = metricZ(cells.depth[c] as number, nearZ, farZ);
    if (!(z > 0)) continue;

    // 法線はセル中心の画素から取る。セル内で平均すると、統合したセルで
    // 法線が寝てしまい、面が丸く膨らんで見える。
    const px = Math.min(width - 1, Math.max(0, Math.round(u)));
    const py = Math.min(height - 1, Math.max(0, Math.round(v)));
    const ni = (py * width + px) * 3;
    const n: [number, number, number] = [
      normals[ni] as number,
      normals[ni + 1] as number,
      normals[ni + 2] as number,
    ];

    const pos = unproject(u, v, z, focalPx, cx, cy);
    let a = Math.round((cells.alpha[c] as number) * 255);
    let footprint = size;
    let nrm = n;

    // リム処理（docs/03 §3.6.1）。シルエットの内側 rimWidth までは
    // α を落とし、スケールを広げ、法線を視線方向へ寄せる。
    // 縁のサーフェルは深度が背景と混ざって当てにならないので、そこを
    // そのまま立てると輪郭がぎざぎざに切れる。
    if (params.rimWidth > 0 && silhouetteDist) {
      const d = silhouetteDist[py * width + px] as number;
      if (d < params.rimWidth) {
        const t = Math.max(0, Math.min(1, d / params.rimWidth)); // 0 = 縁
        a = Math.round(a * (0.35 + 0.65 * t));
        footprint = size * (1 + (params.rimScale - 1) * (1 - t));
        // 視線方向（面からカメラへ）。パイプライン側なのでカメラは原点。
        const len = Math.hypot(pos[0], pos[1], pos[2]) || 1;
        const view: [number, number, number] = [-pos[0] / len, -pos[1] / len, -pos[2] / len];
        const w = params.rimNormalBlend * (1 - t);
        const bx = n[0] * (1 - w) + view[0] * w;
        const by = n[1] * (1 - w) + view[1] * w;
        const bz = n[2] * (1 - w) + view[2] * w;
        const bl = Math.hypot(bx, by, bz) || 1;
        nrm = [bx / bl, by / bl, bz / bl];
      }
    }

    points.push({
      pos,
      nrm,
      footprint,
      z,
      rgba: [
        Math.round((cells.color[c * 3] as number) * 255),
        Math.round((cells.color[c * 3 + 1] as number) * 255),
        Math.round((cells.color[c * 3 + 2] as number) * 255),
        a,
      ],
    });
  }

  const frontCount = points.length;

  // --- ② 背面シェル
  if (params.backShell && thickness) {
    const stride = Math.max(1, params.backStride);
    for (let y = 0; y < height; y += stride) {
      for (let x = 0; x < width; x += stride) {
        const i = y * width + x;
        if ((alpha[i] as number) < SUBJECT_ALPHA) continue;
        const t = thickness[i] as number;
        if (!(t > 0)) continue;

        const ci = cells.cellId[i] as number;
        if (ci < 0) continue;
        const z = metricZ(cells.depth[ci] as number, nearZ, farZ);
        if (!(z > 0)) continue;

        // 厚みは正規化された単位（被写体の奥行きに対する比）で与えられるので、
        // 実距離に直してから奥へずらす。
        const back = z + t * (farZ - nearZ);
        const src = backColorPlane;
        const bi = i * (src && src.length >= width * height * 4 ? 4 : 3);
        const shade = params.backShade;
        const rgb: [number, number, number] = src
          ? [src[bi] as number, src[bi + 1] as number, src[bi + 2] as number]
          : [
              (color[i * 4] as number) * shade,
              (color[i * 4 + 1] as number) * shade,
              (color[i * 4 + 2] as number) * shade,
            ];

        points.push({
          pos: unproject(x + 0.5, y + 0.5, back, focalPx, cx, cy),
          // 背面は前面の裏。法線は反転させる。
          nrm: [0, 0, 1],
          footprint: stride,
          z: back,
          rgba: [Math.round(rgb[0]), Math.round(rgb[1]), Math.round(rgb[2]), 255],
        });
      }
    }
  }

  const backCount = points.length - frontCount;

  // --- ③ スカート（docs/03 §3.6.3）
  //
  // 深度が不連続な縁の奥側には、視点を振ると「入力画像には写っていない領域」が
  // 露出する。そこへ、手前の面から奥の面へ向かって帯状のガウシアンを立てて塞ぐ。
  // 色は本来 ⑧ のインペイント結果から採るが、それが無い間は奥側の色を伸ばす
  // （v1 方式）。プレビューはこの色で出し、インペイント完了後に差し替える。
  if (params.skirt) {
    // 閾値は深度マップ自身の分布から決める（v2.3、実写で判明）。
    //
    // 固定値（0.02）は、較正の仕方が変わると意味が変わってしまう。実写で
    // 測ると、隣接画素の差の p90 がちょうど 0.02 前後にあり、**普通の顔の
    // 傾斜がそのまま「不連続」と判定されて**スカートが顔じゅうに林立した
    // （全スプラットの 63% がスカートになり、顔に縦の帯が走った）。
    //
    // 段差かどうかは本来「まわりに比べて跳ねているか」で決まる。下の
    // skirtStepRatio がその判定で、こちらは「小さすぎる段差を捨てる床」に
    // 徹する。分布の上位だけを通せば、較正の仕方によらず枚数が暴れない。
    const gapAt = (i: number): number => {
      if ((alpha[i] as number) < SUBJECT_ALPHA) return -1;
      const ci = cells.cellId[i] as number;
      if (ci < 0) return -1;
      const dHere = cells.depth[ci] as number;
      let g = 0;
      for (const j of [i - 1, i + 1, i - width, i + width]) {
        if ((alpha[j] as number) < SUBJECT_ALPHA) continue;
        const cj = cells.cellId[j] as number;
        if (cj < 0) continue;
        const d = (cells.depth[cj] as number) - dHere;
        if (d > g) g = d;
      }
      return g;
    };
    const sample: number[] = [];
    for (let y = 1; y < height - 1; y++) {
      for (let x = 1; x < width - 1; x++) {
        const g = gapAt(y * width + x);
        if (g >= 0) sample.push(g);
      }
    }
    let adaptive = params.skirtThreshold;
    if (sample.length >= 64) {
      sample.sort((a, b) => a - b);
      adaptive = sample[Math.floor(sample.length * params.skirtGapPercentile)] as number;
    }
    const gapLimit = Math.max(params.skirtThreshold, adaptive);
    for (let y = 1; y < height - 1; y++) {
      for (let x = 1; x < width - 1; x++) {
        const i = y * width + x;
        if ((alpha[i] as number) < SUBJECT_ALPHA) continue;
        const ci = cells.cellId[i] as number;
        if (ci < 0) continue;
        const dHere = cells.depth[ci] as number;

        // 4近傍で最も大きい「奥向きの」段差と、その向き。
        let gap = 0;
        let gx = 0;
        let gy = 0;
        const consider = (j: number, ox: number, oy: number): void => {
          if ((alpha[j] as number) < SUBJECT_ALPHA) return;
          const cj = cells.cellId[j] as number;
          if (cj < 0) return;
          const d = (cells.depth[cj] as number) - dHere;
          if (d > gap) {
            gap = d;
            gx = ox;
            gy = oy;
          }
        };
        consider(i - 1, -1, 0);
        consider(i + 1, 1, 0);
        consider(i - width, 0, -1);
        consider(i + width, 0, 1);
        if (gap < gapLimit) continue;

        // 反対側の隣との差と比べ、勾配が跳ねている所だけを段差とみなす。
        // 滑らかな急斜面では前後の差がほぼ等しいので、ここで落ちる。
        const opp = i - gy * width - gx;
        let backDiff = 0;
        if ((alpha[opp] as number) >= SUBJECT_ALPHA) {
          const co = cells.cellId[opp] as number;
          if (co >= 0) backDiff = Math.abs(dHere - (cells.depth[co] as number));
        }
        if (gap < backDiff * params.skirtStepRatio) continue;

        // シルエットのすぐ内側は背面シェルとリム処理が受け持つ。
        // ここにスカートまで立てると、輪郭が二重に厚くなる。
        if (silhouetteDist && (silhouetteDist[i] as number) < params.rimWidth + 1) continue;

        const zNear = metricZ(dHere, nearZ, farZ);
        const zFar = metricZ(dHere + gap, nearZ, farZ);
        const pixelWorldZ = zNear / focalPx;
        const steps = Math.max(
          1,
          Math.min(params.skirtMaxSteps, Math.round((zFar - zNear) / (pixelWorldZ * params.skirtStep))),
        );

        // 壁の法線は画像平面内で段差に垂直、**奥側**（＝勾配の向き）を向く。
        //
        // 崖に喩えると分かりやすい。左に台地（手前）、右に谷（奥）があるとき、
        // 崖の露出面は右を向いている。左から見れば台地が崖を隠すので見えない。
        // 視点を右（+x）へ振ると、台地は谷より大きく左へ動くので台地の右端の
        // 裏が露出する。そこを塞ぐのがこの壁で、面は右＝勾配の向きを向く。
        //
        // 最初ここを逆向き（-gx, -gy）にしていた。スカートは生成されるのに
        // 背面カリングで全部落ち、描画数も絵も1画素も変わらなかった。
        // 穴の割合を測って初めて分かった（tests/e2e/pipeline.spec.ts）。
        const gl = Math.hypot(gx, gy) || 1;
        const wallN: [number, number, number] = [gx / gl, gy / gl, 0];

        // 視点を振ったときに、この段差の奥側が何画素ぶん露出するか。
        // スカートの各枚は、露出した先の色を持つべきなので、
        // 帯の中のどこに当たるかで色を採る位置を変える。
        const band = exposedBandPx(focalPx, zNear, zFar, (45 * Math.PI) / 180);
        const src = skirtColorPlane ?? color;

        for (let s = 1; s <= steps; s++) {
          const t = s / (steps + 1);
          const z = zNear + t * (zFar - zNear);
          // 奥へ行くほど、段差から遠い位置の色になる。
          const off = Math.round(t * band);
          const sx = Math.max(0, Math.min(width - 1, x + gx * off));
          const sy = Math.max(0, Math.min(height - 1, y + gy * off));
          const fi = sy * width + sx;
          const cr = src[fi * 4] as number;
          const cg = src[fi * 4 + 1] as number;
          const cb = src[fi * 4 + 2] as number;
          // 奥へ行くほど薄くする。奥の端まで不透明だと、そこに板が
          // 見えてしまう（塞ぎたいのは隙間であって、面を足したいのではない）。
          const fade = 1 - t;
          points.push({
            pos: unproject(x + 0.5, y + 0.5, z, focalPx, cx, cy),
            nrm: wallN,
            footprint: params.skirtStep,
            z,
            rgba: [cr, cg, cb, Math.round(255 * fade)],
          });
        }
      }
    }
  }

  const skirtCount = points.length - frontCount - backCount;

  // --- ③ ワールド空間へ移し、単位立方体に正規化する
  let minX = Infinity, minY = Infinity, minZ = Infinity;
  let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
  const world: [number, number, number][] = [];
  for (const p of points) {
    const w = toWorld(p.pos);
    world.push(w);
    if (w[0] < minX) minX = w[0];
    if (w[1] < minY) minY = w[1];
    if (w[2] < minZ) minZ = w[2];
    if (w[0] > maxX) maxX = w[0];
    if (w[1] > maxY) maxY = w[1];
    if (w[2] > maxZ) maxZ = w[2];
  }

  const center: [number, number, number] =
    points.length > 0
      ? [(minX + maxX) / 2, (minY + maxY) / 2, (minZ + maxZ) / 2]
      : [0, 0, 0];
  const extent = Math.max(maxX - minX, maxY - minY, maxZ - minZ, 1e-6);
  // 単位立方体（一辺 1）に収める。レンダラの既定カメラは距離 1.0 でこれを見る。
  const scale = points.length > 0 ? 1 / extent : 1;
  const normalization: Normalization = { center, scale };

  // --- ④ 24 バイトに詰める
  const buf = new ArrayBuffer(points.length * SPLAT_BYTES);
  const f32 = new Float32Array(buf);
  const u32 = new Uint32Array(buf);
  const stride32 = SPLAT_BYTES / 4;

  let outNear = Infinity;
  let outFar = -Infinity;

  for (let k = 0; k < points.length; k++) {
    const p = points[k] as Point;
    const w = world[k] as [number, number, number];
    const pos: [number, number, number] = [
      (w[0] - center[0]) * scale,
      (w[1] - center[1]) * scale,
      (w[2] - center[2]) * scale,
    ];

    const nWorld = toWorld(p.nrm);
    const nlen = Math.hypot(nWorld[0], nWorld[1], nWorld[2]) || 1;
    const nx = nWorld[0] / nlen;
    const ny = nWorld[1] / nlen;
    const nz = nWorld[2] / nlen;

    // 画面 1 画素ぶんの実寸 → ワールド寸法
    const pixelWorld = (p.z / focalPx) * scale;
    // 傾斜面は同じ footprint でも面上では広い。ただし際限なく伸ばすと
    // 輪郭で針のようなサーフェルが出るので上限を掛ける。
    const slant = 1 / Math.max(Math.abs(nz), params.cosLimit);
    const radius = 0.5 * p.footprint * pixelWorld * params.spread * slant;

    const o = k * stride32;
    f32[o] = pos[0];
    f32[o + 1] = pos[1];
    f32[o + 2] = pos[2];
    u32[o + 3] = encodeOct(nx, ny, nz);
    u32[o + 4] = packHalf2(radius, radius);
    u32[o + 5] = packRgba8(p.rgba[0], p.rgba[1], p.rgba[2], p.rgba[3]);

    // レンダラは「カメラからの距離」でソートする。既定カメラは +z 側の
    // 距離 1.0 にいるので、そこからの距離を見ておく。
    const d = 1 - pos[2];
    if (d < outNear) outNear = d;
    if (d > outFar) outFar = d;
  }

  return {
    data: new Uint8Array(buf),
    count: points.length,
    frontCount,
    backCount,
    skirtCount,
    normalization,
    nearZ: Number.isFinite(outNear) ? outNear : 0.5,
    farZ: Number.isFinite(outFar) ? outFar : 1.5,
  };
}
