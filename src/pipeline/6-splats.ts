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
   * docs/03 §3.6.1 は 1.4。小さいと隙間が空き、大きいとぼける。
   */
  readonly spread: number;
  /** 傾斜面での引き伸ばしの上限。`1/max(|n·v|, cosLimit)`。 */
  readonly cosLimit: number;
  /** 背面シェルを作るか。 */
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
}

export const DEFAULT_BUILD_PARAMS: BuildParams = {
  spread: 1.4,
  cosLimit: 0.3,
  backShell: true,
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

/** パイプライン側カメラ空間 → 描画側ワールド空間の回転（y と z を反転）。 */
function toWorld(p: readonly [number, number, number]): [number, number, number] {
  return [p[0], -p[1], -p[2]];
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
    const gapLimit = params.skirtThreshold;
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

        // 色は段差の奥側の画素から採る（インペイントが無いときの縮退）。
        const fi = (y + gy) * width + (x + gx);
        const cr = color[fi * 4] as number;
        const cg = color[fi * 4 + 1] as number;
        const cb = color[fi * 4 + 2] as number;

        for (let s = 1; s <= steps; s++) {
          const t = s / (steps + 1);
          const z = zNear + t * (zFar - zNear);
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
