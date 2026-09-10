/**
 * Ⓑ 位置合わせ（docs/12 §12.7）。
 *
 * 素材ではカメラが1画素も動いていない（docs/12 §12.2 O1）ので、view の間の差は
 * ほぼ「被写体が鉛直軸まわりに何度回ったか」に尽きる。それでも一般の写真が
 * 入りうるので、未知数は view あたり 7（yaw・pitch・roll・尺度・平行移動3）持つ。
 * InstantSplat の bundle adjustment（全ガウシアン + 全カメラ）とは桁が2つ違うので、
 * 微分可能ラスタライザは要らない（docs/12 D27）。
 *
 * ## 目的関数
 *
 * 3項ある。**どれか1つでは足りないことを、合成データで実際に確かめた**
 * （docs/12 §12.15.2 に失敗の記録）。
 *
 *   1. **収まり**: view i の点を view j へ投影したとき、mask_j の中に入っていること。
 *      外に出たぶんを、被写体までの距離（距離変換）で罰する。
 *   2. **面の一致**: 両方の view が同じ面を見ているところで、深度が合うこと。
 *      **角度と、奥行き方向の位置・尺度の情報は、ほぼこの項が持っている。**
 *      横向き付近ではシルエットの幅が角度に対して停留するので、1 だけでは向きが決まらない。
 *      両側（手前も奥も）を罰するので、被写体を遠ざけたり縮めたりする抜け道も塞がる。
 *      ゲートを超えた差は遮蔽とみなして頭打ちにする（ICP の対応距離の閾値と同じ考え）。
 *      **割る数は「投影した点の総数」で固定する。** マスクに入った点だけで平均すると、
 *      姿勢をずらして合わない点をマスクの外へ追い出すほど平均が下がる、という
 *      抜け道ができる（実際にそれで真値が最小にならなかった）。
 *
 * 3つめとして「覆い」（mask_j が他の view の投影で覆われていること）も実装したが、
 * **測って捨てた**。docs/12 §12.15.2 に記録がある。
 *
 * 距離は**小数の座標で読む**（双一次補間）。横向きの被写体は粗いグリッドで
 * 15〜20 画素しか幅がないので、整数へ丸めると丸めの差が幾何の差を上回る。
 *
 * ## 解き方
 *
 * 枠（docs/12 D24）で yaw の初期値が分かっているので、その周りを粗く走査してから
 * パターン探索で詰める。微分は使わない。決定的に動くので、同じ入力なら同じ結果が出る。
 */
import {
  buildSilhouette,
  iou,
  sampleBilinear,
  type SilhouetteGrid,
} from './silhouette';
import {
  fromReference,
  IDENTITY_FRAME,
  project,
  REFERENCE_POSE,
  rotationMatrix,
  slotYaw,
  toReference,
  unproject,
  type CameraIntrinsics,
  type Mat3,
  type PoseFrame,
  type ViewPose,
  type ViewSlot,
} from './rigid';

/** 位置合わせに渡す1枚ぶん。⓪①②③ を通した後の状態を想定している。 */
export interface AlignView {
  /** UI の枠。yaw の初期値になる。 */
  readonly slot: ViewSlot;
  readonly width: number;
  readonly height: number;
  readonly camera: CameraIntrinsics;
  /** 0〜255。0 は被写体の外。 */
  readonly alpha: ArrayLike<number>;
  /** 画素ごとの深度（カメラからの z、メートル）。α=0 の画素は見ない。 */
  readonly depth: ArrayLike<number>;
  /**
   * 位置合わせに使ってよい画素（胴と頭）。腕・脚は 0 にする（docs/12 §12.7）。
   *
   * 省略すると α だけで判断する。3枚で腕の位置が違う素材では、
   * 省略すると解が腕に引きずられる。
   */
  readonly usable?: ArrayLike<number>;
}

export interface RegisterOptions {
  /** 位置合わせを回すグリッドの長辺。既定 128。 */
  readonly gridLongSide?: number;
  /** view ごとに使う点の数の目安。既定 3000。 */
  readonly samplesPerView?: number;
  /** 粗い走査で yaw を振る幅[rad]。既定 ±40°。 */
  readonly yawSearchRange?: number;
  /** 粗い走査の刻み[rad]。既定 2°。 */
  readonly yawSearchStep?: number;
  /** パターン探索の上限回数。既定 60。 */
  readonly maxRounds?: number;
}

export interface ViewResult {
  readonly slot: ViewSlot;
  readonly pose: ViewPose;
  /** 他の view から運んできた点の、平均はみ出し（粗いグリッドの画素）。 */
  readonly containmentPx: number;
  /**
   * **M1（docs/12 §12.13）。** 他の view から運んできた点のうち、
   * この view のマスクの内側に落ちた割合。合っていれば 1 に近づく。
   */
  readonly insideRatio: number;
  /**
   * 参考値。この view のマスクと、他の view の投影の IoU。
   *
   * **1 にはならない。** 正面のマスクの真ん中（胸の正面）は、左右どちらの
   * 横向きからも見えていないので、他の view の投影では埋まらない。
   * 当初 M1 をこれで定義したが、構造的に届かない閾値になっていた（docs/12 §12.15.3）。
   */
  readonly iou: number;
}

export interface RegisterResult {
  /** 入力と同じ順。基準 view は `REFERENCE_POSE` そのもの。 */
  readonly views: readonly ViewResult[];
  /** 基準にした view の添字。 */
  readonly referenceIndex: number;
  /** 目的関数の最終値。小さいほど良い。 */
  readonly cost: number;
  /** M1 の最小値。合格ラインは 0.95（docs/12 §12.13）。 */
  readonly worstInsideRatio: number;
}

/**
 * 与えた姿勢での目的関数を測る（調査用）。
 *
 * 最適化が真の姿勢を追い越したときに、どの項が悪さをしているかを見るために置いてある。
 * 毎回グリッドと点を作り直すので遅い。生成の経路からは呼ばない。
 */
export function costAtPoses(
  views: readonly AlignView[],
  poses: readonly ViewPose[],
  options: RegisterOptions = {},
): CostParts {
  const gridLongSide = options.gridLongSide ?? 128;
  const samples = options.samplesPerView ?? 3000;
  const grids = views.map((v) =>
    buildSilhouette(v.usable ? maskFromUsable(v) : v.alpha, v.width, v.height, {
      targetLongSide: gridLongSide,
      depth: v.depth,
    }),
  );
  const points = views.map((v) => samplePoints(v, samples));
  const cams = views.map((v) => v.camera);
  const refIndex = Math.max(0, views.findIndex((v) => v.slot === 'front'));
  const refCentroid = (points[refIndex] as ViewPoints).centroid;
  const frames = points.map((pt, i) =>
    i === refIndex ? IDENTITY_FRAME : { source: pt.centroid, target: refCentroid },
  );
  const mats = poses.map((pp) => rotationMatrix(pp));
  const scratch = makeScratch(Math.max(...grids.map((g) => g.width * g.height)));
  return evaluate(points, grids, cams, poses, mats, frames, scratch);
}

/** `usable` と α の論理積を、α と同じ 0〜255 の形で返す。 */
function maskFromUsable(v: AlignView): Uint8Array {
  const n = v.width * v.height;
  const out = new Uint8Array(n);
  const usable = v.usable;
  for (let i = 0; i < n; i++) {
    const a = v.alpha[i] as number;
    if (a >= 128 && usable && (usable[i] as number) !== 0) out[i] = a;
  }
  return out;
}

/** `usable` を外した写し。`exactOptionalPropertyTypes` があるので明示的に組み直す。 */
function withoutUsable(v: AlignView): AlignView {
  return {
    slot: v.slot,
    width: v.width,
    height: v.height,
    camera: v.camera,
    alpha: v.alpha,
    depth: v.depth,
  };
}

/** ある view から取り出した、カメラ座標の点群。 */
interface ViewPoints {
  readonly x: Float64Array;
  readonly y: Float64Array;
  readonly z: Float64Array;
  readonly count: number;
  /** ヨーに対して変わらない縦の長さ（メートル）。尺度の拘束に使う。 */
  readonly verticalSpan: number;
  /** 点の重心（この view のカメラ座標）。回転の中心になる。 */
  readonly centroid: readonly [number, number, number];
}

/**
 * 位置合わせに使う点を間引いて取り出す。
 *
 * α と `usable` の両方が立っている画素だけを使う。狙った個数に近づくよう
 * 一定間隔で拾うので、同じ入力なら必ず同じ点が選ばれる。
 */
function samplePoints(view: AlignView, target: number): ViewPoints {
  const n = view.width * view.height;
  const usable = view.usable;
  let candidates = 0;
  for (let i = 0; i < n; i++) {
    if ((view.alpha[i] as number) < 128) continue;
    if (usable && (usable[i] as number) === 0) continue;
    const z = view.depth[i] as number;
    if (!(z > 0) || !Number.isFinite(z)) continue;
    candidates++;
  }
  const stride = Math.max(1, Math.floor(candidates / Math.max(1, target)));

  const x = new Float64Array(Math.ceil(candidates / stride) + 1);
  const y = new Float64Array(x.length);
  const z = new Float64Array(x.length);
  const tmp = new Float64Array(3);

  let seen = 0;
  let count = 0;
  let vTop = Infinity;
  let vBottom = -Infinity;
  let zSum = 0;
  let cx = 0;
  let cy = 0;
  let cz = 0;
  for (let v = 0; v < view.height; v++) {
    for (let u = 0; u < view.width; u++) {
      const i = v * view.width + u;
      if ((view.alpha[i] as number) < 128) continue;
      if (usable && (usable[i] as number) === 0) continue;
      const d = view.depth[i] as number;
      if (!(d > 0) || !Number.isFinite(d)) continue;
      if (v < vTop) vTop = v;
      if (v > vBottom) vBottom = v;
      zSum += d;
      if (seen++ % stride !== 0 || count >= x.length) continue;
      unproject(u + 0.5, v + 0.5, d, view.camera, tmp);
      x[count] = tmp[0] as number;
      y[count] = tmp[1] as number;
      z[count] = tmp[2] as number;
      cx += tmp[0] as number;
      cy += tmp[1] as number;
      cz += tmp[2] as number;
      count++;
    }
  }

  // ヨー不変の縦の長さ（docs/12 §12.7、O8 で身長から差し替えた）。
  // 回っても縦の広がりは変わらないので、view 間の尺度をこれで縛れる。
  const zMean = candidates > 0 ? zSum / candidates : 1;
  const verticalSpan =
    candidates > 0 ? ((vBottom - vTop + 1) * zMean) / view.camera.focalPx : 0;

  const inv = count > 0 ? 1 / count : 0;
  return { x, y, z, count, verticalSpan, centroid: [cx * inv, cy * inv, cz * inv] };
}

/** 投影の覆いを描くための作業領域。評価のたびに使い回す。 */
interface Scratch {
  readonly coverage: Uint8Array;
  readonly point: Float64Array;
  readonly ref: Float64Array;
  readonly local: Float64Array;
  readonly proj: Float64Array;
}

function makeScratch(size: number): Scratch {
  return {
    coverage: new Uint8Array(size),
    point: new Float64Array(3),
    ref: new Float64Array(3),
    local: new Float64Array(3),
    proj: new Float64Array(3),
  };
}

/** 目的関数の内訳。デバッグと記録のために分けて返す。 */
export interface CostParts {
  readonly containment: number;
  /** 共視した面の深度差（ゲートで頭打ち）。 */
  readonly surface: number;
  readonly uncovered: number;
  /** 投影の縦の広がりと、マスクの縦の広がりの食い違い。 */
  readonly extent: number;
  readonly total: number;
}

/**
 * 面の一致とみなす深度差の上限（メートル）。ICP の対応距離の閾値にあたる。
 *
 * これを超えた差は「別の面を見ている（遮蔽）」とみなして、それ以上は罰を増やさない。
 * 超えたぶんまで正直に足すと、遮蔽の多い横向き同士で解が壊れる。
 */
const SURFACE_GATE = 0.08;

/**
 * 項の重み。
 *
 * **覆いの項は 0 にしてある。** 実装して測ったところ、この項は
 * 「投影が広がるほど良くなる」向きに効き、しかも他の項の 18 倍の大きさがあって
 * 全部を押し流していた。yaw を 90° から離すほど下がり続け、真の姿勢が最小に
 * ならなかった。捨てた記録は docs/12 §12.15.2 に残す。数字は診断に出したいので
 * 計算だけは続ける（グリッドの大きさに比例するだけで、点の数には依らない）。
 */
const W_CONTAINMENT = 1;
const W_SURFACE = 0;
const W_UNCOVERED = 0;
/**
 * 縦の広がりを合わせる錨の重み。
 *
 * これは「投影が縮む・遠ざかる」のを止めるためだけに要る。角度は決めさせない。
 * 0.5 にしたら、縦の広がりの微妙な yaw 依存（見える点が変わるので完全に不変では
 * ない）が収まりの谷を 8° ずらした。潰れは桁違いに大きい効果なので、
 * 錨としてはこれで十分効く。
 */
const W_EXTENT = 0.15;

/**
 * 目的関数。
 *
 * `containment` は「はみ出した距離の平均 ÷ 被写体の大きさ」、
 * `surface` は「共視した面の深度差 ÷ ゲート」、
 * `uncovered` は「他の view に覆われなかった mask の割合」。いずれも 0 が最良。
 */
function evaluate(
  points: readonly ViewPoints[],
  grids: readonly SilhouetteGrid[],
  cams: readonly CameraIntrinsics[],
  poses: readonly ViewPose[],
  mats: readonly Mat3[],
  frames: readonly PoseFrame[],
  scratch: Scratch,
): CostParts {
  const nViews = points.length;
  let outSum = 0;
  let outCount = 0;
  let freeSum = 0;
  let uncoveredSum = 0;
  let targets = 0;
  let extentSum = 0;
  let extentCount = 0;

  for (let j = 0; j < nViews; j++) {
    const gj = grids[j] as SilhouetteGrid;
    const cj = cams[j] as CameraIntrinsics;
    const pj = poses[j] as ViewPose;
    const mj = mats[j] as Mat3;
    const fj = frames[j] as PoseFrame;
    const cov = scratch.coverage.subarray(0, gj.width * gj.height);
    cov.fill(0);
    // 投影した点の縦の広がり。ヨーでは変わらないので、尺度と奥行き位置の錨になる。
    let projTop = Infinity;
    let projBottom = -Infinity;

    for (let i = 0; i < nViews; i++) {
      if (i === j) continue; // 自分の点は必ず自分を覆うので、覆いの判定に入れない
      const pi = points[i] as ViewPoints;
      const posei = poses[i] as ViewPose;
      const mi = mats[i] as Mat3;
      const fi = frames[i] as PoseFrame;

      for (let k = 0; k < pi.count; k++) {
        toReference(mi, posei, fi, pi.x[k] as number, pi.y[k] as number, pi.z[k] as number, scratch.ref);
        fromReference(
          mj,
          pj,
          fj,
          scratch.ref[0] as number,
          scratch.ref[1] as number,
          scratch.ref[2] as number,
          scratch.local,
        );
        const ok = project(
          scratch.local[0] as number,
          scratch.local[1] as number,
          scratch.local[2] as number,
          cj,
          scratch.proj,
        );
        outCount++;
        if (!ok) {
          outSum += 1; // カメラの後ろ。最大の罰
          freeSum += 1;
          continue;
        }
        // グリッド座標（小数のまま）
        const gx = (scratch.proj[0] as number) * gj.scale - 0.5;
        const gy = (scratch.proj[1] as number) * gj.scale - 0.5;
        if (gx < -1 || gy < -1 || gx > gj.width || gy > gj.height) {
          outSum += 1; // 画面の外。同じく最大の罰
          freeSum += 1;
          continue;
        }
        // 1. 収まり。はみ出しは横向きに起きるので、被写体の幅で割る。
        // 横向きの view は幅が狭いので、同じ 1 画素のはみ出しでも重く効く。それが正しい。
        const d = sampleBilinear(gj.distanceToSubject, gj.width, gj.height, gx, gy);
        outSum += Math.min(1, d / gj.bodyWidth);
        if (gy < projTop) projTop = gy;
        if (gy > projBottom) projBottom = gy;

        const cx = Math.round(gx);
        const cy = Math.round(gy);
        if (cx < 0 || cy < 0 || cx >= gj.width || cy >= gj.height) {
          freeSum += 1;
          continue;
        }
        const idx = cy * gj.width + cx;
        cov[idx] = 1;

        // 2. 面の一致。両方の view が同じ面を見ているところでは、深度が合うはず。
        // 片側（手前だけ）を罰すると「面を奥へ逃がすほど良い」という抜け道ができる
        // ので、両側を見る。ゲートを超えた差は遮蔽とみなして頭打ちにする。
        const zSeen = gj.depth[idx] as number;
        if (zSeen > 0) {
          const diff = Math.abs(zSeen - (scratch.proj[2] as number));
          freeSum += Math.min(SURFACE_GATE, diff) / SURFACE_GATE;
        } else {
          freeSum += 1; // マスクの外に落ちた。合っていないので最大の罰
        }
      }
    }

    // 3. 覆い。点はまばらなので、1画素ぶん膨らませてから数える。
    let uncovered = 0;
    for (let y = 0; y < gj.height; y++) {
      for (let x = 0; x < gj.width; x++) {
        const idx = y * gj.width + x;
        if ((gj.mask[idx] as number) === 0) continue;
        let hit = false;
        for (let dy = -1; dy <= 1 && !hit; dy++) {
          const yy = y + dy;
          if (yy < 0 || yy >= gj.height) continue;
          for (let dx = -1; dx <= 1; dx++) {
            const xx = x + dx;
            if (xx < 0 || xx >= gj.width) continue;
            if ((cov[yy * gj.width + xx] as number) !== 0) {
              hit = true;
              break;
            }
          }
        }
        if (!hit) uncovered++;
      }
    }
    if (gj.area > 0) {
      uncoveredSum += uncovered / gj.area;
      targets++;
    }

    // 3. 縦の広がりの一致。投影が縮む・遠ざかるのを止める錨。
    // 縦はヨーで変わらないので、この項が角度の解を引っぱることはない。
    if (projBottom > projTop) {
      const got = projBottom - projTop;
      extentSum += Math.abs(got - gj.bodyHeight) / gj.bodyHeight;
    } else {
      extentSum += 1;
    }
    extentCount++;
  }

  const containment = outCount > 0 ? outSum / outCount : 0;
  const surface = outCount > 0 ? freeSum / outCount : 1;
  const uncoveredRatio = targets > 0 ? uncoveredSum / targets : 0;
  const extent = extentCount > 0 ? extentSum / extentCount : 0;
  return {
    containment,
    surface,
    uncovered: uncoveredRatio,
    extent,
    total:
      W_CONTAINMENT * containment +
      W_SURFACE * surface +
      W_UNCOVERED * uncoveredRatio +
      W_EXTENT * extent,
  };
}

const PARAM_KEYS = ['yaw', 'pitch', 'roll', 'scale', 'tx', 'ty', 'tz'] as const;
type ParamKey = (typeof PARAM_KEYS)[number];

/**
 * 尺度が初期値から離れてよい割合。
 *
 * 尺度はヨー不変の縦の長さから決まっていて（§12.7）、探索で動かすものではない。
 * docs/12 は「尺度を最適化から外せるので解が安定する」と書いた。完全に外すと
 * 縦の長さの測り方の誤差を直せないので、±10% だけ許す。
 * 実素材で外したまま回したら、片方の view が 1.45 倍まで膨らんで壊れた
 * （docs/12 §12.15.3）。
 */
const SCALE_TOLERANCE = 0.1;

/**
 * pitch と roll の上限[rad]。
 *
 * 立っている人を手持ちで撮った3枚で、カメラの傾きがこれを超えることはない。
 * 上限を置かないと、シルエットの食い違いを傾きで言い訳する解に落ちる
 * （実素材で 21° の pitch が出た）。
 */
const TILT_LIMIT = (12 * Math.PI) / 180;

/** パターン探索の刻み。角度は rad、平行移動はメートル。 */
const INITIAL_STEP: Readonly<Record<ParamKey, number>> = {
  yaw: (4 * Math.PI) / 180,
  pitch: (3 * Math.PI) / 180,
  roll: (3 * Math.PI) / 180,
  scale: 0.02,
  tx: 0.03,
  ty: 0.03,
  tz: 0.05,
};

function withParam(pose: ViewPose, key: ParamKey, value: number): ViewPose {
  return { ...pose, [key]: value };
}

/** 探索を許す範囲に入っているか。外れた手は試さない。 */
function withinBounds(pose: ViewPose, initialScale: number): boolean {
  if (Math.abs(pose.pitch) > TILT_LIMIT) return false;
  if (Math.abs(pose.roll) > TILT_LIMIT) return false;
  const lo = initialScale * (1 - SCALE_TOLERANCE);
  const hi = initialScale * (1 + SCALE_TOLERANCE);
  return pose.scale >= lo && pose.scale <= hi;
}

/**
 * 3枚（または2枚）の位置合わせを解く。
 *
 * 基準は `slot === 'front'` の view。無ければシルエットの面積が最大の view
 * （docs/12 §12.7「2枚のとき」）。基準の姿勢は動かさない。
 */
export function registerViews(views: readonly AlignView[], options: RegisterOptions = {}): RegisterResult {
  if (views.length < 2) throw new Error('位置合わせには2枚以上が要ります');

  const gridLongSide = options.gridLongSide ?? 128;
  const samples = options.samplesPerView ?? 3000;
  const yawRange = options.yawSearchRange ?? (40 * Math.PI) / 180;
  const yawStep = options.yawSearchStep ?? (2 * Math.PI) / 180;
  const maxRounds = options.maxRounds ?? 60;

  // **合わせるときは、点もマスクも「胴と頭」で揃える。**
  // 胴だけの点を全身のマスクと比べると、収まりは甘くなり、縦の広がりは
  // 40% も食い違って、錨がでたらめな向きに効く。実素材で実際にそうなった
  // （docs/12 §12.15.3）。世界の見え方は最適化の中で一貫していないといけない。
  const grids = views.map((v) =>
    buildSilhouette(v.usable ? maskFromUsable(v) : v.alpha, v.width, v.height, {
      targetLongSide: gridLongSide,
      depth: v.depth,
    }),
  );
  // 測るときは被写体全体で。合格の判定は「全身がどれだけ説明できたか」である。
  const measureGrids = views.some((v) => v.usable)
    ? views.map((v) =>
        buildSilhouette(v.alpha, v.width, v.height, { targetLongSide: gridLongSide, depth: v.depth }),
      )
    : grids;
  // **合わせに使う点と、測る点は別である。**
  // 合わせるのは胴と頭だけ（usable、docs/12 §12.7）。しかし M1 は被写体全体の
  // マスクに対して測らないと、胴だけの投影が全身のマスクを覆えるはずもなく、
  // 「腕を外すと M1 が下がる」という中身のない結果が出る。最初それで測って
  // 混乱した（docs/12 §12.15.3）。
  const points = views.map((v) => samplePoints(v, samples));
  const measurePoints = views.some((v) => v.usable)
    ? views.map((v) => samplePoints(withoutUsable(v), samples))
    : points;
  const cams = views.map((v) => v.camera);

  let referenceIndex = views.findIndex((v) => v.slot === 'front');
  if (referenceIndex < 0) {
    let best = -1;
    let bestArea = -1;
    grids.forEach((g, i) => {
      if (g.area > bestArea) {
        bestArea = g.area;
        best = i;
      }
    });
    referenceIndex = Math.max(0, best);
  }

  // 初期姿勢。yaw は枠から、尺度はヨー不変の縦の長さから決める（docs/12 §12.7）。
  const refSpan = (points[referenceIndex] as ViewPoints).verticalSpan;
  const poses: ViewPose[] = views.map((v, i) => {
    if (i === referenceIndex) return REFERENCE_POSE;
    const span = (points[i] as ViewPoints).verticalSpan;
    const scale = span > 1e-6 && refSpan > 1e-6 ? refSpan / span : 1;
    return { ...REFERENCE_POSE, yaw: slotYaw(v.slot) - slotYaw((views[referenceIndex] as AlignView).slot), scale };
  });
  const mats = poses.map((p) => rotationMatrix(p));
  const initialScale = poses.map((p) => p.scale);

  // 回転の中心。その view の重心を、基準 view の重心へ運ぶ（rigid.ts 冒頭）。
  const refCentroid = (points[referenceIndex] as ViewPoints).centroid;
  const frames: PoseFrame[] = points.map((p, i) =>
    i === referenceIndex ? IDENTITY_FRAME : { source: p.centroid, target: refCentroid },
  );

  const maxCells = Math.max(...grids.map((g) => g.width * g.height), ...measureGrids.map((g) => g.width * g.height));
  const scratch = makeScratch(maxCells);

  const cost = (): number => evaluate(points, grids, cams, poses, mats, frames, scratch).total;

  // 段1: 枠の周りで yaw を粗く走査する。view ごとに、他は初期値のまま動かさない。
  for (let i = 0; i < views.length; i++) {
    if (i === referenceIndex) continue;
    const base = poses[i] as ViewPose;
    let bestYaw = base.yaw;
    let bestCost = Infinity;
    for (let d = -yawRange; d <= yawRange + 1e-9; d += yawStep) {
      poses[i] = { ...base, yaw: base.yaw + d };
      mats[i] = rotationMatrix(poses[i] as ViewPose);
      const c = cost();
      if (c < bestCost) {
        bestCost = c;
        bestYaw = base.yaw + d;
      }
    }
    poses[i] = { ...base, yaw: bestYaw };
    mats[i] = rotationMatrix(poses[i] as ViewPose);
  }

  // 段2: 全 view・全パラメータをまとめてパターン探索で詰める。
  const steps: Record<ParamKey, number> = { ...INITIAL_STEP };
  let current = cost();
  for (let round = 0; round < maxRounds; round++) {
    let improved = false;
    for (let i = 0; i < views.length; i++) {
      if (i === referenceIndex) continue;
      for (const key of PARAM_KEYS) {
        const base = poses[i] as ViewPose;
        const step = steps[key];
        for (const sign of [1, -1]) {
          const trial = withParam(base, key, base[key] + sign * step);
          if (!withinBounds(trial, initialScale[i] as number)) continue;
          const keep = poses[i] as ViewPose;
          const keepMat = mats[i] as Mat3;
          poses[i] = trial;
          mats[i] = rotationMatrix(trial);
          const c = cost();
          if (c < current - 1e-9) {
            current = c;
            improved = true;
            break; // この向きで良くなったので、次のパラメータへ
          }
          poses[i] = keep;
          mats[i] = keepMat;
        }
      }
    }
    if (!improved) {
      let allSmall = true;
      for (const key of PARAM_KEYS) {
        steps[key] *= 0.5;
        if (steps[key] > INITIAL_STEP[key] * 0.02) allSmall = false;
      }
      if (allSmall) break;
    }
  }

  // 仕上げに M1（IoU）と収まりを view ごとに出す。合格の判定はこちらで行う。
  const parts = evaluate(points, grids, cams, poses, mats, frames, scratch);
  // 測るときは被写体全体の点を使う（上のコメント）。重心（frames）は合わせに
  // 使った点のものをそのまま使う。姿勢はそれを前提に解いてあるため。
  const results: ViewResult[] = views.map((v, j) => ({
    slot: v.slot,
    pose: poses[j] as ViewPose,
    containmentPx: measureContainment(measurePoints, measureGrids, cams, poses, mats, frames, scratch, j)
      .meanSpill,
    insideRatio: measureContainment(measurePoints, measureGrids, cams, poses, mats, frames, scratch, j)
      .insideRatio,
    iou: measureIou(measurePoints, measureGrids, cams, poses, mats, frames, scratch, j),
  }));

  return {
    views: results,
    referenceIndex,
    cost: parts.total,
    worstInsideRatio: Math.min(...results.map((r) => r.insideRatio)),
  };
}

/** view j のマスクと、他の view の投影の IoU（M1）。 */
function measureIou(
  points: readonly ViewPoints[],
  grids: readonly SilhouetteGrid[],
  cams: readonly CameraIntrinsics[],
  poses: readonly ViewPose[],
  mats: readonly Mat3[],
  frames: readonly PoseFrame[],
  scratch: Scratch,
  j: number,
): number {
  const gj = grids[j] as SilhouetteGrid;
  const cov = renderCoverage(points, grids, cams, poses, mats, frames, scratch, j);
  return iou(cov, gj.mask);
}

/** view j のマスクからのはみ出し。`insideRatio` が M1。 */
function measureContainment(
  points: readonly ViewPoints[],
  grids: readonly SilhouetteGrid[],
  cams: readonly CameraIntrinsics[],
  poses: readonly ViewPose[],
  mats: readonly Mat3[],
  frames: readonly PoseFrame[],
  scratch: Scratch,
  j: number,
): { readonly meanSpill: number; readonly insideRatio: number } {
  const gj = grids[j] as SilhouetteGrid;
  const cj = cams[j] as CameraIntrinsics;
  const pj = poses[j] as ViewPose;
  const mj = mats[j] as Mat3;
  const fj = frames[j] as PoseFrame;
  let sum = 0;
  let count = 0;
  let inside = 0;
  for (let i = 0; i < points.length; i++) {
    if (i === j) continue;
    const pi = points[i] as ViewPoints;
    const posei = poses[i] as ViewPose;
    const mi = mats[i] as Mat3;
    const fi = frames[i] as PoseFrame;
    for (let k = 0; k < pi.count; k++) {
      toReference(mi, posei, fi, pi.x[k] as number, pi.y[k] as number, pi.z[k] as number, scratch.ref);
      fromReference(mj, pj, fj, scratch.ref[0] as number, scratch.ref[1] as number, scratch.ref[2] as number, scratch.local);
      const ok = project(scratch.local[0] as number, scratch.local[1] as number, scratch.local[2] as number, cj, scratch.proj);
      count++;
      if (!ok) {
        sum += Math.hypot(gj.width, gj.height);
        continue;
      }
      const gx = Math.round((scratch.proj[0] as number) * gj.scale - 0.5);
      const gy = Math.round((scratch.proj[1] as number) * gj.scale - 0.5);
      if (gx < 0 || gy < 0 || gx >= gj.width || gy >= gj.height) {
        sum += Math.hypot(gj.width, gj.height);
        continue;
      }
      const d = gj.distanceToSubject[gy * gj.width + gx] as number;
      sum += d;
      if (d === 0) inside++;
    }
  }
  return {
    meanSpill: count > 0 ? sum / count : 0,
    insideRatio: count > 0 ? inside / count : 0,
  };
}

/** view j のグリッドに、他の view の投影を描く（1画素膨張つき）。 */
function renderCoverage(
  points: readonly ViewPoints[],
  grids: readonly SilhouetteGrid[],
  cams: readonly CameraIntrinsics[],
  poses: readonly ViewPose[],
  mats: readonly Mat3[],
  frames: readonly PoseFrame[],
  scratch: Scratch,
  j: number,
): Uint8Array {
  const gj = grids[j] as SilhouetteGrid;
  const cj = cams[j] as CameraIntrinsics;
  const pj = poses[j] as ViewPose;
  const mj = mats[j] as Mat3;
  const fj = frames[j] as PoseFrame;
  const raw = new Uint8Array(gj.width * gj.height);
  for (let i = 0; i < points.length; i++) {
    if (i === j) continue;
    const pi = points[i] as ViewPoints;
    const posei = poses[i] as ViewPose;
    const mi = mats[i] as Mat3;
    const fi = frames[i] as PoseFrame;
    for (let k = 0; k < pi.count; k++) {
      toReference(mi, posei, fi, pi.x[k] as number, pi.y[k] as number, pi.z[k] as number, scratch.ref);
      fromReference(mj, pj, fj, scratch.ref[0] as number, scratch.ref[1] as number, scratch.ref[2] as number, scratch.local);
      if (!project(scratch.local[0] as number, scratch.local[1] as number, scratch.local[2] as number, cj, scratch.proj)) continue;
      const gx = Math.round((scratch.proj[0] as number) * gj.scale - 0.5);
      const gy = Math.round((scratch.proj[1] as number) * gj.scale - 0.5);
      if (gx < 0 || gy < 0 || gx >= gj.width || gy >= gj.height) continue;
      raw[gy * gj.width + gx] = 1;
    }
  }
  // まばらさを埋めるための1画素膨張。M1 を甘くしないよう、膨張は1回だけ。
  const out = new Uint8Array(raw.length);
  for (let y = 0; y < gj.height; y++) {
    for (let x = 0; x < gj.width; x++) {
      let hit = 0;
      for (let dy = -1; dy <= 1 && !hit; dy++) {
        const yy = y + dy;
        if (yy < 0 || yy >= gj.height) continue;
        for (let dx = -1; dx <= 1; dx++) {
          const xx = x + dx;
          if (xx < 0 || xx >= gj.width) continue;
          if ((raw[yy * gj.width + xx] as number) !== 0) {
            hit = 1;
            break;
          }
        }
      }
      out[y * gj.width + x] = hit;
    }
  }
  return out;
}
