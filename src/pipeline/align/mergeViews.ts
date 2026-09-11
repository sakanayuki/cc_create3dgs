/**
 * Ⓒ 融合 — 位置合わせした複数の view を、1つのスプラット群にまとめる（docs/12 §12.8）。
 *
 * 各 view が作ったスプラットを姿勢で共通座標へ運び、**重なった面を落として**
 * 1つにする。
 *
 * ## 重複をどう落とすか
 *
 * 設計（§12.8）は「上位 view の深度バッファへ投影して |Δz| < ε なら落とす」と
 * 書いていた。ここでは同じ考えを**点の側**で実装する。深度バッファを持ち回ら
 * なくても、運んだあとの点どうしの距離で同じことが言える。
 *
 *   1. 運んだ点を、間隔ほどの大きさの格子に入れる
 *   2. **別の view から来た点**が、同じあたりに、**同じ向きの面**として既に
 *      あるなら、それは同じ面である
 *   3. 残すのは、**その面をより正面から見ていた view の点**
 *
 * 3 は設計の「面を正面から見ている view の点を優先する」そのものだが、
 * **view の順位ではなく点ごとに決める**。同じ view の中でも、正面から見えて
 * いる面（胸）と、かすめて見えている面（体の側面）が混ざっているからである。
 * 「正面 view を第1位」と固定すると、正面写真がかすめて見ただけの体の側面が、
 * 横向き写真が正面から見た同じ面を追い出してしまう。
 *
 * 見込み角は**カメラ座標の法線の z 成分**でそのまま測れる。`toWorld` が恒等
 * （src/pipeline/6-splats.ts）なので、法線はその view のカメラ座標で入っていて、
 * カメラは +z を向いている。回転は内積を変えないので、運ぶ前に測ってよい。
 *
 * ## 同じ view の点どうしは決して比べない
 *
 * 格子の目は点の間隔ほどしかない。同じ view の隣り合う点を比べると、
 * **1枚の面を間引いてしまう**。落としたいのは view をまたいだ重なりだけである。
 * 先に入れた view の点に対してだけ照合し、同じ view の中では照合しない。
 *
 * ## なぜパック済みのバッファを解いて合成するのか
 *
 * `buildSplats` は 565 行あり、点の生成と詰め込みが地続きになっている。
 * そこを割って共通座標を通す改造もできるが、既存の単一画像モードに手を
 * 入れることになる。**出来上がった 24 バイト × N を解いて座標を移し、詰め直す**。
 * `toWorld` が恒等写像なので、パックされた位置はカメラ座標を正規化しただけの
 * ものであり、正確に戻せる。失うのは符号化の精度だけである
 * （法線は八面体 32bit、半径は half）。
 */
import { decodeOct, encodeOct, packHalf2, unpackHalf2 } from '../../codec/pack';
import type { SplatBuild } from '../6-splats';
import { rotationMatrix, toReference, type Mat3, type PoseFrame, type ViewPose } from './rigid';

/** 24 バイト × N。src/pipeline/6-splats.ts の詰め方と同じ。 */
const SPLAT_BYTES = 24;
const STRIDE32 = SPLAT_BYTES / 4;

export interface MergeSource {
  readonly build: SplatBuild;
  /** この view の点を基準 view の座標へ運ぶ変換。 */
  readonly pose: ViewPose;
  /** 回転の中心（src/pipeline/align/rigid.ts）。 */
  readonly frame: PoseFrame;
}

export interface MergeOptions {
  /** 重複を落とすか。false にすると並べるだけ（調査用）。既定 true。 */
  readonly dedupe?: boolean;
  /**
   * 格子の目を、**体の大きさ**の何割にするか。既定 0.02（2%）。
   *
   * ## 半径（点の間隔）を基準にしてはいけない
   *
   * 最初は「半径の中央値の 1.5 倍」にした。**まったく落ちなかった**
   * （実素材で 0.9%）。点の間隔は体の 0.1% ほどしかないのに、位置合わせの
   * 残差は 128 画素のグリッドで 1 画素ぶん、つまり体の 0.8% ある。
   * **同じ面が、点の間隔の 7 倍以上離れて置かれている。** 間隔ほどの目では
   * 決して届かない（docs/12 §12.16.6）。
   *
   * 設計（§12.8）が「ε は体の奥行きの数%」と書いていたのは正しく、
   * こちらが実装で読み替えを誤っていた。
   *
   * 目を粗くするほど落ちるが、粗くした先で残るのは「どちらか一方の view の
   * 面」なので、**位置合わせの残差より細かい構造は、もともと区別できない**。
   *
   * 実素材（39.4 万点）で振った結果。二重率は「別の view から来た同じ面が
   * 体の 1% 以内にいる点の割合」（docs/12 §12.16.6）。
   *
   * | 目 | 点の数 | 落とした | 二重率 |
   * |---|---|---|---|
   * | 落とさない | 393,523 | 0.0% | 24.7% |
   * | 1.0% | 355,681 | 9.6% | 13.1% |
   * | 1.5% | 341,910 | 13.1% | 6.3% |
   * | **2.0%** | **329,687** | **16.2%** | **2.5%** |
   * | 3.0% | 305,388 | 22.4% | 2.1% |
   * | 5.0% | 270,164 | 31.3% | 0.9% |
   *
   * 2% が折れ目である。ここまでで二重像はほぼ消える。3% にすると点をさらに
   * 6% 落として二重率は 0.4 ぶんしか下がらない。
   */
  readonly cellRatio?: number;
  /**
   * 合成した点が、どの view から来たかを返すか（調査用）。既定 false。
   *
   * 二重像が残っているかは「**別の view から来た**点が近くにいるか」でしか
   * 測れない。同じ view の中の隣り合う点を数えると、落とす前も落とした後も
   * 100% になる（実際にそれで測って無意味な数字を出した。§12.16.6）。
   */
  readonly keepSourceIds?: boolean;
  /**
   * 向かい合った面（薄い部位の表と裏）を分けるか。既定 true。
   *
   * false にすると向きを一切見ない。**体の表と裏を潰すので本番では使えない。**
   * 残っている二重像が向きの扱いのせいかどうかを測るために置いてある。
   */
  readonly splitSides?: boolean;
}

/** 合成の内訳。落とした数を画面と検査に出すため。 */
export interface MergeStats {
  /** 運んだ点の総数（落とす前）。 */
  readonly before: number;
  /** 残った点の数。 */
  readonly after: number;
  /** 落とした点の数。 */
  readonly dropped: number;
  /** 使った格子の目（基準座標の長さ）。 */
  readonly cell: number;
  /** `keepSourceIds` を頼んだときだけ。残った点が、どの view から来たか。 */
  readonly sourceOf?: Uint16Array;
}

/**
 * 重なった面を落とす（docs/12 §12.8）。
 *
 * **小さな区画ごとに、その面をいちばんよく見ている view を1つ選び、
 * そこでは他の view の点を使わない。**
 *
 * 区画は（格子の目 × 法線の向き）で切る。向きで分けるのは、薄い部位で
 * 表と裏が同じ目に入るからで、分けないと体を貫通して潰す。向きは6方向
 * （±x, ±y, ±z のどれに最も近いか）まで丸める。細かく分けるほど「同じ面」の
 * 判定が厳しくなり、落ちなくなる。
 *
 * ## 近傍探索をやめた理由
 *
 * 最初は点ごとに近所を探して1対1で比べた。**格子の目が点の間隔の 20 倍ある
 * ので、1マスに数百点が入り、27 マスぶんの総当たりで実素材 39 万点に 12 秒
 * かかった**（docs/12 §12.16.6）。区画ごとに勝者を決める形なら、走査は
 * 2 回で済んで O(N) になる。
 *
 * 選び方も、こちらのほうが素直である。1対1だと「A が B に勝ち、B が C に
 * 勝ち、C が A に勝つ」が起きうるが、区画ごとなら勝者は1つに決まる。
 */
function dropOverlaps(
  count: number,
  pos: Float64Array,
  nrm: Float64Array,
  source: Uint16Array,
  faceOn: Float32Array,
  cell: number,
  min: readonly [number, number, number],
  splitSides: boolean,
): { readonly keep: Uint8Array; readonly dropped: number } {
  const keep = new Uint8Array(count).fill(1);
  if (!(cell > 0)) return { keep, dropped: 0 };

  // **格子を半マスずらして2回通す。**
  //
  // 1回だけだと、同じ面の2つが区画の境目をまたいだときに別々の区画に入り、
  // どちらも「その区画では唯一の view」として残る。実素材で二重率が 24.7% →
  // 16.8% までしか下がらなかったのはこれである（docs/12 §12.16.6）。
  // 半マスずらした2回目では、境目をまたいでいた対が同じ区画に入る。
  let dropped = 0;
  for (const shift of [0, 0.5]) {
    dropped += onePass(count, pos, nrm, source, faceOn, cell, min, shift, splitSides, keep);
  }
  return { keep, dropped };
}

function onePass(
  count: number,
  pos: Float64Array,
  nrm: Float64Array,
  source: Uint16Array,
  faceOn: Float32Array,
  cell: number,
  min: readonly [number, number, number],
  shift: number,
  splitSides: boolean,
  keep: Uint8Array,
): number {
  const inv = 1 / cell;
  const off = shift * cell;
  const nx = Math.max(1, Math.ceil((maxOf(pos, count, 0) - min[0]) * inv)) + 3;
  const ny = Math.max(1, Math.ceil((maxOf(pos, count, 1) - min[1]) * inv)) + 3;

  /** 場所だけの区画番号。 */
  const spatialKey = (i: number): number => {
    const ix = Math.floor(((pos[i * 3] as number) - min[0] + off) * inv);
    const iy = Math.floor(((pos[i * 3 + 1] as number) - min[1] + off) * inv);
    const iz = Math.floor(((pos[i * 3 + 2] as number) - min[2] + off) * inv);
    return ix + nx * (iy + ny * iz);
  };

  // 1周目: 区画ごとに「いちばん正面から見えている点」を選ぶ。
  //
  // **この点の法線を、その区画の面の向きの代表にする。** 深度から起こした
  // 法線は、かすめて見た面ほどカメラ側へ寄って外れる。いちばん正面から
  // 見えている点の法線が、その区画でいちばん確からしい。
  const seed = new Map<number, number>();
  if (splitSides) {
    for (let i = 0; i < count; i++) {
      if (keep[i] === 0) continue;
      const k = spatialKey(i);
      const cur = seed.get(k);
      if (cur === undefined || (faceOn[i] as number) > (faceOn[cur] as number)) seed.set(k, i);
    }
  }

  /**
   * 区画番号。場所と、代表の法線に対する**表か裏か**で切る。
   *
   * 6方向に丸める案は**外れだった**。同じ面でも view ごとに法線がカメラ側へ
   * 寄るので、45° の境目をまたいで別の区画に入ってしまう。実素材で、向きを
   * 見ない場合は二重率 2.1% まで落ちるのに、6方向に切ると 11.5% で止まった
   * （docs/12 §12.16.6）。
   *
   * 分けないといけないのは**向かい合った面**（薄い部位の表と裏）だけである。
   * 同じ面の推定違い（内積 0 以上）は、まとめてよい。
   */
  const cellKey = (i: number): number => {
    const k = spatialKey(i);
    if (!splitSides) return k * 2;
    const sIdx = seed.get(k);
    if (sIdx === undefined) return k * 2;
    const dot =
      (nrm[i * 3] as number) * (nrm[sIdx * 3] as number) +
      (nrm[i * 3 + 1] as number) * (nrm[sIdx * 3 + 1] as number) +
      (nrm[i * 3 + 2] as number) * (nrm[sIdx * 3 + 2] as number);
    return k * 2 + (dot >= 0 ? 0 : 1);
  };

  // 2周目: 区画ごとに、view ごとの「正面から見ている度合い」を足す。
  //
  // 合計にするのは、**よく見えている view はたいてい点も多い**からである。
  // 平均にすると、かすめて見た面にたまたま数点だけ乗った view が勝ちうる。
  const score = new Map<number, Map<number, number>>();
  for (let i = 0; i < count; i++) {
    if (keep[i] === 0) continue; // 前の周で落ちた点は勘定に入れない
    const k = cellKey(i);
    let per = score.get(k);
    if (!per) {
      per = new Map<number, number>();
      score.set(k, per);
    }
    const si = source[i] as number;
    per.set(si, (per.get(si) ?? 0) + (faceOn[i] as number));
  }

  // 勝者を決める。同点なら添字の小さい view（＝正面に近いほう）。
  const winner = new Map<number, number>();
  for (const [k, per] of score) {
    if (per.size === 1) continue; // 1つの view しか無い区画は触らない
    let best = -1;
    let bestScore = -Infinity;
    for (const [si, v] of per) {
      if (v > bestScore) {
        bestScore = v;
        best = si;
      }
    }
    winner.set(k, best);
  }

  // 3周目: 勝者以外を落とす。
  let dropped = 0;
  for (let i = 0; i < count; i++) {
    if (keep[i] === 0) continue;
    const w = winner.get(cellKey(i));
    if (w === undefined || w === (source[i] as number)) continue;
    keep[i] = 0;
    dropped++;
  }

  return dropped;
}

function maxOf(a: Float64Array, count: number, comp: number): number {
  let m = -Infinity;
  for (let i = 0; i < count; i++) {
    const v = a[i * 3 + comp] as number;
    if (v > m) m = v;
  }
  return m;
}

/**
 * 複数の view のスプラットを、基準 view の座標で1つにまとめる。
 *
 * 正規化は最後に**まとめて1回**やり直す。view ごとの正規化のままつなぐと、
 * 大きさの違う3つが混ざる。
 */
export function mergeBuilds(
  sources: readonly MergeSource[],
  options: MergeOptions = {},
): SplatBuild & { readonly mergeStats: MergeStats } {
  if (sources.length === 0) throw new Error('合成する view がありません');
  if (sources.length === 1) {
    const only = sources[0]?.build as SplatBuild;
    return {
      ...only,
      mergeStats: { before: only.count, after: only.count, dropped: 0, cell: 0 },
    };
  }

  const total = sources.reduce((n, s) => n + s.build.count, 0);
  const pos = new Float64Array(total * 3);
  const nrm = new Float64Array(total * 3);
  const radius = new Float64Array(total);
  const rgba = new Uint32Array(total);
  /** どの view から来たか。同じ view の点どうしを比べないために要る。 */
  const source = new Uint16Array(total);
  /** その view がその面をどれだけ正面から見ていたか（0〜1）。 */
  const faceOn = new Float32Array(total);
  /** 0=前面 1=背面 2=スカート。落としたあとに数え直すため。 */
  const kind = new Uint8Array(total);

  let minX = Infinity;
  let minY = Infinity;
  let minZ = Infinity;

  const tmp = new Float64Array(3);
  let k = 0;

  sources.forEach((src, si) => {
    const { build, pose, frame } = src;
    const R: Mat3 = rotationMatrix(pose);
    const f32 = new Float32Array(build.data.buffer, build.data.byteOffset, build.count * STRIDE32);
    const u32 = new Uint32Array(build.data.buffer, build.data.byteOffset, build.count * STRIDE32);
    const invScale = 1 / build.normalization.scale;
    const c = build.normalization.center;
    const backFrom = build.frontCount;
    const skirtFrom = build.frontCount + build.backCount;

    for (let i = 0; i < build.count; i++) {
      const o = i * STRIDE32;

      // 正規化を戻してカメラ座標へ（toWorld は恒等なのでそのまま）
      const cxx = (f32[o] as number) * invScale + c[0];
      const cyy = (f32[o + 1] as number) * invScale + c[1];
      const czz = (f32[o + 2] as number) * invScale + c[2];

      toReference(R, pose, frame, cxx, cyy, czz, tmp);
      const x = tmp[0] as number;
      const y = tmp[1] as number;
      const z = tmp[2] as number;
      pos[k * 3] = x;
      pos[k * 3 + 1] = y;
      pos[k * 3 + 2] = z;
      if (x < minX) minX = x;
      if (y < minY) minY = y;
      if (z < minZ) minZ = z;

      // 法線は回すだけ（平行移動も尺度も掛けない）
      const n = decodeOct(u32[o + 3] as number);
      nrm[k * 3] = (R[0] as number) * n[0] + (R[1] as number) * n[1] + (R[2] as number) * n[2];
      nrm[k * 3 + 1] = (R[3] as number) * n[0] + (R[4] as number) * n[1] + (R[5] as number) * n[2];
      nrm[k * 3 + 2] = (R[6] as number) * n[0] + (R[7] as number) * n[1] + (R[8] as number) * n[2];

      // **見込み角は運ぶ前に測る。** カメラは +z を向いているので、
      // カメラ座標の法線の z 成分がそのまま「どれだけ正面から見たか」になる。
      // 回転は内積を変えないので、運んだあとに測り直す必要はない。
      faceOn[k] = Math.abs(n[2]);

      // 半径は「その view の正規化が掛かった値」なので、いったん実寸へ戻して
      // 姿勢の尺度を掛ける。最後に共通の正規化を掛け直す。
      radius[k] = unpackHalf2(u32[o + 4] as number)[0] * invScale * pose.scale;
      rgba[k] = u32[o + 5] as number;
      source[k] = si;
      kind[k] = i >= skirtFrom ? 2 : i >= backFrom ? 1 : 0;
      k++;
    }
  });

  // --- 重なった面を落とす
  //
  // 格子の目は**体の大きさ**から決める。点の間隔からではない（MergeOptions 参照）。
  const bodySize = Math.max(
    maxOf(pos, total, 0) - minX,
    maxOf(pos, total, 1) - minY,
    maxOf(pos, total, 2) - minZ,
    1e-9,
  );
  const cell = options.dedupe === false ? 0 : bodySize * (options.cellRatio ?? 0.02);
  const { keep, dropped } =
    cell > 0
      ? dropOverlaps(total, pos, nrm, source, faceOn, cell, [minX, minY, minZ], options.splitSides !== false)
      : { keep: new Uint8Array(total).fill(1), dropped: 0 };

  const count = total - dropped;

  // --- 残った点で境界を取り直す
  //
  // **落とす前の境界を使ってはいけない。** 落とした点が端にいたら、
  // 正規化の中心と倍率がずれ、書き出した実寸もずれる。
  let bMinX = Infinity;
  let bMinY = Infinity;
  let bMinZ = Infinity;
  let bMaxX = -Infinity;
  let bMaxY = -Infinity;
  let bMaxZ = -Infinity;
  let frontCount = 0;
  let backCount = 0;
  let skirtCount = 0;
  for (let i = 0; i < total; i++) {
    if (keep[i] === 0) continue;
    const x = pos[i * 3] as number;
    const y = pos[i * 3 + 1] as number;
    const z = pos[i * 3 + 2] as number;
    if (x < bMinX) bMinX = x;
    if (y < bMinY) bMinY = y;
    if (z < bMinZ) bMinZ = z;
    if (x > bMaxX) bMaxX = x;
    if (y > bMaxY) bMaxY = y;
    if (z > bMaxZ) bMaxZ = z;
    const t = kind[i] as number;
    if (t === 0) frontCount++;
    else if (t === 1) backCount++;
    else skirtCount++;
  }

  // --- まとめて正規化し直す（buildSplats の ③ と同じ規則）
  const center: [number, number, number] =
    count > 0 ? [(bMinX + bMaxX) / 2, (bMinY + bMaxY) / 2, (bMinZ + bMaxZ) / 2] : [0, 0, 0];
  const extent = Math.max(bMaxX - bMinX, bMaxY - bMinY, bMaxZ - bMinZ, 1e-6);
  const scale = count > 0 ? 1 / extent : 1;

  const buf = new ArrayBuffer(count * SPLAT_BYTES);
  const of32 = new Float32Array(buf);
  const ou32 = new Uint32Array(buf);
  let outNear = Infinity;
  let outFar = -Infinity;

  const sourceOf = options.keepSourceIds ? new Uint16Array(count) : null;
  let w = 0;
  for (let i = 0; i < total; i++) {
    if (keep[i] === 0) continue;
    if (sourceOf) sourceOf[w] = source[i] as number;
    const o = w * STRIDE32;
    const px = ((pos[i * 3] as number) - center[0]) * scale;
    const py = ((pos[i * 3 + 1] as number) - center[1]) * scale;
    const pz = ((pos[i * 3 + 2] as number) - center[2]) * scale;
    of32[o] = px;
    of32[o + 1] = py;
    of32[o + 2] = pz;

    const nx = nrm[i * 3] as number;
    const ny = nrm[i * 3 + 1] as number;
    const nz = nrm[i * 3 + 2] as number;
    const nl = Math.hypot(nx, ny, nz) || 1;
    ou32[o + 3] = encodeOct(nx / nl, ny / nl, nz / nl);

    const r = (radius[i] as number) * scale;
    ou32[o + 4] = packHalf2(r, r);
    ou32[o + 5] = rgba[i] as number;

    // レンダラは既定カメラ（+z 側の距離 1.0）からの距離でソートする
    const d = 1 - pz;
    if (d < outNear) outNear = d;
    if (d > outFar) outFar = d;
    w++;
  }

  return {
    data: new Uint8Array(buf),
    count,
    frontCount,
    backCount,
    skirtCount,
    normalization: { center, scale },
    metricHeight: Math.max(bMaxY - bMinY, 0),
    nearZ: Number.isFinite(outNear) ? outNear : 0.5,
    farZ: Number.isFinite(outFar) ? outFar : 1.5,
    mergeStats: { before: total, after: count, dropped, cell, ...(sourceOf ? { sourceOf } : {}) },
  };
}
