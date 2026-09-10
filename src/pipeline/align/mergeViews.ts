/**
 * Ⓒ 融合 — 位置合わせした複数の view を、1つのスプラット群にまとめる（docs/12 §12.8）。
 *
 * **いまは重複を落とさない。** 各 view が作ったスプラットを、姿勢で共通座標へ
 * 運んで並べるだけである。docs/12 §12.8 が書いた co-visibility による重複除去は
 * まだ入っていないので、
 *
 *   ・点の数が view の数だけ増える（3枚なら約 126 万）
 *   ・両方の view から見えている面は二重に置かれる
 *   ・view ごとの露出差がそのまま継ぎ目の色差として出る
 *
 * それでも先にこれを作るのは、**写真3枚から立体が出るところまで一度通して
 * 目で見るため**である。重複除去と色合わせは、実物の見え方を見てから詰める。
 *
 * ## なぜパック済みのバッファを解いて合成するのか
 *
 * `buildSplats` は 565 行あり、点の生成と詰め込みが地続きになっている。
 * そこを割って共通座標を通す改造もできるが、既存の単一画像モードに手を
 * 入れることになる。いまは「見せる」ことが目的なので、**出来上がった
 * 24 バイト × N を解いて座標を移し、詰め直す**。`toWorld` が恒等写像
 * （src/pipeline/6-splats.ts）なので、パックされた位置はカメラ座標を
 * 正規化しただけのものであり、正確に戻せる。
 *
 * 失うのは符号化の精度だけである（法線は八面体 32bit、半径は half）。
 * 重複除去を入れるときに、点の段階で合成する形へ作り直す。
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

/**
 * 複数の view のスプラットを、基準 view の座標で1つにまとめる。
 *
 * 正規化は最後に**まとめて1回**やり直す。view ごとの正規化のままつなぐと、
 * 大きさの違う3つが混ざる。
 */
export function mergeBuilds(sources: readonly MergeSource[]): SplatBuild {
  if (sources.length === 0) throw new Error('合成する view がありません');
  if (sources.length === 1) return sources[0]?.build as SplatBuild;

  const total = sources.reduce((n, s) => n + s.build.count, 0);
  const pos = new Float64Array(total * 3);
  const nrm = new Float64Array(total * 3);
  const radius = new Float64Array(total);
  const rgba = new Uint32Array(total);

  let frontCount = 0;
  let backCount = 0;
  let skirtCount = 0;
  let minX = Infinity;
  let minY = Infinity;
  let minZ = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  let maxZ = -Infinity;

  const tmp = new Float64Array(3);
  let k = 0;

  for (const src of sources) {
    const { build, pose, frame } = src;
    const R: Mat3 = rotationMatrix(pose);
    const f32 = new Float32Array(build.data.buffer, build.data.byteOffset, build.count * STRIDE32);
    const u32 = new Uint32Array(build.data.buffer, build.data.byteOffset, build.count * STRIDE32);
    const invScale = 1 / build.normalization.scale;
    const c = build.normalization.center;

    frontCount += build.frontCount;
    backCount += build.backCount;
    skirtCount += build.skirtCount;

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
      if (x > maxX) maxX = x;
      if (y > maxY) maxY = y;
      if (z > maxZ) maxZ = z;

      // 法線は回すだけ（平行移動も尺度も掛けない）
      const n = decodeOct(u32[o + 3] as number);
      const nx = (R[0] as number) * n[0] + (R[1] as number) * n[1] + (R[2] as number) * n[2];
      const ny = (R[3] as number) * n[0] + (R[4] as number) * n[1] + (R[5] as number) * n[2];
      const nz = (R[6] as number) * n[0] + (R[7] as number) * n[1] + (R[8] as number) * n[2];
      nrm[k * 3] = nx;
      nrm[k * 3 + 1] = ny;
      nrm[k * 3 + 2] = nz;

      // 半径は「その view の正規化が掛かった値」なので、いったん実寸へ戻して
      // 姿勢の尺度を掛ける。最後に共通の正規化を掛け直す。
      const r = unpackHalf2(u32[o + 4] as number)[0];
      radius[k] = r * invScale * pose.scale;

      rgba[k] = u32[o + 5] as number;
      k++;
    }
  }

  // --- まとめて正規化し直す（buildSplats の ③ と同じ規則）
  const center: [number, number, number] =
    total > 0 ? [(minX + maxX) / 2, (minY + maxY) / 2, (minZ + maxZ) / 2] : [0, 0, 0];
  const extent = Math.max(maxX - minX, maxY - minY, maxZ - minZ, 1e-6);
  const scale = total > 0 ? 1 / extent : 1;

  const buf = new ArrayBuffer(total * SPLAT_BYTES);
  const of32 = new Float32Array(buf);
  const ou32 = new Uint32Array(buf);
  let outNear = Infinity;
  let outFar = -Infinity;

  for (let i = 0; i < total; i++) {
    const o = i * STRIDE32;
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
  }

  return {
    data: new Uint8Array(buf),
    count: total,
    frontCount,
    backCount,
    skirtCount,
    normalization: { center, scale },
    metricHeight: Math.max(maxY - minY, 0),
    nearZ: Number.isFinite(outNear) ? outNear : 0.5,
    farZ: Number.isFinite(outFar) ? outFar : 1.5,
  };
}
