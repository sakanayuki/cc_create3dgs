/**
 * 複数枚モードの司令塔（docs/12 §12.5）。
 *
 * 写真を 2〜3 枚受け取り、1枚ずつ既存の `generate()` に通してから、
 * 位置合わせ（Ⓑ）で姿勢を解き、共通座標へまとめて（Ⓒ）1つの立体にする。
 *
 * ## いまの範囲
 *
 * **「まず動かして見せる」ところまで。** docs/12 §12.5 のうち、
 *
 *   ⓪①②③ … 既存の `generate()` をそのまま view ごとに回す（品質の工夫は全部効く）
 *   Ⓐ Ⓑ  … `registerViews()`。実素材で M1 = 0.985 まで確かめてある（§12.15.5）
 *   Ⓒ    … `mergeBuilds()`。**重複除去はまだ無い。並べるだけ**
 *   Ⓓ    … **まだ無い**（色合わせ）
 *
 * したがって点の数は view の数だけ増え、両方から見えている面は二重に置かれる。
 * 実物の見え方を見てから、重複除去と色合わせを詰める。
 *
 * ## 時間
 *
 * ②（深度推定）が支配的で、それを枚数ぶん繰り返す。3枚なら単純に 3 倍近い。
 * docs/12 §12.12 の見積もりでは WebGPU で 12.6 秒、位置合わせがさらに 2.6 秒。
 * **10 秒の SLO には収まらない。** 決定 D30 で複数枚モードは別 SLO にしてある。
 */
import { generate, type GenerateOptions, type GenerateResult, type SubjectMode } from './generate';
import { mergeBuilds, type MergeSource } from './align/mergeViews';
import { registerViews, type AlignView, type RegisterResult } from './align/registerViews';
import { torsoAndHead } from './align/bodyParts';
import { type ViewSlot } from './align/rigid';
import type { SplatBuild } from './6-splats';
import type { Backend } from '../runtime/OrtSession';

/** UI の枠1つぶん。`slot` は利用者がどの枠に入れたか（docs/12 D24）。 */
export interface MultiPhoto {
  readonly slot: ViewSlot;
  readonly blob: Blob;
}

export interface GenerateMultiOptions {
  readonly mode: SubjectMode;
  readonly grid: number;
  readonly reduction: number;
  readonly backend: Backend;
  readonly shaderF16?: boolean;
  readonly onProgress?: (fraction: number, label: string) => void;
  /** 1枚目（正面）ができた時点で1回呼ばれる。待たせずに見せるため。 */
  readonly onPreview?: (build: SplatBuild) => void;
  readonly inpaint?: boolean;
  readonly depthTiles?: boolean;
  readonly fineGridRatio?: number;
}

export interface GenerateMultiResult {
  readonly build: SplatBuild;
  /** view ごとの生成結果。統計の表示に使う。 */
  readonly views: readonly { readonly slot: ViewSlot; readonly result: GenerateResult }[];
  readonly registration: RegisterResult;
  readonly metricFix: number;
}

const SLOT_LABEL: Record<ViewSlot, string> = {
  front: '正面',
  right: '右向き',
  left: '左向き',
};

/**
 * view ごとの結果から、位置合わせの入力を作る。
 *
 * 深度は 0..1 で返ってくるので実距離へ戻す。位置合わせは焦点距離と深度が
 * 噛み合っていることを前提にしていて、そこがずれると透視が狂う
 * （docs/12 §12.15.4 で踏んだ）。`generate()` は ③ の較正を通しているので、
 * その `depthRange` を使えば噛み合う。
 */
function toAlignView(slot: ViewSlot, r: GenerateResult, grid: number): AlignView {
  const span = r.depthRange.farZ - r.depthRange.nearZ;
  const depth = new Float32Array(grid * grid);
  const alpha = r.planes.alpha;
  for (let i = 0; i < depth.length; i++) {
    depth[i] =
      (alpha[i] as number) >= 128 ? r.depthRange.nearZ + (r.planes.depth[i] as number) * span : 0;
  }
  return {
    slot,
    width: grid,
    height: grid,
    camera: { focalPx: r.stats.focalPx, cx: grid / 2, cy: grid / 2 },
    alpha,
    depth,
    // 胴と頭だけで合わせる案は、実素材では**不利**だった（0.985 対 0.965、
    // docs/12 §12.15.5）。腕も形の手がかりとして効いているので渡さない。
    // 姿勢が大きく違う3枚を入れられたときは考え直す。
    ...(r.headYawDeg !== null ? { headYawDeg: r.headYawDeg } : {}),
  };
}

/** 位置合わせに使わない領域を作る（いまは使っていない。§12.15.5 の記録）。 */
export { torsoAndHead };

export async function generateMulti(
  photos: readonly MultiPhoto[],
  options: GenerateMultiOptions,
): Promise<GenerateMultiResult> {
  if (photos.length < 2) throw new Error('複数枚モードには2枚以上が要ります');

  const grid = options.grid;
  const report = (f: number, label: string): void => options.onProgress?.(f, label);

  // --- ⓪①②③ を view ごとに回す
  //
  // 進捗は「枚数ぶんの区間」に割り振る。最後の 15% を位置合わせと合成に残す。
  const perView = 0.85 / photos.length;
  const results: { slot: ViewSlot; result: GenerateResult }[] = [];

  for (let i = 0; i < photos.length; i++) {
    const photo = photos[i] as MultiPhoto;
    const base = i * perView;
    const opts: GenerateOptions = {
      mode: options.mode,
      grid,
      reduction: options.reduction,
      backend: options.backend,
      ...(options.shaderF16 !== undefined ? { shaderF16: options.shaderF16 } : {}),
      ...(options.inpaint !== undefined ? { inpaint: options.inpaint } : {}),
      ...(options.depthTiles !== undefined ? { depthTiles: options.depthTiles } : {}),
      ...(options.fineGridRatio !== undefined ? { fineGridRatio: options.fineGridRatio } : {}),
      onProgress: (f, label) =>
        report(base + f * perView, `${SLOT_LABEL[photo.slot]}: ${label}`),
      // 1枚目（正面）だけ、できた時点で見せる。3枚待たせない。
      ...(i === 0 && options.onPreview ? { onPreview: options.onPreview } : {}),
    };
    const result = await generate(photo.blob, opts);
    results.push({ slot: photo.slot, result });
  }

  // --- Ⓑ 位置合わせ
  report(0.88, '3枚の向きを合わせています');
  const alignViews = results.map((r) => toAlignView(r.slot, r.result, grid));
  const registration = registerViews(alignViews);

  // --- Ⓒ 合成（重複除去はまだ無い）
  report(0.96, '3枚を1つにまとめています');
  const refIndex = registration.referenceIndex;
  const sources: MergeSource[] = results.map((r, i) => {
    const view = registration.views[i];
    if (!view) throw new Error(`位置合わせの結果が足りません: ${i}`);
    // **回転の中心は registerViews が返したものをそのまま使う。**
    // ここで重心を計算し直してはいけない。向こうは間引いた点で求めているので
    // 値が微妙にずれ、姿勢は正しいのに位置だけ食い違う。
    return { build: r.result.build, pose: view.pose, frame: view.frame };
  });
  const build = mergeBuilds(sources);

  report(1, '完成しました');
  return {
    build,
    views: results,
    registration,
    metricFix: (results[refIndex]?.result.metricFix ?? 1),
  };
}
