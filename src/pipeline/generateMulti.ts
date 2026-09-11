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
  /** 元のファイル名。失敗したときにどの写真かを言うためだけに使う。 */
  readonly name?: string;
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

/**
 * 合成を許す M1（収まり率）の下限（docs/12 §12.13）。
 *
 * これを割ったら、**合成しない**。同じ人物・同じ場所でない3枚や、
 * カメラが動いた3枚を入れられると位置合わせは解けず、そのまま重ねると
 * 見るに堪えない立体が出る。しかも利用者には理由が分からない。
 * docs/12 §12.14 R17 / R21 は「閾値を割ったら諦めて正面1枚を出し、
 * 理由を日本語で伝える」と決めている。
 */
const MIN_INSIDE_RATIO = 0.95;

/**
 * 例外を利用者に読める一文にする。
 *
 * 復号の失敗だけは特別扱いする。`ensureDecodable` を通っているので
 * ファイル自体は開ける。それでも復号できないなら、**端末の資源が尽きた**ほうを
 * 疑うべきで、「写真が壊れています」と言うと利用者を誤った方向へ送る。
 */
export function explainFailure(e: unknown): string {
  const name = e instanceof Error ? e.name : '';
  if (name === 'InvalidStateError' || /could not be decoded/i.test(String(e))) {
    return (
      '写真を復号できませんでした。最初に開けることは確認済みなので、' +
      '端末の資源（メモリ）が足りなくなった可能性が高いです。' +
      '枚数を2枚に減らすか、画素数の小さい写真で試してください。'
    );
  }
  return String(e);
}

const SLOT_LABEL: Record<ViewSlot, string> = {
  front: '正面',
  right: '右向き',
  left: '左向き',
};

/**
 * 位置合わせの結果が、合成してよいものかを判定する（docs/12 R17 / R21）。
 *
 * 合成の手前に必ず通す。ここを通さずに重ねると、解けていない姿勢のまま
 * 壊れた立体が画面と書き出しに出て、利用者には理由が分からない。
 *
 * 純粋な関数にしてあるのは、**試験できるようにするため**である。
 * `generateMulti` ごと動かすにはモデルの推論が要り、単体試験では回せない。
 */
export function registrationGate(registration: RegisterResult): {
  readonly ok: boolean;
  /** 通らなかったとき、いちばん合っていなかった view。 */
  readonly worst: RegisterResult['views'][number];
} {
  const worst = registration.views.reduce((a, b) => (a.insideRatio <= b.insideRatio ? a : b));
  return { ok: registration.worstInsideRatio >= MIN_INSIDE_RATIO, worst };
}

/** 合成できなかったときに利用者へ出す説明（docs/12 R17）。 */
export function fallbackMessage(worst: RegisterResult['views'][number], refSlot: ViewSlot): string {
  return (
    `写真どうしの向きが合いませんでした（${SLOT_LABEL[worst.slot]}の一致度 ` +
    `${(worst.insideRatio * 100).toFixed(0)}%、必要 ${MIN_INSIDE_RATIO * 100}%）。` +
    `同じ人物・同じ姿勢・同じ場所で、その場で向きだけ変えて撮った写真かを確かめてください。` +
    `今回は${SLOT_LABEL[refSlot]}の1枚だけで作りました。`
  );
}

export interface GenerateMultiResult {
  readonly build: SplatBuild;
  /**
   * 合成できたか。false なら `build` は基準 view 1枚ぶんで、
   * `fallbackReason` に理由が入る（docs/12 R17）。
   */
  readonly merged: boolean;
  readonly fallbackReason: string | null;
  /** view ごとの生成結果。統計の表示に使う。 */
  readonly views: readonly { readonly slot: ViewSlot; readonly result: GenerateResult }[];
  readonly registration: RegisterResult;
  readonly metricFix: number;
}

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

/** 失敗したときに「どの写真か」を言うための呼び名。 */
function photoLabel(photo: MultiPhoto): string {
  return photo.name ? `${SLOT_LABEL[photo.slot]}（${photo.name}）` : SLOT_LABEL[photo.slot];
}

/**
 * 生成を始める前に、**全部の写真が本当に開けるか**を確かめる（docs/12 §12.16.4）。
 *
 * 開けない写真が2枚目にあると、1枚目に数十秒かけたあとで
 * `InvalidStateError: The source image could not be decoded.` だけが出る。
 * 利用者にはどの写真が悪いのか分からないし、かけた時間も無駄になる。
 *
 * ここで通しておくと、**あとで同じ例外が出たときの意味が変わる**。
 * ファイルは開けると分かっているので、そのときは端末の資源が尽きたほうを疑う。
 * その切り分けのためにも、先に一度開けておく価値がある。
 *
 * 開いた `ImageBitmap` はすぐ閉じる。抱えたままにすると、ここでの確認が
 * そのまま資源の枯渇を招く。
 */
export async function ensureDecodable(photos: readonly MultiPhoto[]): Promise<void> {
  if (typeof createImageBitmap !== 'function') return;
  for (const photo of photos) {
    let bitmap: ImageBitmap | null = null;
    try {
      bitmap = await createImageBitmap(photo.blob);
    } catch (cause) {
      throw new Error(
        `${photoLabel(photo)}の写真を開けませんでした。` +
          `このブラウザが扱えない形式（HEIC など）かもしれません。` +
          `JPEG か PNG で保存し直して入れ直してください。`,
        { cause },
      );
    } finally {
      bitmap?.close();
    }
  }
}

export async function generateMulti(
  photos: readonly MultiPhoto[],
  options: GenerateMultiOptions,
): Promise<GenerateMultiResult> {
  if (photos.length < 2) throw new Error('複数枚モードには2枚以上が要ります');

  // 1枚目に時間をかける前に、全部開けることを確かめる。
  await ensureDecodable(photos);

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
      // 「何枚目か」を必ず出す。これが無いと、1枚目の下書きが出た時点で
      // 終わったように見える（実機でそう報告された）。
      onProgress: (f, label) =>
        report(
          base + f * perView,
          `[${i + 1}/${photos.length}] ${SLOT_LABEL[photo.slot]}: ${label}`,
        ),
      // 1枚目（正面）だけ、できた時点で見せる。3枚待たせない。
      ...(i === 0 && options.onPreview ? { onPreview: options.onPreview } : {}),
    };
    const t0 = performance.now();
    // どの view で落ちたかを必ず言う。生の例外だけだと、3枚のうちどれの
    // 話なのかが利用者にも開発者にも分からない（docs/12 §12.16.4）。
    let result: GenerateResult;
    try {
      result = await generate(photo.blob, opts);
    } catch (cause) {
      throw new Error(`${i + 1}枚目・${photoLabel(photo)}で失敗しました: ${explainFailure(cause)}`, {
        cause,
      });
    }
    // どこまで進んだかを残す。画面が固まって見えたときに、どの view で
    // 止まったのかが分からないと追えない。
    console.info(
      `[photosplat] 複数枚 ${i + 1}/${photos.length} (${photo.slot}) 完了 ` +
        `${Math.round(performance.now() - t0)} ms / ${result.build.count} splats`,
    );
    results.push({ slot: photo.slot, result });
  }

  // --- Ⓑ 位置合わせ
  report(0.88, `${photos.length}枚の向きを合わせています`);
  const alignViews = results.map((r) => toAlignView(r.slot, r.result, grid));
  const registration = registerViews(alignViews);

  // --- 合成する前に、位置合わせが立ったかを見る（docs/12 R17 / R21）
  //
  // **ここを通さずに合成してはいけない。** 解けていない姿勢で重ねると、
  // 壊れた立体がそのまま画面と書き出しに出る。利用者には何が起きたか
  // 分からない。設計にはゲートにすると書いてあったのに、実装で抜けていた。
  const refIndex = registration.referenceIndex;
  const gate = registrationGate(registration);
  if (!gate.ok) {
    const refResult = results[refIndex]?.result;
    if (!refResult) throw new Error('基準 view の結果がありません');
    report(1, '1枚だけで作りました');
    return {
      build: refResult.build,
      merged: false,
      fallbackReason: fallbackMessage(gate.worst, results[refIndex]?.slot ?? 'front'),
      views: results,
      registration,
      metricFix: refResult.metricFix,
    };
  }

  // --- Ⓒ 合成（重複除去はまだ無い）
  report(0.96, `${photos.length}枚を1つにまとめています`);
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
    merged: true,
    fallbackReason: null,
    views: results,
    registration,
    metricFix: results[refIndex]?.result.metricFix ?? 1,
  };
}
