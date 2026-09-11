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
 *   Ⓐ Ⓑ  … `registerViews()`。実素材で M1 = 0.969（§12.16.5）
 *   Ⓒ    … `mergeBuilds()`。重なった面を落とす。実素材で二重率 24.7% → 2.5%
 *   Ⓓ    … **まだ無い**（色合わせ）
 *
 * 色合わせが無いので、view ごとの露出差は継ぎ目の色差として残る。
 *
 * ## 時間
 *
 * ②（深度推定）が支配的で、それを枚数ぶん繰り返す。3枚なら単純に 3 倍近い。
 * docs/12 §12.12 の見積もりでは WebGPU で 12.6 秒、位置合わせがさらに 2.6 秒。
 * **10 秒の SLO には収まらない。** 決定 D30 で複数枚モードは別 SLO にしてある。
 */
import { generate, type GenerateOptions, type GenerateResult, type SubjectMode } from './generate';
import { mergeBuilds, type MergeSource, type MergeStats } from './align/mergeViews';
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
 * 合成を許す M1（収まり率）の下限（docs/12 §12.13、§12.16.5）。
 *
 * これを割った view は**合成に入れない**。同じ人物・同じ場所でない写真や、
 * カメラが動いた写真を入れられると位置合わせは解けず、そのまま重ねると
 * 見るに堪えない立体が出る。しかも利用者には理由が分からない。
 *
 * ## 実測から決め直した（0.95 → 0.90 / 0.85）
 *
 * 0.95 は「正しく合ったときの値」だけを見て置いた。**負けている側を測って
 * いなかった。** `scripts/gate_probe.ts` で両側を測る（実素材3枚、既定の設定）。
 *
 * | 場合 | 最悪 M1 | 最悪はみ出し |
 * |---|---|---|
 * | **正解（3枚）** | **0.969** | 0.32 px |
 * | yaw を 10° 取り違え | 0.850 | 0.86 px |
 * | yaw を 20° 取り違え | 0.680 | 1.83 px |
 * | 合わせを一切しない | 0.818 | 0.98 px |
 * | 尺度が 15% 外れる | 0.874 | 6.98 px |
 * | 3枚のうち1枚が別人（合成データ） | 0.788 | 1.01 px |
 *
 * 正解が 0.969 なので、0.95 では**余裕が 0.02 しかない**。M1 の上限は
 * マットの粗さと体の細さで決まる（128px のグリッドでは輪郭の1画素帯だけで
 * 面積の 6〜7% になる）ので、被写体が変われば簡単に割る。実際、利用者の
 * 写真3枚が 0.95 を割って合成できなかった。0.90 なら正解から 0.069 下、
 * いちばん近い間違い（0.874）から 0.026 上で、上の負け筋はすべて落ちる。
 *
 * ## 2枚のときは 0.85（**別の線を引く必要がある**）
 *
 * 角度を縛るのは3枚目である。2枚だと拘束が足りず、**正しい組でも M1 が下がる**。
 *
 * | 場合 | 最悪 M1 |
 * |---|---|
 * | **正解（実素材 正面+右）** | **0.928** |
 * | **正解（実素材 正面+左）** | **0.920** |
 * | 正面+右 で 20° 取り違え | 0.884 |
 * | 正面+左 で 20° 取り違え | 0.667 |
 * | 正面+右 で 45° 取り違え | 0.750 |
 *
 * 0.90 のままだと正解（0.920）との余裕が 0.02 しかない。3枚のときと同じだけの
 * 余裕（約 0.07）を取って 0.85 にする。
 *
 * **2枚では弁別できないことは認める。** 合成データで別人の2枚を組ませると
 * 0.929〜0.934 になり、実素材の**正しい**2枚（0.920〜0.928）と完全に重なる。
 * どこに線を引いても分けられない。2枚のときのゲートは「大きく壊れた場合だけ
 * 止める」ものであって、写真の取り違えを見つける力は無い。3枚目があって
 * 初めてその力が出る（docs/12 §12.16.5）。
 */
function minInsideRatio(viewCount: number): number {
  return viewCount >= 3 ? 0.9 : 0.85;
}

/**
 * 例外を利用者に読める一文にする。
 *
 * 復号の失敗だけは特別扱いする。ここへ来る時点で `ensureDecodable` は通って
 * いて、**その写真は一度開けている**。同じ写真が後から開けなくなったのだから、
 * 形式ではなく資源のほうを疑う。ここで「写真が壊れています」と言うのは
 * 明確に誤りである。
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
  /** そのとき使った下限。理由の文面に出す。 */
  readonly threshold: number;
} {
  const worst = registration.views.reduce((a, b) => (a.insideRatio <= b.insideRatio ? a : b));
  const threshold = minInsideRatio(registration.views.length);
  return { ok: registration.worstInsideRatio >= threshold, worst, threshold };
}

/** 合成できなかったときに利用者へ出す説明（docs/12 R17）。 */
export function fallbackMessage(
  worst: RegisterResult['views'][number],
  refSlot: ViewSlot,
  threshold: number,
): string {
  return (
    `写真どうしの向きが合いませんでした（${SLOT_LABEL[worst.slot]}の一致度 ` +
    `${(worst.insideRatio * 100).toFixed(0)}%、必要 ${(threshold * 100).toFixed(0)}%）。` +
    `同じ人物・同じ姿勢・同じ場所で、その場で向きだけ変えて撮った写真かを確かめてください。` +
    `今回は${SLOT_LABEL[refSlot]}の1枚だけで作りました。`
  );
}

/** 合わなかった view を外して作り直したときの説明（docs/12 §12.16.5）。 */
export function droppedMessage(dropped: readonly ViewSlot[], kept: readonly ViewSlot[]): string {
  return (
    `${dropped.map((s) => SLOT_LABEL[s]).join('と')}が他の写真と合わなかったので外しました。` +
    `${kept.map((s) => SLOT_LABEL[s]).join('と')}の${kept.length}枚で作っています。` +
    `外した写真だけ撮り直すと、${kept.length + dropped.length}枚ぶんになります。`
  );
}

/**
 * 合わない view を1枚ずつ外しながら、合う組み合わせを探す（docs/12 §12.16.5）。
 *
 * **1枚が合わないだけで全部を捨ててはいけない。** 元の作りは、3枚のうち
 * 1枚でも閾値を割ると基準の1枚だけに戻していた。利用者から見れば、
 * 3枚撮って3枚とも処理させたのに手元に残るのは1枚ぶん、という結果になる。
 * 失うものが大きすぎる。合う2枚があるなら、その2枚で作るほうがよい。
 *
 * 外すのは**いちばん低い非基準の view** である。基準は外せない（座標の
 * 原点なので）。M1 は「他の view から運んだ点が自分のマスクに収まった割合」
 * なので、姿勢がずれている view は自分の値がいちばん低く出る。実測でも
 * 10° ずらした view が最下位になった（`gate_probe.ts`）。
 *
 * ただし**ずれた view が2枚あって互いに整合している**ときは、基準のほうが
 * 最下位に出る（尺度を揃えて2枚ずらすと基準が 0.874 で最下位になった）。
 * そのときは外す先が無いので、諦めて基準1枚に戻る。これは正しい降伏である。
 *
 * 再合わせは `registerViews` をやり直す。残った組で解き直さないと、
 * 外した view に引っ張られた姿勢がそのまま残る。
 */
export function solveWithDrops(
  views: readonly AlignView[],
  onStep?: (remaining: number) => void,
): {
  /** 残した view の、`views` における添字。合成はこの順で行う。 */
  readonly keep: readonly number[];
  readonly registration: RegisterResult;
  /** 外した view の、`views` における添字。 */
  readonly dropped: readonly number[];
} {
  let keep = views.map((_, i) => i);
  let registration = registerViews(keep.map((i) => views[i] as AlignView));
  const dropped: number[] = [];

  while (!registrationGate(registration).ok && keep.length > 2) {
    const refIndex = registration.referenceIndex;
    let worst = -1;
    let worstRatio = Infinity;
    registration.views.forEach((v, i) => {
      if (i === refIndex) return;
      if (v.insideRatio < worstRatio) {
        worstRatio = v.insideRatio;
        worst = i;
      }
    });
    // 非基準がいなければ外す先が無い。降伏して抜ける。
    if (worst < 0) break;
    dropped.push(keep[worst] as number);
    keep = keep.filter((_, i) => i !== worst);
    onStep?.(keep.length);
    registration = registerViews(keep.map((i) => views[i] as AlignView));
  }
  return { keep, registration, dropped };
}

export interface GenerateMultiResult {
  readonly build: SplatBuild;
  /**
   * 合成できたか。false なら `build` は基準 view 1枚ぶんで、
   * `fallbackReason` に理由が入る（docs/12 R17）。
   */
  readonly merged: boolean;
  readonly fallbackReason: string | null;
  /** view ごとの生成結果。**入れた枚数ぶん全部**。統計の表示に使う。 */
  readonly views: readonly { readonly slot: ViewSlot; readonly result: GenerateResult }[];
  /**
   * 実際に合成に使った位置合わせ。`views` より**少ないことがある**
   * （合わなかった写真を外したとき。docs/12 §12.16.5）。
   */
  readonly registration: RegisterResult;
  /** 合わなかったので外した枠。空なら全部使った。 */
  readonly droppedSlots: readonly ViewSlot[];
  /** 合成の内訳（重複をいくつ落としたか）。合成しなかったときは null。 */
  readonly mergeStats: MergeStats | null;
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
 * **ここの値打ちは「どの写真か」を早く言えること**で、そこは確実に効く。
 *
 * **原因までは決めつけない。** ここで落ちたからといって形式のせいとは限らない。
 * 同じタブで前に生成していれば、モデルのセッションや GPU の資源が残ったままで、
 * 1枚目を開く時点でもう足りないことがありうる。`createImageBitmap` は
 * 資源不足でも同じ例外を投げるので、**投げられた場所から原因は読めない**
 * （PR #7 の Codex の指摘。docs/12 §12.16.4）。だから両方を挙げる。
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
          `このブラウザが扱えない形式（HEIC など）か、端末の資源（メモリ）が` +
          `足りないかのどちらかです。JPEG か PNG で保存し直すか、` +
          `画素数の小さい写真で試してください。` +
          `続けて何度も生成しているなら、ページを開き直すと直ることがあります。`,
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
  //
  // **合わない view は外して解き直す。** 1枚が合わないだけで全部を捨てると、
  // 3枚撮って処理させたのに手元に残るのは1枚ぶん、になる（docs/12 §12.16.5）。
  report(0.88, `${photos.length}枚の向きを合わせています`);
  const alignViews = results.map((r) => toAlignView(r.slot, r.result, grid));
  const solved = solveWithDrops(alignViews, (remaining) =>
    report(0.9, `合わない写真を外して、${remaining}枚で合わせ直しています`),
  );
  const registration = solved.registration;
  const active = solved.keep.map((i) => results[i] as { slot: ViewSlot; result: GenerateResult });
  const dropped = solved.dropped.map((i) => (results[i] as { slot: ViewSlot }).slot);

  // --- 合成する前に、位置合わせが立ったかを見る（docs/12 R17 / R21）
  //
  // **ここを通さずに合成してはいけない。** 解けていない姿勢で重ねると、
  // 壊れた立体がそのまま画面と書き出しに出る。利用者には何が起きたか
  // 分からない。設計にはゲートにすると書いてあったのに、実装で抜けていた。
  const refIndex = registration.referenceIndex;
  const gate = registrationGate(registration);
  if (!gate.ok) {
    // ここまで来たのは、2枚まで減らしても合わなかった（あるいは元から2枚
    // だった）とき。外す先はもう無い。基準の1枚で出す。
    const refResult = active[refIndex]?.result;
    if (!refResult) throw new Error('基準 view の結果がありません');
    report(1, '1枚だけで作りました');
    return {
      build: refResult.build,
      merged: false,
      fallbackReason: fallbackMessage(gate.worst, active[refIndex]?.slot ?? 'front', gate.threshold),
      views: results,
      registration,
      droppedSlots: dropped,
      mergeStats: null,
      metricFix: refResult.metricFix,
    };
  }

  // --- Ⓒ 合成（重なった面を落とす。docs/12 §12.8）
  report(0.96, `${active.length}枚を1つにまとめています`);
  const sources: MergeSource[] = active.map((r, i) => {
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
    // 外した写真があるなら、合成はできていても黙ってはいけない。
    fallbackReason:
      dropped.length > 0 ? droppedMessage(dropped, active.map((a) => a.slot)) : null,
    views: results,
    registration,
    droppedSlots: dropped,
    mergeStats: build.mergeStats,
    metricFix: active[refIndex]?.result.metricFix ?? 1,
  };
}
