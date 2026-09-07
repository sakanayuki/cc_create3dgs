/**
 * `PhotoSplatDocument` と `.pgs` の相互変換（docs/05 §5.2）。
 *
 * コンテナ（pgs.ts）と画像符号化（images.ts）を束ねる層。
 * ここが「何を保存し、何を保存しないか」という設計判断（docs/04 §4.5.3）を
 * コードとして表している箇所になる。
 *
 * 保存する:   深度（上位/下位2プレーン）・前面色・α・背面色・補完テクスチャ
 * 保存しない: 位置・法線・スケール・厚み・スカート幾何・適応サンプリングの分割
 *             （いずれも読み込み時に決定的に再計算できる）
 */
import type { PhotoSplatDocument } from '../doc/PhotoSplatDocument';
import { validateDocument } from '../doc/PhotoSplatDocument';
import {
  type ImageCodec,
  grayToRgba,
  mergeDepth,
  rgbToRgba,
  rgbaToGray,
  rgbaToRgb,
  splitDepth,
} from './images';
import { missingRequiredChunks, packPgs, unpackPgs } from './pgs';

export interface EncodeDocumentOptions {
  /** 深度の量子化ビット数。標準・高品質は 12、軽量は 10（docs/04 §4.7）。 */
  readonly depthBits?: number;
  /** 前面色の WebP 品質 0..1。高品質プリセットは lossless を使う。 */
  readonly colorQuality?: number;
  /** 前面色を可逆で保存する（高品質プリセット）。 */
  readonly colorLossless?: boolean;
  /** 背面色の WebP 品質。 */
  readonly backColorQuality?: number;
  /** 補完テクスチャの WebP 品質。 */
  readonly inpaintQuality?: number;
}

const DEFAULTS = {
  depthBits: 12,
  colorQuality: 0.9,
  colorLossless: false,
  backColorQuality: 0.85,
  inpaintQuality: 0.88,
} as const;

/** マニフェストの形。読み込み側が逆量子化と再計算に使う。 */
export interface PgsManifest {
  version: number;
  grid: { width: number; height: number };
  camera: PhotoSplatDocument['camera'];
  depthRange: PhotoSplatDocument['depthRange'];
  depthBits: number;
  backShell: PhotoSplatDocument['backShell'];
  skirt: PhotoSplatDocument['skirt'];
  sampling: PhotoSplatDocument['sampling'];
  meta: PhotoSplatDocument['meta'];
}

export async function encodeDocument(
  doc: PhotoSplatDocument,
  codec: ImageCodec,
  options: EncodeDocumentOptions = {},
): Promise<Uint8Array> {
  const problems = validateDocument(doc);
  if (problems.length > 0) {
    throw new Error(`ドキュメントが不正です:\n- ${problems.join('\n- ')}`);
  }
  const opt = { ...DEFAULTS, ...options };
  const { width, height } = doc;
  const hw = Math.floor(width / 2);
  const hh = Math.floor(height / 2);

  // 深度は 12bit に量子化してから上位/下位に分ける。16bit のまま持つと
  // 下位がノイズになり PNG の予測が壊れる（docs/04 §4.5.4）。
  const { high, low } = splitDepth(doc.frontDepth, opt.depthBits);

  const chunks = new Map<string, Uint8Array>();
  chunks.set('DPTH', await codec.encode(grayToRgba(high, width, height), width, height, { format: 'png' }));
  chunks.set('DPTL', await codec.encode(grayToRgba(low, width, height), width, height, { format: 'png' }));
  chunks.set(
    'COLR',
    await codec.encode(rgbToRgba(doc.frontColor, width, height), width, height,
      opt.colorLossless ? { format: 'png' } : { format: 'webp', quality: opt.colorQuality }),
  );
  chunks.set('ALFA', await codec.encode(grayToRgba(doc.alpha, width, height), width, height, { format: 'png' }));
  chunks.set(
    'BCOL',
    await codec.encode(rgbToRgba(doc.backColor, hw, hh), hw, hh, {
      format: 'webp', quality: opt.backColorQuality,
    }),
  );
  if (doc.inpaint) {
    // α 付きなので RGBA のまま。バンド以外は透明で、そこはよく縮む。
    chunks.set(
      'INPT',
      await codec.encode(doc.inpaint, hw, hh, { format: 'webp', quality: opt.inpaintQuality }),
    );
  }

  const manifest: PgsManifest = {
    version: 1,
    grid: { width, height },
    camera: doc.camera,
    depthRange: doc.depthRange,
    depthBits: opt.depthBits,
    backShell: doc.backShell,
    skirt: doc.inpaint ? doc.skirt : { ...doc.skirt, colorSource: 'stretch' },
    sampling: doc.sampling,
    meta: doc.meta,
  };
  return packPgs(manifest as unknown as Record<string, unknown>, chunks);
}

export async function decodeDocument(
  bytes: Uint8Array,
  codec: ImageCodec,
): Promise<PhotoSplatDocument> {
  const file = unpackPgs(bytes);
  const missing = missingRequiredChunks(file);
  if (missing.length > 0) {
    throw new Error(`必須チャンクが欠けています: ${missing.join(', ')}`);
  }
  const m = file.manifest as unknown as PgsManifest;
  if (!m.grid || !m.camera || !m.depthRange) {
    throw new Error('マニフェストに必須項目がありません（grid / camera / depthRange）');
  }
  const { width, height } = m.grid;
  const hw = Math.floor(width / 2);
  const hh = Math.floor(height / 2);

  const need = async (id: string) => {
    const payload = file.chunks.get(id);
    if (!payload) throw new Error(`チャンクがありません: ${id}`);
    const img = await codec.decode(payload);
    if (img.width !== width && img.width !== hw) {
      throw new Error(`${id} の幅がマニフェストと合いません: ${img.width}`);
    }
    return img;
  };

  const highImg = await need('DPTH');
  const lowImg = await need('DPTL');
  const colorImg = await need('COLR');
  const alphaImg = await need('ALFA');
  const backImg = await need('BCOL');

  const frontDepth = mergeDepth(
    {
      high: rgbaToGray(highImg.rgba, width, height),
      low: rgbaToGray(lowImg.rgba, width, height),
    },
    m.depthBits ?? DEFAULTS.depthBits,
  );

  const inpaintPayload = file.chunks.get('INPT');
  const inpaint = inpaintPayload ? (await codec.decode(inpaintPayload)).rgba : null;

  const doc: PhotoSplatDocument = {
    width,
    height,
    camera: m.camera,
    depthRange: m.depthRange,
    frontDepth,
    frontColor: rgbaToRgb(colorImg.rgba, width, height),
    alpha: rgbaToGray(alphaImg.rgba, width, height),
    backColor: rgbaToRgb(backImg.rgba, hw, hh),
    inpaint,
    backShell: m.backShell,
    // 補完テクスチャが無ければスカートの色は引き伸ばしに落ちる
    skirt: inpaint ? m.skirt : { ...m.skirt, colorSource: 'stretch' },
    sampling: m.sampling,
    meta: m.meta,
  };

  const problems = validateDocument(doc);
  if (problems.length > 0) {
    throw new Error(`復元したドキュメントが不正です:\n- ${problems.join('\n- ')}`);
  }
  return doc;
}
