/**
 * ⓪ 前処理（docs/03 §3.2）。
 *
 * 入力の写真を、以降の全段が前提にする「作業グリッド」に載せる。
 *   ・EXIF の回転を適用して正立させる
 *   ・長辺 1024px に縮める（軽量プリセットは 512px）
 *   ・アスペクト比を保ったままレターボックスで正方形にする
 *
 * 正方形にするのは、深度モデルもマットモデルも正方の入力を取るため。
 * 引き伸ばして正方にすると被写体の形が歪み、それが最後まで残る。
 */

/** 作業グリッドの一辺。決定 D17。 */
export const WORKING_GRID = 1024;
/** 軽量プリセットの作業グリッド。 */
export const LIGHT_GRID = 512;

export interface Letterbox {
  /** 元画像に掛ける倍率。 */
  readonly scale: number;
  /** 縮小後の大きさ。 */
  readonly width: number;
  readonly height: number;
  /** 正方グリッド内での左上の位置。 */
  readonly offsetX: number;
  readonly offsetY: number;
  /** グリッドの一辺。 */
  readonly size: number;
}

/**
 * 長辺を `size` に合わせ、中央に置くレターボックスを求める。
 *
 * 拡大はしない。小さい写真を引き伸ばしても情報は増えず、
 * ガウシアン数だけが無駄に増える。
 */
export function letterbox(srcWidth: number, srcHeight: number, size = WORKING_GRID): Letterbox {
  if (srcWidth <= 0 || srcHeight <= 0) throw new Error('画像の大きさが不正です');
  const scale = Math.min(1, size / Math.max(srcWidth, srcHeight));
  const width = Math.max(1, Math.round(srcWidth * scale));
  const height = Math.max(1, Math.round(srcHeight * scale));
  return {
    scale,
    width,
    height,
    offsetX: Math.floor((size - width) / 2),
    offsetY: Math.floor((size - height) / 2),
    size,
  };
}

/** グリッド座標を元画像の座標へ戻す。UI のタップ位置を写真に対応づけるのに使う。 */
export function gridToSource(box: Letterbox, gx: number, gy: number): [number, number] {
  return [(gx - box.offsetX) / box.scale, (gy - box.offsetY) / box.scale];
}

export interface PreparedImage {
  /** 作業グリッドに載せた RGBA8。長さ size×size×4。 */
  readonly rgba: Uint8ClampedArray;
  readonly box: Letterbox;
}

/**
 * 画像を作業グリッドに載せる。
 *
 * EXIF の回転は `createImageBitmap` の `imageOrientation: 'from-image'` に任せる。
 * 自前で EXIF を読んで回すより確実で、ブラウザ側で最適化されている。
 *
 * パディング領域は α=0 で残す。ここに被写体が無いことは後段（マット、深度）が
 * α で判別できる。黒で埋めると「黒い背景」として扱われてしまう。
 */
export async function prepareImage(source: Blob, size = WORKING_GRID): Promise<PreparedImage> {
  const bitmap = await createImageBitmap(source, { imageOrientation: 'from-image' });
  try {
    const box = letterbox(bitmap.width, bitmap.height, size);
    const canvas = new OffscreenCanvas(size, size);
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    if (!ctx) throw new Error('2D コンテキストを取得できませんでした');
    // 既定の transparent black のまま描く。clearRect も fillRect も要らない。
    ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(bitmap, box.offsetX, box.offsetY, box.width, box.height);
    return { rgba: ctx.getImageData(0, 0, size, size).data, box };
  } finally {
    bitmap.close();
  }
}

// --- 深度パスの入力 ---------------------------------------------------------

export interface Rect {
  readonly x: number;
  readonly y: number;
  readonly width: number;
  readonly height: number;
}

/**
 * α から被写体の外接矩形を求める。被写体が無ければ null。
 *
 * @param margin 外接矩形をこの割合だけ広げる。境界ちょうどで切ると、
 *               タイル推論のときに縁の文脈が失われて深度が乱れる。
 */
export function subjectBBox(
  alpha: ArrayLike<number>,
  width: number,
  height: number,
  threshold = 128,
  margin = 0.05,
): Rect | null {
  let x0 = width;
  let y0 = height;
  let x1 = -1;
  let y1 = -1;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      if ((alpha[y * width + x] as number) < threshold) continue;
      if (x < x0) x0 = x;
      if (y < y0) y0 = y;
      if (x > x1) x1 = x;
      if (y > y1) y1 = y;
    }
  }
  if (x1 < 0) return null;

  const mx = Math.round((x1 - x0 + 1) * margin);
  const my = Math.round((y1 - y0 + 1) * margin);
  const nx0 = Math.max(0, x0 - mx);
  const ny0 = Math.max(0, y0 - my);
  const nx1 = Math.min(width - 1, x1 + mx);
  const ny1 = Math.min(height - 1, y1 + my);
  return { x: nx0, y: ny0, width: nx1 - nx0 + 1, height: ny1 - ny0 + 1 };
}

/**
 * 被写体の外接矩形を 2×2 に分けたタイルを返す（docs/03 §3.4「2パス構成」）。
 *
 * 隣り合うタイルは `overlap` の割合だけ重ねる。重なり領域が無いと、
 * 全体パスとの整合（最小二乗フィット）もタイル同士の融合もできない。
 *
 * 返す矩形は必ず画像の内側に収まる。はみ出したまま推論に渡すと、
 * 端が黒で埋まってそこに偽の深度段差ができる。
 */
export function depthTiles(bbox: Rect, imageWidth: number, imageHeight: number, overlap = 0.2): Rect[] {
  const tw = Math.min(imageWidth, Math.ceil((bbox.width * (1 + overlap)) / 2));
  const th = Math.min(imageHeight, Math.ceil((bbox.height * (1 + overlap)) / 2));
  const tiles: Rect[] = [];
  for (const [fx, fy] of [
    [0, 0],
    [1, 0],
    [0, 1],
    [1, 1],
  ] as const) {
    // 右下タイルは外接矩形の右下端に揃える。左上から等間隔に置くと
    // 端数のぶんだけ被写体の右下がどのタイルにも入らないことがある。
    const x = fx === 0 ? bbox.x : bbox.x + bbox.width - tw;
    const y = fy === 0 ? bbox.y : bbox.y + bbox.height - th;
    tiles.push({
      x: Math.max(0, Math.min(imageWidth - tw, x)),
      y: Math.max(0, Math.min(imageHeight - th, y)),
      width: tw,
      height: th,
    });
  }
  return tiles;
}
