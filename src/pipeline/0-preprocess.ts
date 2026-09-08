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
 * レターボックスの詰め物に、内容の端の画素を伸ばして入れる（α は 0 のまま）。
 *
 * 詰め物を黒のままモデルに渡すと、**内容の縁に強い段差ができる**。実写で
 * 測ると、MODNet はその段差に反応して画像の左端に沿った縦帯を前景と
 * 誤判定し、プールサイドを 88 画素ぶん被写体に含めていた（y=900 の行で、
 * 腕は x=314 からなのに x=226 から被写体とされていた）。詰め物を端の
 * 画素で埋めると x=312 まで下がり、正しくなる。被写体の内部にできていた
 * 穴（同じ行で 2 か所）も消える。
 *
 * α は 0 のまま残す。後段が「ここに被写体は無い」と判別できる必要がある
 * ので、そちらの意味は変えない。変えるのは RGB だけで、これはモデルが
 * 見る値である。
 */
export function replicateEdgesIntoPadding(
  rgba: Uint8ClampedArray,
  size: number,
  box: Letterbox,
): void {
  const x0 = box.offsetX;
  const y0 = box.offsetY;
  const x1 = box.offsetX + box.width - 1;
  const y1 = box.offsetY + box.height - 1;
  if (x0 <= 0 && y0 <= 0 && x1 >= size - 1 && y1 >= size - 1) return;

  for (let y = 0; y < size; y++) {
    const sy = Math.min(y1, Math.max(y0, y));
    for (let x = 0; x < size; x++) {
      if (x >= x0 && x <= x1 && y >= y0 && y <= y1) continue;
      const sx = Math.min(x1, Math.max(x0, x));
      const di = (y * size + x) * 4;
      const si = (sy * size + sx) * 4;
      rgba[di] = rgba[si] as number;
      rgba[di + 1] = rgba[si + 1] as number;
      rgba[di + 2] = rgba[si + 2] as number;
      // α は 0 のまま
    }
  }
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
    const rgba = ctx.getImageData(0, 0, size, size).data;
    replicateEdgesIntoPadding(rgba, size, box);
    return { rgba, box };
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
 * 被写体の外接矩形を覆う**正方形**タイルを、長辺に沿って並べて返す
 * （docs/03 §3.4「2パス構成」）。
 *
 * 隣り合うタイルは `overlap` の割合だけ重ねる。重なり領域が無いと、
 * 全体パスとの整合（最小二乗フィット）もタイル同士の融合もできない。
 *
 * 返す矩形は必ず画像の内側に収まる。はみ出したまま推論に渡すと、
 * 端が黒で埋まってそこに偽の深度段差ができる。
 *
 * **正方形であること・長辺に沿って並べることが要**（v2.3、実写で判明）。
 * 以前は外接矩形を一律 2×2 に割っていたが、立っている人物では
 *
 *   - タイルが縦長（実測 330×550）になり、518² へ引き伸ばすと**顔が横に
 *     1.67 倍伸びる**。歪んだ顔から返る深度は使えない。
 *   - 縦の継ぎ目が外接矩形の左右中央、つまり**顔のまん中**に来る。実測では
 *     重なり帯が顔幅の 62% を横断し、顔の 21% がタイルの外へ出ていた。
 *     左右のタイルが別々に尺度を合わせるので、顔の中央に段差が入る。
 *
 * 立った人物の外接矩形は縦長で、横はもともと 518 に近い。横に割っても
 * 解像度は得られず、顔を切る害だけが残る。長辺（＝縦）にだけ並べる。
 */
export function depthTiles(bbox: Rect, imageWidth: number, imageHeight: number, overlap = 0.2): Rect[] {
  const longSide = Math.max(bbox.width, bbox.height);
  const shortSide = Math.min(bbox.width, bbox.height);

  // 短辺は 1 枚で覆いきる。割ると被写体を縦に切ってしまう。
  // 長辺は 2 枚に分ける目安で、短辺のほうが大きければそちらに合わせる。
  const side = Math.max(
    1,
    Math.min(imageWidth, imageHeight, Math.max(shortSide, Math.ceil(longSide / 2))),
  );

  // 1 枚では覆えない残り。これが僅かなら割らない。ほとんど同じタイルを
  // 2 枚推論しても解像度は上がらず、融合の継ぎ目を増やすだけ損になる。
  const remainder = longSide - side;
  const step = Math.max(1, Math.round(side * (1 - overlap)));
  const count = remainder > side * 0.1 ? Math.ceil(remainder / step) + 1 : 1;
  const alongX = bbox.width >= bbox.height;

  const tiles: Rect[] = [];
  const seen = new Set<string>();
  for (let i = 0; i < count; i++) {
    // 端をぴったり合わせて等間隔に置く。端数で被写体の端が欠けるのを防ぐ。
    const offset = count === 1 ? 0 : Math.round((i * (longSide - side)) / (count - 1));
    const rawX = alongX ? bbox.x + offset : bbox.x + Math.round((bbox.width - side) / 2);
    const rawY = alongX ? bbox.y + Math.round((bbox.height - side) / 2) : bbox.y + offset;
    const x = Math.max(0, Math.min(imageWidth - side, rawX));
    const y = Math.max(0, Math.min(imageHeight - side, rawY));
    const key = `${x},${y}`;
    if (seen.has(key)) continue;
    seen.add(key);
    tiles.push({ x, y, width: side, height: side });
  }
  return tiles;
}

/**
 * 矩形を、画像の内側に収まる正方形へ広げる。
 *
 * 深度モデルは正方の入力を取る。正方でない領域を 518² へ引き伸ばすと
 * 縦横で倍率が変わり、被写体（とくに顔）が歪む。黒でパディングする手も
 * あるが、縁に偽の深度段差ができるので、実際の画素で埋められるほうを採る。
 *
 * 画像そのものが正方形でなく、短辺より大きな正方形が取れない場合は、
 * 取れるだけの正方形（＝短辺）を返す。
 */
export function expandToSquare(rect: Rect, imageWidth: number, imageHeight: number): Rect {
  const side = Math.min(
    Math.max(rect.width, rect.height),
    imageWidth,
    imageHeight,
  );
  const cx = rect.x + rect.width / 2;
  const cy = rect.y + rect.height / 2;
  const x = Math.round(Math.max(0, Math.min(imageWidth - side, cx - side / 2)));
  const y = Math.round(Math.max(0, Math.min(imageHeight - side, cy - side / 2)));
  return { x, y, width: side, height: side };
}
