/**
 * 画像プレーンの符号化・復号（docs/05 §5.2.1）。
 *
 * `.pgs` のペイロードは PNG / WebP のバイト列そのものなので、符号化はブラウザの
 * ネイティブ実装（OffscreenCanvas）に任せ、自前のエンコーダを書かない。
 * その代わりこの層をインタフェースで切り、コンテナ層（pgs.ts）は純粋なまま保つ。
 */

export type ImageFormat = 'png' | 'webp';

export interface EncodeOptions {
  readonly format: ImageFormat;
  /** WebP の品質 0..1。png では無視される。 */
  readonly quality?: number;
}

export interface ImageCodec {
  /** RGBA バイト列（w×h×4）を符号化する。 */
  encode(rgba: Uint8ClampedArray, width: number, height: number, opts: EncodeOptions): Promise<Uint8Array>;
  /** 符号化されたバイト列を RGBA に戻す。 */
  decode(bytes: Uint8Array): Promise<{ rgba: Uint8ClampedArray; width: number; height: number }>;
}

/** OffscreenCanvas を使う実装。ブラウザと Worker の両方で動く。 */
export class CanvasImageCodec implements ImageCodec {
  async encode(
    rgba: Uint8ClampedArray,
    width: number,
    height: number,
    opts: EncodeOptions,
  ): Promise<Uint8Array> {
    if (rgba.length !== width * height * 4) {
      throw new Error(`RGBA の長さが合いません: ${rgba.length}（期待 ${width * height * 4}）`);
    }
    const canvas = new OffscreenCanvas(width, height);
    const ctx = canvas.getContext('2d', { willReadFrequently: false });
    if (!ctx) throw new Error('2D コンテキストを取得できませんでした');
    // ImageData は ArrayBuffer 実体を要求する。subarray 由来だと型が合わないので写す。
    const owned = new Uint8ClampedArray(rgba.length);
    owned.set(rgba);
    ctx.putImageData(new ImageData(owned, width, height), 0, 0);

    const blob = await canvas.convertToBlob({
      type: opts.format === 'webp' ? 'image/webp' : 'image/png',
      ...(opts.format === 'webp' && opts.quality !== undefined ? { quality: opts.quality } : {}),
    });
    return new Uint8Array(await blob.arrayBuffer());
  }

  async decode(bytes: Uint8Array): Promise<{ rgba: Uint8ClampedArray; width: number; height: number }> {
    // subarray のことがあるので、Blob には実体をコピーして渡す
    const blob = new Blob([bytes.slice()]);
    const bitmap = await createImageBitmap(blob);
    try {
      const canvas = new OffscreenCanvas(bitmap.width, bitmap.height);
      const ctx = canvas.getContext('2d', { willReadFrequently: true });
      if (!ctx) throw new Error('2D コンテキストを取得できませんでした');
      ctx.drawImage(bitmap, 0, 0);
      const data = ctx.getImageData(0, 0, bitmap.width, bitmap.height);
      return { rgba: data.data, width: bitmap.width, height: bitmap.height };
    } finally {
      bitmap.close();
    }
  }
}

// --- プレーンの詰め替え -----------------------------------------------------
// 画像は常に RGBA だが、保存したいのは 1ch（深度・α）だったり 3ch（色）だったりする。
// 使わないチャンネルを潰しておくとコーデックの予測が効いて小さくなる。

/** 1チャンネルのデータを RGBA に広げる（RGB に同じ値を入れ、A は 255）。 */
export function grayToRgba(gray: ArrayLike<number>, width: number, height: number): Uint8ClampedArray {
  const n = width * height;
  if (gray.length !== n) throw new Error(`長さが合いません: ${gray.length}（期待 ${n}）`);
  const out = new Uint8ClampedArray(n * 4);
  for (let i = 0; i < n; i++) {
    const v = gray[i] as number;
    out[i * 4] = v;
    out[i * 4 + 1] = v;
    out[i * 4 + 2] = v;
    out[i * 4 + 3] = 255;
  }
  return out;
}

export function rgbaToGray(rgba: ArrayLike<number>, width: number, height: number): Uint8ClampedArray {
  const n = width * height;
  const out = new Uint8ClampedArray(n);
  for (let i = 0; i < n; i++) out[i] = rgba[i * 4] as number;
  return out;
}

/** 3チャンネル（RGB）を RGBA に広げる。 */
export function rgbToRgba(rgb: ArrayLike<number>, width: number, height: number): Uint8ClampedArray {
  const n = width * height;
  if (rgb.length !== n * 3) throw new Error(`長さが合いません: ${rgb.length}（期待 ${n * 3}）`);
  const out = new Uint8ClampedArray(n * 4);
  for (let i = 0; i < n; i++) {
    out[i * 4] = rgb[i * 3] as number;
    out[i * 4 + 1] = rgb[i * 3 + 1] as number;
    out[i * 4 + 2] = rgb[i * 3 + 2] as number;
    out[i * 4 + 3] = 255;
  }
  return out;
}

export function rgbaToRgb(rgba: ArrayLike<number>, width: number, height: number): Uint8ClampedArray {
  const n = width * height;
  const out = new Uint8ClampedArray(n * 3);
  for (let i = 0; i < n; i++) {
    out[i * 3] = rgba[i * 4] as number;
    out[i * 3 + 1] = rgba[i * 4 + 1] as number;
    out[i * 3 + 2] = rgba[i * 4 + 2] as number;
  }
  return out;
}

// --- 深度の 2 プレーン分離（docs/04 §4.5.4） --------------------------------
//
// 16bit をそのまま持ってはいけない。深度の実効精度は 12bit 程度で、16bit にすると
// 下位 4bit が実質ノイズになり、PNG の予測フィルタ後の残差がランダム化して
// 圧縮率が壊滅する。精度が要らないビットを持つことは無駄ではなく有害である。
//
// さらに上位／下位を別プレーンに分ける。PNG のフィルタはバイト単位で動くので、
// 16bit を 1 枚で持つと滑らかな上位バイトと不規則な下位バイトが交互に並び、
// どちらの予測も当たらない。

export interface SplitDepth {
  /** 上位 8bit。非常に滑らかで高圧縮。 */
  readonly high: Uint8ClampedArray;
  /** 下位ビット。上位ニブルに詰めて 8bit プレーンとして持つ。 */
  readonly low: Uint8ClampedArray;
}

/**
 * 16bit 深度を `bits` bit に量子化し、上位8bit と残りに分ける。
 * @param bits 12 が既定（標準プリセット）。軽量は 10、高品質も 12。
 */
export function splitDepth(depth16: Uint16Array, bits = 12): SplitDepth {
  if (bits < 8 || bits > 16) throw new Error(`深度ビット数が範囲外です: ${bits}`);
  const n = depth16.length;
  const high = new Uint8ClampedArray(n);
  const low = new Uint8ClampedArray(n);
  const lowBits = bits - 8;
  const shift = 16 - bits;

  for (let i = 0; i < n; i++) {
    const q = (depth16[i] as number) >>> shift; // bits ビットに量子化
    high[i] = (q >>> lowBits) & 0xff;
    // 下位を上位ニブル側に寄せる。PNG の 8bit プレーンとして扱えるようにするため。
    low[i] = lowBits > 0 ? (q & ((1 << lowBits) - 1)) << (8 - lowBits) : 0;
  }
  return { high, low };
}

/** splitDepth の逆。16bit の値域に戻す。 */
export function mergeDepth(split: SplitDepth, bits = 12): Uint16Array {
  const n = split.high.length;
  const out = new Uint16Array(n);
  const lowBits = bits - 8;
  const shift = 16 - bits;
  const maxQ = (1 << bits) - 1;

  for (let i = 0; i < n; i++) {
    const lo = lowBits > 0 ? (split.low[i] as number) >>> (8 - lowBits) : 0;
    const q = (((split.high[i] as number) << lowBits) | lo) & maxQ;
    // 量子化幅の中央に戻す。切り捨てのまま返すと系統的に手前へずれる。
    out[i] = Math.min(65535, (q << shift) + (shift > 0 ? 1 << (shift - 1) : 0));
  }
  return out;
}
