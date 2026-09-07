/**
 * ⑧ 遮蔽部のインペイント（docs/03 §3.7、決定 D16）。
 *
 * 視点を横に振ると、深度の不連続の**奥側**に、入力画像では手前の物体に
 * 隠されていた領域が露出する。v1 はそこへ奥側の色を引き伸ばしていたが、
 * それは「テクスチャが縞になって流れる」という 2.5D の典型的な破綻を生む。
 *
 * このモジュールが受け持つのは、モデルに渡す**マスクの作り方**と、
 * 返ってきた絵の**取り込み方**。推論そのものは呼び出し側が行う。
 * マスクの作り方こそがここの品質を決めるので、単体で検算できるようにしておく。
 */

const SUBJECT_ALPHA = 128;

export interface MaskParams {
  /** 露出しうる幅の下限（画素）。狭すぎるとモデルが文脈を掴めない。 */
  readonly minBand: number;
  /** 上限（画素）。広すぎると被写体の見えている部分まで塗り潰す。 */
  readonly maxBand: number;
  /** 段差とみなす深度ギャップの下限（正規化深度）。 */
  readonly threshold: number;
  /** 視点をどこまで振る前提で幅を見積もるか（ラジアン）。docs/06 §6.7 の上限。 */
  readonly maxYaw: number;
}

export const DEFAULT_MASK_PARAMS: MaskParams = {
  minBand: 4,
  maxBand: 32,
  threshold: 0.02,
  maxYaw: (45 * Math.PI) / 180,
};

/**
 * 露出する帯の幅（画素）を求める。
 *
 * 視点を横に b だけ動かすと、深度 z の点は画面上で f·b/z ずれる。
 * 手前と奥ではずれ方が違うので、その差だけ奥側が露出する。
 *
 *   w = f · b · (1/z_near − 1/z_far) = f · b · Δz / (z_near · z_far)
 *
 * b は被写体までの距離に対する横移動なので、振り角 θ に対して
 * b ≈ z_near · tan(θ) とする。
 */
export function exposedBandPx(
  focalPx: number,
  zNear: number,
  zFar: number,
  yaw: number,
): number {
  if (!(zFar > zNear) || !(zNear > 0)) return 0;
  const b = zNear * Math.tan(yaw);
  return (focalPx * b * (zFar - zNear)) / (zNear * zFar);
}

export interface InpaintMask {
  /** 0 = そのまま、255 = 補完してほしい。長さ width×height。 */
  readonly mask: Uint8ClampedArray;
  /** マスクした画素数。0 ならインペイントを回す必要がない。 */
  readonly maskedPixels: number;
}

/**
 * 深度の段差の奥側に、露出しうる帯をマスクする。
 *
 * マスクするのは**奥側**であることが肝心。手前側は入力画像に写っている
 * のだから塗り替えてはいけない。奥側は「隠れていて写っていない」領域が
 * これから見えるようになる場所なので、そこを描いてもらう。
 */
export function buildInpaintMask(
  depth: ArrayLike<number>,
  alpha: ArrayLike<number>,
  width: number,
  height: number,
  focalPx: number,
  nearZ: number,
  farZ: number,
  params: MaskParams = DEFAULT_MASK_PARAMS,
): InpaintMask {
  const mask = new Uint8ClampedArray(width * height);
  const span = farZ - nearZ;
  let masked = 0;

  const metric = (d: number): number => nearZ + d * span;

  for (let y = 1; y < height - 1; y++) {
    for (let x = 1; x < width - 1; x++) {
      const i = y * width + x;
      if ((alpha[i] as number) < SUBJECT_ALPHA) continue;
      const dHere = depth[i] as number;

      // 4近傍で最も大きい「奥向きの」段差と、その向き
      let gap = 0;
      let gx = 0;
      let gy = 0;
      const consider = (j: number, ox: number, oy: number): void => {
        if ((alpha[j] as number) < SUBJECT_ALPHA) return;
        const d = (depth[j] as number) - dHere;
        if (d > gap) {
          gap = d;
          gx = ox;
          gy = oy;
        }
      };
      consider(i - 1, -1, 0);
      consider(i + 1, 1, 0);
      consider(i - width, 0, -1);
      consider(i + width, 0, 1);
      if (gap < params.threshold) continue;

      const w = Math.round(
        Math.min(
          params.maxBand,
          Math.max(
            params.minBand,
            exposedBandPx(focalPx, metric(dHere), metric(dHere + gap), params.maxYaw),
          ),
        ),
      );

      // 段差の奥側へ w 画素ぶん。手前側（自分の側）は触らない。
      for (let s = 1; s <= w; s++) {
        const nx = x + gx * s;
        const ny = y + gy * s;
        if (nx < 0 || nx >= width || ny < 0 || ny >= height) break;
        const j = ny * width + nx;
        if (mask[j] === 255) continue;
        mask[j] = 255;
        masked++;
      }
    }
  }
  return { mask, maskedPixels: masked };
}

/**
 * インペイント結果を取り込む。
 *
 * マスクの内側だけを採り、境界を `feather` 画素かけて元の絵へ繋ぐ。
 * 境界をそのまま切ると、補完した領域の輪郭が線として見えてしまう。
 */
export function compositeInpaint(
  base: ArrayLike<number>,
  painted: ArrayLike<number>,
  mask: ArrayLike<number>,
  width: number,
  height: number,
  feather = 2,
): Uint8ClampedArray {
  // マスクを羽根の幅ぶんぼかして、混ぜる重みにする。
  const weight = new Float32Array(width * height);
  for (let i = 0; i < weight.length; i++) weight[i] = (mask[i] as number) / 255;

  if (feather > 0) {
    const r = Math.max(1, Math.round(feather));
    const tmp = new Float32Array(width * height);
    const blur = (src: Float32Array, dst: Float32Array, horizontal: boolean): void => {
      for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
          let sum = 0;
          let n = 0;
          for (let k = -r; k <= r; k++) {
            const xx = horizontal ? x + k : x;
            const yy = horizontal ? y : y + k;
            if (xx < 0 || xx >= width || yy < 0 || yy >= height) continue;
            sum += src[yy * width + xx] as number;
            n++;
          }
          dst[y * width + x] = sum / Math.max(n, 1);
        }
      }
    };
    blur(weight, tmp, true);
    blur(tmp, weight, false);
  }

  const out = new Uint8ClampedArray(width * height * 4);
  for (let i = 0; i < width * height; i++) {
    const w = Math.max(0, Math.min(1, weight[i] as number));
    for (let c = 0; c < 3; c++) {
      const b = base[i * 4 + c] as number;
      const p = painted[i * 4 + c] as number;
      out[i * 4 + c] = b * (1 - w) + p * w;
    }
    out[i * 4 + 3] = 255;
  }
  return out;
}

/**
 * インペイントを使わないときの色（v1 方式の縮退、docs/03 §3.7「失敗時の縮退」）。
 *
 * マスクされた画素に、段差の奥側の色を伸ばして入れる。縞になって流れるが、
 * 何も無いよりはよい。プレビューはこれで出し、モデルが通れば差し替える。
 */
export function stretchFallback(
  color: ArrayLike<number>,
  mask: ArrayLike<number>,
  width: number,
  height: number,
): Uint8ClampedArray {
  const out = new Uint8ClampedArray(color.length);
  for (let i = 0; i < color.length; i++) out[i] = color[i] as number;

  // 左右方向に、マスクされていない最寄りの色で埋める。
  // 縦横4方向を平均する手もあるが、露出は横に振ったときに出るので
  // 横方向の伸長で足りる。
  for (let y = 0; y < height; y++) {
    let lastGood = -1;
    for (let x = 0; x < width; x++) {
      const i = y * width + x;
      if ((mask[i] as number) < 128) {
        lastGood = i;
        continue;
      }
      if (lastGood >= 0) {
        for (let c = 0; c < 4; c++) out[i * 4 + c] = out[lastGood * 4 + c] as number;
      }
    }
    // 行頭側にマスクが残っていれば、右から左へもう一度なめる
    let nextGood = -1;
    for (let x = width - 1; x >= 0; x--) {
      const i = y * width + x;
      if ((mask[i] as number) < 128) {
        nextGood = i;
        continue;
      }
      if (nextGood >= 0 && (mask[i] as number) >= 128) {
        // 左からの伸長が届いていない（行頭側）画素だけ埋める
        let filled = false;
        for (let c = 0; c < 3; c++) if (out[i * 4 + c] !== (color[i * 4 + c] as number)) filled = true;
        if (!filled) for (let c = 0; c < 4; c++) out[i * 4 + c] = out[nextGood * 4 + c] as number;
      }
    }
  }
  return out;
}
