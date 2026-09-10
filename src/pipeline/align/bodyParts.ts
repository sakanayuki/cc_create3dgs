/**
 * 胴と頭だけを取り出す（docs/12 §12.7「非剛体（O5）への対処」）。
 *
 * 素材の3枚で腕の位置が違う（正面は手を背中に、横向きは体側に下ろしている）。
 * 腕を含めて位置合わせしようとすると、剛体では説明できない差に解が引きずられる。
 * そこで**合わせるのに使わない領域**を大まかに外す。
 *
 * **精度は要らない。** 学習したモデルも姿勢推定も足さない。使うのは
 * シルエットだけで、次の2つの素朴な事実に頼る。
 *
 *   1. 脚は下のほうにある
 *   2. 腕は、その高さの体の**外側**にある
 *
 * 外しすぎても困らない（合わせに使える画素が減るだけ）が、
 * 残しすぎると解が歪む。迷ったら外す側に倒してある。
 */

export interface BodyPartOptions {
  /** 被写体の高さのうち、上から何割を「頭と胴」とみなすか。既定 0.55（腰から下を落とす）。 */
  readonly torsoFraction?: number;
  /** 各行で中央から何割の幅を残すか。既定 0.6（左右の腕を落とす）。 */
  readonly coreWidth?: number;
}

/**
 * 位置合わせに使ってよい画素を 1、使わない画素を 0 で返す。
 *
 * @param alpha 0〜255。128 以上を被写体とみなす。
 */
export function torsoAndHead(
  alpha: ArrayLike<number>,
  width: number,
  height: number,
  options: BodyPartOptions = {},
): Uint8Array {
  const torsoFraction = options.torsoFraction ?? 0.55;
  const coreWidth = options.coreWidth ?? 0.6;
  const out = new Uint8Array(width * height);

  // 被写体の縦の範囲
  let top = -1;
  let bottom = -1;
  for (let y = 0; y < height; y++) {
    let any = false;
    for (let x = 0; x < width; x++) {
      if ((alpha[y * width + x] as number) >= 128) {
        any = true;
        break;
      }
    }
    if (any) {
      if (top < 0) top = y;
      bottom = y;
    }
  }
  if (top < 0) return out;

  const limit = top + (bottom - top + 1) * torsoFraction;
  for (let y = top; y <= bottom && y <= limit; y++) {
    // その行の被写体の左右の端
    let x0 = -1;
    let x1 = -1;
    for (let x = 0; x < width; x++) {
      if ((alpha[y * width + x] as number) >= 128) {
        if (x0 < 0) x0 = x;
        x1 = x;
      }
    }
    if (x0 < 0) continue;
    // 中央の帯だけ残す。腕はこの外側に出る。
    const center = (x0 + x1) / 2;
    const half = ((x1 - x0 + 1) * coreWidth) / 2;
    const lo = Math.ceil(center - half);
    const hi = Math.floor(center + half);
    for (let x = Math.max(x0, lo); x <= Math.min(x1, hi); x++) {
      if ((alpha[y * width + x] as number) >= 128) out[y * width + x] = 1;
    }
  }
  return out;
}
