/**
 * 合成の人体で、位置合わせの数式を検算するための描画（docs/12 §12.15）。
 *
 * 実素材で測る前にここを通す。閉形式のシフト推定を合成球で確かめた
 * （docs/01 改訂 #16）のと同じやり方である。既知の角度で回した3枚を作り、
 * その角度が復元できることを見る。
 *
 * ## 形をどう選んだか
 *
 * 最初は「楕円柱＋頭の球」で作った。**これは検算にならなかった。**
 * 楕円柱はどの高さで切っても同じ断面なので、回してもシルエットの形がほとんど
 * 変わらず、30° ずらしても点の 8% しかはみ出さない。実際の人体は肩・腰・頭で
 * 断面がまるで違うので、この形は問題を実際より難しく見せていた（docs/12 §12.15.2）。
 *
 * いまは**楕円体を縦に積む**。高さごとに幅と奥行きの比が変わるので、
 * 回すとシルエットの形が変わる。人体が実際にそうであるように。
 *
 *   頭・首・胸（幅広で薄い）・腰・尻（幅広）・鼻（前へ出る）
 *
 * 座標は本体と同じ約束（src/pipeline/align/rigid.ts）。y は**下**が正。
 */

/** 楕円体1つ。中心は体の座標（回す前）。 */
export interface BodyPart {
  readonly x: number;
  readonly y: number;
  /** 体の座標の z。正面（カメラ側）が負。 */
  readonly z: number;
  readonly rx: number;
  readonly ry: number;
  readonly rz: number;
}

export interface SyntheticBody {
  /** 回転軸の位置（世界座標の x, z）。 */
  readonly axisX: number;
  readonly axisZ: number;
  readonly parts: readonly BodyPart[];
}

/**
 * 立っている人のつもりの形。
 * 幅（rx）と奥行き（rz）の比が部位ごとに違うのが肝で、ここが yaw の手がかりになる。
 */
export const DEFAULT_BODY: SyntheticBody = {
  axisX: 0,
  axisZ: 3.2,
  parts: [
    { x: 0, y: -0.74, z: -0.01, rx: 0.085, ry: 0.115, rz: 0.1 }, // 頭
    { x: 0, y: -0.6, z: 0, rx: 0.05, ry: 0.05, rz: 0.05 }, // 首
    { x: 0, y: -0.42, z: 0, rx: 0.2, ry: 0.16, rz: 0.1 }, // 肩・胸（幅広で薄い）
    { x: 0, y: -0.18, z: 0, rx: 0.145, ry: 0.14, rz: 0.095 }, // 腰（くびれ）
    { x: 0, y: 0.05, z: 0, rx: 0.185, ry: 0.15, rz: 0.13 }, // 尻（幅も奥行きもある）
    { x: -0.08, y: 0.35, z: 0, rx: 0.075, ry: 0.22, rz: 0.085 }, // 右脚
    { x: 0.08, y: 0.35, z: 0, rx: 0.075, ry: 0.22, rz: 0.085 }, // 左脚
    { x: 0, y: -0.72, z: -0.095, rx: 0.028, ry: 0.035, rz: 0.045 }, // 鼻（向きの符号を作る）
  ],
};

export interface RenderedView {
  readonly width: number;
  readonly height: number;
  readonly alpha: Uint8Array;
  /** カメラからの z。被写体の外は 0。 */
  readonly depth: Float32Array;
}

export interface RenderCamera {
  readonly focalPx: number;
  readonly cx: number;
  readonly cy: number;
}

/**
 * 被写体を鉛直軸まわりに `subjectYaw`[rad] 回して描く。
 *
 * **符号の約束**: 体の正面は自分の座標で −z（＝ yaw 0 でカメラを向く）。
 * `subjectYaw = −90°` で正面が +x、つまり**画面の右**を向く（= 枠の「右向き」）。
 * このとき `registerViews` が返すべき姿勢の yaw は **+90°** である
 * （姿勢は「その view の点を基準へ運ぶ」向きなので符号が逆になる）。
 */
export function renderBody(
  body: SyntheticBody,
  cam: RenderCamera,
  width: number,
  height: number,
  subjectYaw: number,
): RenderedView {
  const alpha = new Uint8Array(width * height);
  const depth = new Float32Array(width * height);
  const cf = Math.cos(subjectYaw);
  const sf = Math.sin(subjectYaw);

  // 各部位の中心を世界座標へ。楕円体の軸は鉛直軸まわりに回るので、
  // ray を部位の座標へ戻してから単位球と当てる。
  const centers = body.parts.map((p) => ({
    cx: body.axisX + cf * p.x + sf * p.z,
    cy: p.y,
    cz: body.axisZ - sf * p.x + cf * p.z,
    rx: p.rx,
    ry: p.ry,
    rz: p.rz,
  }));

  for (let v = 0; v < height; v++) {
    for (let u = 0; u < width; u++) {
      const dx = (u + 0.5 - cam.cx) / cam.focalPx;
      const dy = (v + 0.5 - cam.cy) / cam.focalPx;
      let best = Infinity;

      for (const c of centers) {
        // ray: P(t) = (t·dx, t·dy, t)。中心を引いて、鉛直軸まわりに −yaw 回し、
        // 半径で割ると単位球との交差になる。
        const ox = -c.cx;
        const oz = -c.cz;
        // 方向と原点を部位の座標へ（R_y(−yaw)）
        const dLx = (cf * dx - sf * 1) / c.rx;
        const dLy = dy / c.ry;
        const dLz = (sf * dx + cf * 1) / c.rz;
        const oLx = (cf * ox - sf * oz) / c.rx;
        const oLy = -c.cy / c.ry;
        const oLz = (sf * ox + cf * oz) / c.rz;

        const qa = dLx * dLx + dLy * dLy + dLz * dLz;
        const qb = 2 * (dLx * oLx + dLy * oLy + dLz * oLz);
        const qc = oLx * oLx + oLy * oLy + oLz * oLz - 1;
        const disc = qb * qb - 4 * qa * qc;
        if (disc < 0) continue;
        const sq = Math.sqrt(disc);
        const t0 = (-qb - sq) / (2 * qa);
        const t = t0 > 0 ? t0 : (-qb + sq) / (2 * qa);
        if (t > 0 && t < best) best = t;
      }

      if (best < Infinity) {
        const i = v * width + u;
        alpha[i] = 255;
        depth[i] = best;
      }
    }
  }
  return { width, height, alpha, depth };
}
