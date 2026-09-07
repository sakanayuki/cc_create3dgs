/**
 * パイプライン・描画・コーデックの3者が共有する唯一の型（docs/02 §2.4）。
 *
 * 素朴な実装なら「ガウシアンの配列」を持つところを、**画像平面に並んだ属性マップ**
 * として持っている。これが軽量化設計（docs/04）の土台で、次の3つが効く。
 *
 *   1. 位置を保存しなくてよい。(u,v) と深度とカメラから逆投影で一意に決まる
 *   2. 法線とスケールを保存しなくてよい。深度マップから解析的に導出できる
 *   3. 画像コーデックがそのまま使える。隣接ピクセルの値が似ているため
 *
 * 一般の3DGS圧縮ではこの並び順を作るために重いソート（PLAS 等）が要るが、
 * 深度マップ由来のガウシアンは元から画像のピクセル格子に乗っているので不要。
 */

/** 焦点距離をどこから得たか。立体感の信頼度として UI に出す。 */
export type FocalSource = 'da3' | 'exif' | 'assumed';

export interface CameraParams {
  /** 焦点距離（ピクセル単位）。優先順: 深度モデルの推定 → EXIF → 画角55°仮定。 */
  readonly focalPx: number;
  readonly cx: number;
  readonly cy: number;
  readonly focalSource: FocalSource;
}

/** 深度の正規化。z = nearZ + depth01 · (farZ − nearZ) */
export interface DepthRange {
  readonly nearZ: number;
  readonly farZ: number;
}

/** 背面シェルの生成パラメータ。厚みマップ自体は保存せず、α の距離変換から再計算する。 */
export interface BackShellParams {
  readonly enabled: boolean;
  /** 最大厚み。被写体の奥行きの半分。 */
  readonly thicknessT: number;
  /** 断面プロファイル。ellipsoid は sqrt(1−(1−r)²)。円柱状にすると縁が角張る。 */
  readonly profile: 'ellipsoid' | 'cylinder';
  /** 前面画素に対する密度の分母。4 なら 2×2 を1ガウシアンに。 */
  readonly density: number;
  /** 人物は縁色の行方向伸長、物体は水平反転（docs/03 §3.6.2）。 */
  readonly colorMode: 'edge-extend' | 'mirror-h';
  readonly shadeBase: number;
  readonly shadeRange: number;
}

/** スカートの生成パラメータ。幾何は frontDepth と alpha から決定的に導出できる。 */
export interface SkirtParams {
  readonly enabled: boolean;
  /** 深度勾配の閾値（深度レンジ比）。これを超えたら深度エッジとみなす。 */
  readonly edgeThreshold: number;
  /** 深度ギャップに対する帯の長さの比。 */
  readonly lengthScale: number;
  readonly opacityFalloff: 'linear' | 'smooth';
  /** 色の出どころ。inpaint プレーンが無ければ stretch に落ちる。 */
  readonly colorSource: 'inpaint' | 'stretch';
}

/** 適応サンプリング。読み込み時に同じ四分木を再計算するためのパラメータ。 */
export interface SamplingParams {
  readonly enabled: boolean;
  /** 深度分散の閾値（深度レンジ比）。 */
  readonly thetaDepth: number;
  /** 色分散の閾値（sRGB 距離）。 */
  readonly thetaColor: number;
  /** 統合できるセルの最大辺長（画素）。 */
  readonly maxCell: number;
}

export interface DocumentMeta {
  readonly createdAt: string;
  readonly app: string;
  readonly preset: 'light' | 'standard' | 'high';
  readonly subjectMode: 'person' | 'object';
  /** 使ったモデルの識別子。モデルを差し替えたとき再現性を追えるようにする。 */
  readonly models: Readonly<Record<string, string>>;
  readonly gaussianCount?: {
    readonly front: number;
    readonly back: number;
    readonly skirt: number;
  };
}

export interface PhotoSplatDocument {
  /** 作業グリッド。既定 1024×1024（決定 D17）。軽量プリセットは 512。 */
  readonly width: number;
  readonly height: number;
  readonly camera: CameraParams;
  readonly depthRange: DepthRange;

  // ---- 画像平面に並んだ属性（width × height） ----
  /** 前面深度。12bit に量子化した値を 16bit で保持する。 */
  readonly frontDepth: Uint16Array;
  /** 前面色 RGB（3 チャンネル）。 */
  readonly frontColor: Uint8ClampedArray;
  /** 不透明度 兼 占有マスク。0 は「ガウシアンなし」を意味する。 */
  readonly alpha: Uint8ClampedArray;

  // ---- 補助プレーン（半分の解像度） ----
  /** 背面色 RGB（width/2 × height/2）。人物は無地陰影、物体は鏡像。 */
  readonly backColor: Uint8ClampedArray;
  /** 遮蔽部の補完テクスチャ RGBA（width/2 × height/2）。α=0 は「補完なし」。 */
  readonly inpaint: Uint8ClampedArray | null;

  // ---- 決定的に導出される要素（保存しない） ----
  readonly backShell: BackShellParams;
  readonly skirt: SkirtParams;
  readonly sampling: SamplingParams;

  readonly meta: DocumentMeta;
}

/** 画素 (u,v) の深度コードを実距離に直す。 */
export function depthAt(doc: PhotoSplatDocument, u: number, v: number): number {
  const code = doc.frontDepth[v * doc.width + u] ?? 0;
  const t = code / 65535;
  return doc.depthRange.nearZ + t * (doc.depthRange.farZ - doc.depthRange.nearZ);
}

/** 画素 (u,v) を3D位置に逆投影する。位置を保存しなくてよい根拠そのもの。 */
export function unproject(
  doc: PhotoSplatDocument,
  u: number,
  v: number,
): [number, number, number] {
  const z = depthAt(doc, u, v);
  const { focalPx, cx, cy } = doc.camera;
  return [((u - cx) * z) / focalPx, ((v - cy) * z) / focalPx, z];
}

/** 想定される画素数と配列長が噛み合っているか確かめる。 */
export function validateDocument(doc: PhotoSplatDocument): string[] {
  const problems: string[] = [];
  const n = doc.width * doc.height;
  const half = Math.floor(doc.width / 2) * Math.floor(doc.height / 2);

  if (doc.width <= 0 || doc.height <= 0) problems.push('グリッドの大きさが不正です');
  if (doc.frontDepth.length !== n) problems.push(`frontDepth の長さが ${doc.frontDepth.length}（期待 ${n}）`);
  if (doc.frontColor.length !== n * 3) problems.push(`frontColor の長さが ${doc.frontColor.length}（期待 ${n * 3}）`);
  if (doc.alpha.length !== n) problems.push(`alpha の長さが ${doc.alpha.length}（期待 ${n}）`);
  if (doc.backColor.length !== half * 3) problems.push(`backColor の長さが ${doc.backColor.length}（期待 ${half * 3}）`);
  if (doc.inpaint && doc.inpaint.length !== half * 4) {
    problems.push(`inpaint の長さが ${doc.inpaint.length}（期待 ${half * 4}）`);
  }
  if (!(doc.depthRange.farZ > doc.depthRange.nearZ)) problems.push('深度レンジが不正です（farZ ≤ nearZ）');
  if (!(doc.camera.focalPx > 0)) problems.push('焦点距離が不正です');
  return problems;
}
