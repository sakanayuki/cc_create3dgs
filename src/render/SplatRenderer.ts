/**
 * 描画層の唯一の公開インタフェース（docs/06 §6.2）。
 *
 * D9 で Three.js のネイティブ WebGPU splat レンダラ (r186+) を選んだが、
 * 調査時点の npm 最新 three@0.185.1 には splat モジュールが含まれていない。
 * そこでこの薄い抽象を1枚挟み、まず自前 WGSL 実装で進めて r186 が出たら
 * 差し替えられるようにする。両方を維持し続けることはしない。
 */

/** レンダラが消費するスプラットの GPU 表現。1個 24 バイト（docs/04 §4.8.1）。 */
export const SPLAT_BYTES = 24;

export interface ViewState {
  /** 被写体中心からの方位角（ラジアン）。 */
  yaw: number;
  /** 仰角（ラジアン）。 */
  pitch: number;
  /** 被写体までの距離。1.0 が生成時のカメラ位置。 */
  distance: number;
  /** 注視点のずらし（パン）。 */
  target: [number, number, number];
}

export interface RenderStats {
  /** 直近フレームで実際に描画されたスプラット数（背面カリング後）。 */
  drawnSplats: number;
  /** 直近フレームの CPU 側実測時間（ミリ秒）。 */
  frameMs: number;
  /** LOD の間引き段（1 = 間引きなし）。 */
  lodStride: number;
}

export interface SplatRenderer {
  /** 描画対象を差し替える。プレビュー確定時と微調整完了時に呼ばれる。 */
  setSplats(data: ArrayBufferView, count: number): void;
  setCamera(view: ViewState): void;
  /** 深度レンジ。ソートのバケット割り当てに使う。 */
  setDepthRange(nearZ: number, farZ: number): void;
  render(): void;
  /**
   * 直前の render() の GPU 側の完了を待つ。
   *
   * ベンチで正確な時間を測るために要る。通常の描画ループでは呼ばない
   * （待つとパイプラインが止まって遅くなる）。待ち方はバックエンドで違う。
   */
  flush(): Promise<void>;
  resize(width: number, height: number): void;
  dispose(): void;
  readonly stats: RenderStats;
}

/** 単位立方体に正規化された被写体を見るカメラの既定値。 */
export function defaultView(): ViewState {
  return { yaw: 0, pitch: 0, distance: 1.0, target: [0, 0, 0] };
}
