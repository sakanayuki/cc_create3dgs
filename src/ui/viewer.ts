/**
 * ビューア（docs/06 §6.7）。
 *
 * 半球カバー（決定 D2）なので、視点は自由に回せてはいけない。裏側の粗が
 * 見えると「壊れている」と受け取られる。快適範囲を超えるにつれ抵抗を増し、
 * 上限で柔らかく止める（ラバーバンド）。
 */
import { createRenderer, type RendererBackend } from '../render/createRenderer';
import { defaultView, type SplatRenderer, type ViewState } from '../render/SplatRenderer';

/** docs/06 §6.7 の可動範囲。 */
export const LIMITS = {
  yawComfort: (45 * Math.PI) / 180,
  yawHard: (60 * Math.PI) / 180,
  pitchComfort: (25 * Math.PI) / 180,
  pitchHard: (35 * Math.PI) / 180,
  distanceMin: 1 / 2.5,
  distanceMax: 1 / 0.6,
} as const;

/**
 * 快適範囲の外で抵抗を増やす。
 *
 * 硬い壁で止めると操作が引っかかった感じになり、制限なく回せると破綻が見える。
 * 快適範囲までは素通し、その先は残りの余地に対して指数的に詰まる。
 */
export function rubberBand(value: number, comfort: number, hard: number): number {
  const a = Math.abs(value);
  if (a <= comfort) return value;
  const over = a - comfort;
  const room = Math.max(hard - comfort, 1e-6);
  // over が増えても room を超えない。tanh は原点で傾き 1 なので、
  // 快適範囲の境界で滑らかに繋がる。
  const damped = room * Math.tanh(over / room);
  return Math.sign(value) * (comfort + damped);
}

/** ドラッグの起点と、そのときの視点。 */
export interface DragAnchor {
  /** 起点の画面座標。 */
  readonly x: number;
  readonly y: number;
  /** 起点を掴んだ瞬間の視点。 */
  readonly yaw: number;
  readonly pitch: number;
}

/**
 * ドラッグの**起点からの**移動量を視点の角度に直す。
 *
 * 画面の幅いっぱいのドラッグで、快適範囲のちょうど 2 倍だけ回る。
 */
export function dragToView(
  anchor: DragAnchor,
  x: number,
  y: number,
  width: number,
  height: number,
): { yaw: number; pitch: number } {
  const w = Math.max(1, width);
  const h = Math.max(1, height);
  return {
    yaw: anchor.yaw - ((x - anchor.x) / w) * LIMITS.yawComfort * 4,
    // 下へ引いたらカメラが上がる（＝上から見下ろす）。指で物を手前へ倒す
    // 感覚に合わせる。横方向とは符号が逆になるが、これは意図である。
    pitch: anchor.pitch + ((y - anchor.y) / h) * LIMITS.pitchComfort * 4,
  };
}

/** ジェスチャが求めた視点。distance が undefined なら距離は据え置き。 */
export interface GestureView {
  readonly yaw: number;
  readonly pitch: number;
  readonly distance?: number;
}

/**
 * 指（ポインタ）の出入りから視点を決める状態機械。
 *
 * DOM から切り離してあるのは、ここが一番間違えやすいからである。実際、
 * 1 イベント分の差分を「掴んだ瞬間の角度」に足していて、どれだけ動かして
 * も起点付近から離れず、**掴んでも回らなかった**。差分で積むなら基準の
 * 角度も一緒に進めなければならない。起点からの絶対量で測るほうが素直で、
 * 丸め誤差も溜まらない。
 *
 * 視点そのものは持たない。ラバーバンドを掛けたあとの値が真なので、
 * 現在値は毎回呼び出し側から受け取る。
 */
export class GestureTracker {
  private readonly pointers = new Map<number, { x: number; y: number }>();
  private anchor: DragAnchor = { x: 0, y: 0, yaw: 0, pitch: 0 };
  private startDistance = 1;
  private pinchStart = 0;

  get pointerCount(): number {
    return this.pointers.size;
  }

  /** いまの指の位置と視点を新しい起点にする。 */
  private reanchor(x: number, y: number, view: GestureView): void {
    this.anchor = { x, y, yaw: view.yaw, pitch: view.pitch };
  }

  down(id: number, x: number, y: number, view: Required<GestureView>): void {
    this.pointers.set(id, { x, y });
    // 指が増えたときも起点を取り直す。そうしないと、2 本目を置いた瞬間に
    // 1 本目との差だけ視点が飛ぶ。
    this.reanchor(x, y, view);
    this.startDistance = view.distance;
    this.pinchStart = this.pointers.size === 2 ? pinchDistance(this.pointers) : 0;
  }

  /** 動かした結果の視点。掴んでいない指なら null。 */
  move(
    id: number,
    x: number,
    y: number,
    width: number,
    height: number,
    view: Required<GestureView>,
  ): GestureView | null {
    if (!this.pointers.has(id)) return null;
    this.pointers.set(id, { x, y });

    if (this.pointers.size >= 2) {
      const d = pinchDistance(this.pointers);
      if (this.pinchStart <= 0 || d <= 0) return null;
      return { yaw: view.yaw, pitch: view.pitch, distance: this.startDistance * (this.pinchStart / d) };
    }
    return dragToView(this.anchor, x, y, width, height);
  }

  up(id: number, view: Required<GestureView>): void {
    this.pointers.delete(id);
    this.pinchStart = 0;
    this.startDistance = view.distance;
    // 指が 1 本残っているなら、その指を新しい起点にする。残った指の位置で
    // 測り直さないと、離した瞬間に視点が飛ぶ。
    const rest = this.pointers.values().next().value as { x: number; y: number } | undefined;
    if (rest) this.reanchor(rest.x, rest.y, view);
  }
}

export interface ViewerOptions {
  readonly canvas: HTMLCanvasElement;
  /** 生成完了時に短い往復アニメーションを入れるか（docs/06 §6.7）。 */
  readonly introAnimation?: boolean;
  readonly onBackend?: (backend: RendererBackend, fallbackReason?: string) => void;
}

export class Viewer {
  private renderer: SplatRenderer | null = null;
  private view: ViewState = defaultView();
  private raf = 0;
  private dirty = true;
  private intro = 0;
  private introUntil = 0;
  private disposed = false;
  private readonly canvas: HTMLCanvasElement;
  private readonly opts: ViewerOptions;
  private readonly detach: (() => void)[] = [];

  backend: RendererBackend | null = null;

  constructor(opts: ViewerOptions) {
    this.opts = opts;
    this.canvas = opts.canvas;
  }

  async init(): Promise<void> {
    const { renderer, backend, fallbackReason } = await createRenderer({ canvas: this.canvas });
    if (this.disposed) {
      renderer.dispose();
      return;
    }
    this.renderer = renderer;
    this.backend = backend;
    this.opts.onBackend?.(backend, fallbackReason);
    this.attachInput();
    this.resize();
    this.loop();
  }

  setSplats(data: ArrayBufferView, count: number, nearZ: number, farZ: number): void {
    if (!this.renderer) return;
    this.renderer.setSplats(data, count);
    this.renderer.setDepthRange(nearZ, farZ);
    this.dirty = true;
    if (this.opts.introAnimation !== false) {
      this.intro = performance.now();
      this.introUntil = this.intro + 1800;
    }
  }

  /** 視点を直接指定する。E2E で角度を変えて撮るのに使う。 */
  /** いまの視点。E2E から操作の結果を確かめるのに使う。 */
  getView(): ViewState {
    return { ...this.view };
  }

  setView(yaw: number, pitch = this.view.pitch, distance = this.view.distance): void {
    this.introUntil = 0;
    this.applyView(yaw, pitch, distance);
  }

  /**
   * いまの視点で描いて、その中身を読み戻す（検証用）。
   *
   * 画面のスクリーンショットでは取れない。描画バッファを保存しない設定
   * （preserveDrawingBuffer: false）なので、合成が走った後には消えている。
   * 描画と読み戻しを同じタスクの中で行う必要がある。
   */
  async capture(): Promise<Uint8Array | null> {
    if (!this.renderer) return null;
    this.renderer.setCamera(this.view);
    this.renderer.render();
    await this.renderer.flush();
    this.renderer.setCamera(this.view);
    this.renderer.render();
    return this.renderer.readPixels();
  }

  resize(): void {
    if (!this.renderer) return;
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const w = Math.max(1, Math.round(this.canvas.clientWidth * dpr));
    const h = Math.max(1, Math.round(this.canvas.clientHeight * dpr));
    this.renderer.resize(w, h);
    this.dirty = true;
  }

  dispose(): void {
    this.disposed = true;
    cancelAnimationFrame(this.raf);
    for (const off of this.detach) off();
    this.renderer?.dispose();
    this.renderer = null;
  }

  private applyView(yaw: number, pitch: number, distance: number): void {
    this.view = {
      yaw: rubberBand(yaw, LIMITS.yawComfort, LIMITS.yawHard),
      pitch: rubberBand(pitch, LIMITS.pitchComfort, LIMITS.pitchHard),
      distance: Math.max(LIMITS.distanceMin, Math.min(LIMITS.distanceMax, distance)),
      target: this.view.target,
    };
    this.dirty = true;
  }

  private attachInput(): void {
    const el = this.canvas;
    const gestures = new GestureTracker();

    const stopIntro = (): void => {
      // 触れた瞬間に導入アニメーションは止める。勝手に動き続けるほうが不快。
      this.introUntil = 0;
    };
    const now = (): Required<GestureView> => ({
      yaw: this.view.yaw,
      pitch: this.view.pitch,
      distance: this.view.distance,
    });

    const down = (e: PointerEvent): void => {
      el.setPointerCapture(e.pointerId);
      gestures.down(e.pointerId, e.clientX, e.clientY, now());
      stopIntro();
    };

    const move = (e: PointerEvent): void => {
      const next = gestures.move(
        e.pointerId,
        e.clientX,
        e.clientY,
        el.clientWidth,
        el.clientHeight,
        now(),
      );
      if (!next) return;
      stopIntro();
      this.applyView(next.yaw, next.pitch, next.distance ?? this.view.distance);
    };

    const up = (e: PointerEvent): void => {
      gestures.up(e.pointerId, now());
    };

    const wheel = (e: WheelEvent): void => {
      e.preventDefault();
      stopIntro();
      this.applyView(this.view.yaw, this.view.pitch, this.view.distance * (1 + e.deltaY * 0.001));
    };

    el.addEventListener('pointerdown', down);
    el.addEventListener('pointermove', move);
    el.addEventListener('pointerup', up);
    el.addEventListener('pointercancel', up);
    el.addEventListener('wheel', wheel, { passive: false });
    const onResize = (): void => this.resize();
    window.addEventListener('resize', onResize);

    this.detach.push(() => {
      el.removeEventListener('pointerdown', down);
      el.removeEventListener('pointermove', move);
      el.removeEventListener('pointerup', up);
      el.removeEventListener('pointercancel', up);
      el.removeEventListener('wheel', wheel);
      window.removeEventListener('resize', onResize);
    });
  }

  private loop = (): void => {
    this.raf = requestAnimationFrame(this.loop);
    if (!this.renderer) return;

    const now = performance.now();
    if (now < this.introUntil) {
      // 生成直後の短い往復。静止画に見えてしまうのを防ぐ（docs/06 §6.7）。
      const t = (now - this.intro) / 1800;
      const swing = Math.sin(t * Math.PI * 2) * (25 * Math.PI) / 180;
      this.applyView(swing, this.view.pitch, this.view.distance);
    }

    if (!this.dirty) return;
    this.renderer.setCamera(this.view);
    this.renderer.render();
    this.dirty = now < this.introUntil;
  };
}

function pinchDistance(pointers: Map<number, { x: number; y: number }>): number {
  const [a, b] = [...pointers.values()];
  if (!a || !b) return 0;
  return Math.hypot(a.x - b.x, a.y - b.y);
}
