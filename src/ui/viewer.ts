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
    const pointers = new Map<number, { x: number; y: number }>();
    let startYaw = 0;
    let startPitch = 0;
    let startDist = 1;
    let pinchStart = 0;

    const stopIntro = (): void => {
      // 触れた瞬間に導入アニメーションは止める。勝手に動き続けるほうが不快。
      this.introUntil = 0;
    };

    const down = (e: PointerEvent): void => {
      el.setPointerCapture(e.pointerId);
      pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
      startYaw = this.view.yaw;
      startPitch = this.view.pitch;
      startDist = this.view.distance;
      if (pointers.size === 2) pinchStart = pinchDistance(pointers);
      stopIntro();
    };

    const move = (e: PointerEvent): void => {
      const p = pointers.get(e.pointerId);
      if (!p) return;
      const first = pointers.values().next().value as { x: number; y: number };
      pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });

      if (pointers.size >= 2) {
        const d = pinchDistance(pointers);
        if (pinchStart > 0) this.applyView(this.view.yaw, this.view.pitch, startDist * (pinchStart / d));
        return;
      }
      // 画面の幅いっぱいのドラッグで、快適範囲のちょうど2倍だけ回る。
      const dx = e.clientX - (first?.x ?? e.clientX);
      const dy = e.clientY - (first?.y ?? e.clientY);
      const w = el.clientWidth || 1;
      const h = el.clientHeight || 1;
      this.applyView(
        startYaw - (dx / w) * LIMITS.yawComfort * 4,
        startPitch - (dy / h) * LIMITS.pitchComfort * 4,
        this.view.distance,
      );
    };

    const up = (e: PointerEvent): void => {
      pointers.delete(e.pointerId);
      pinchStart = 0;
      if (pointers.size === 0) {
        startYaw = this.view.yaw;
        startPitch = this.view.pitch;
      }
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
