/**
 * 自前 WGSL による splat レンダラ（docs/06）。
 *
 * 描画パス:
 *   ① clearHistogram → ② histogramPass（背面カリング＋可視数の計数）
 *   → ③ scanPass（8192 バケットのプレフィックス和）→ ④ initCursor
 *   → ⑤ scatterPass（ソート済みインデックスの書き出し）→ ⑥ 描画（indirect）
 *
 * 可視数のカウンタが indirect の instanceCount を兼ねるので、CPU への読み戻しが無い。
 */
import commonWgsl from '../wgsl/common.wgsl?raw';
import sortWgsl from '../wgsl/sort.wgsl?raw';
import splatWgsl from '../wgsl/splat.wgsl?raw';
import {
  SCENE_RADIUS,
  SPLAT_BYTES,
  type RenderStats,
  type SplatRenderer,
  type ViewState,
  defaultView,
} from '../SplatRenderer';

const BUCKETS = 8192;
/** カメラ uniform のバイト数: mat4x4 ×2 + vec3+f32 + vec2 + f32×4 + u32×2 → 16 バイト境界に丸める */
const CAMERA_BYTES = 64 + 64 + 16 + 16 + 16 + 16;

export interface WgslRendererOptions {
  device: GPUDevice;
  canvas: HTMLCanvasElement | OffscreenCanvas;
  /** 背面カリングの閾値の cos。既定 -0.15（真横で切るとシルエットが欠けるため）。 */
  cullCos?: number;
  /** 2D 低域フィルタの強さ（画素単位）。既定 1.0。 */
  filter2d?: number;
}

export class WgslSplatRenderer implements SplatRenderer {
  private readonly device: GPUDevice;
  private readonly ctx: GPUCanvasContext;
  private readonly format: GPUTextureFormat;
  private readonly cullCos: number;
  private readonly filter2d: number;

  private splatBuf: GPUBuffer | null = null;
  private sortedBuf: GPUBuffer | null = null;
  private splatCount = 0;

  private readonly camBuf: GPUBuffer;
  private readonly histBuf: GPUBuffer;
  private readonly offsetBuf: GPUBuffer;
  private readonly cursorBuf: GPUBuffer;
  private readonly indirectBuf: GPUBuffer;
  private readonly readbackBuf: GPUBuffer;

  private computePipelines: Record<string, GPUComputePipeline> = {};
  private renderPipeline!: GPURenderPipeline;
  private computeBind: GPUBindGroup | null = null;
  private renderBind: GPUBindGroup | null = null;
  private readonly computeLayout: GPUBindGroupLayout;
  private readonly renderLayout: GPUBindGroupLayout;

  private view: ViewState = defaultView();
  private width = 1;
  private height = 1;
  private lodStride = 1;
  private readbackBusy = false;

  readonly stats: RenderStats = { drawnSplats: 0, frameMs: 0, lodStride: 1 };

  constructor(opts: WgslRendererOptions) {
    this.device = opts.device;
    this.cullCos = opts.cullCos ?? -0.15;
    this.filter2d = opts.filter2d ?? 1.0;

    const ctx = (opts.canvas as HTMLCanvasElement).getContext('webgpu');
    if (!ctx) throw new Error('WebGPU のキャンバスコンテキストを取得できませんでした');
    this.ctx = ctx as unknown as GPUCanvasContext;
    this.format = navigator.gpu.getPreferredCanvasFormat();
    this.ctx.configure({
      device: this.device,
      format: this.format,
      alphaMode: 'premultiplied',
      // COPY_SRC は readPixels() のため。描画そのものには要らないが、
      // 後から付け替えられないので最初から付けておく（コストは無い）。
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
    });

    const d = this.device;
    const S = GPUBufferUsage.STORAGE;
    this.camBuf = d.createBuffer({ size: CAMERA_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.histBuf = d.createBuffer({ size: BUCKETS * 4, usage: S });
    this.offsetBuf = d.createBuffer({ size: BUCKETS * 4, usage: S });
    this.cursorBuf = d.createBuffer({ size: BUCKETS * 4, usage: S });
    this.indirectBuf = d.createBuffer({
      size: 16,
      usage: S | GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_SRC,
    });
    this.readbackBuf = d.createBuffer({ size: 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });

    this.computeLayout = d.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      ],
    });
    this.renderLayout = d.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
        { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform' } },
        { binding: 2, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
      ],
    });

    this.buildPipelines();
  }

  private buildPipelines(): void {
    const d = this.device;
    const sortModule = d.createShaderModule({
      label: 'sort',
      code: `${commonWgsl}\n${sortWgsl}`,
    });
    const splatModule = d.createShaderModule({
      label: 'splat',
      code: `${commonWgsl}\n${splatWgsl}`,
    });

    const computePipelineLayout = d.createPipelineLayout({ bindGroupLayouts: [this.computeLayout] });
    for (const entryPoint of ['clearHistogram', 'histogramPass', 'scanPass', 'initCursor', 'scatterPass']) {
      this.computePipelines[entryPoint] = d.createComputePipeline({
        label: entryPoint,
        layout: computePipelineLayout,
        compute: { module: sortModule, entryPoint },
      });
    }

    this.renderPipeline = d.createRenderPipeline({
      label: 'splat',
      layout: d.createPipelineLayout({ bindGroupLayouts: [this.renderLayout] }),
      vertex: { module: splatModule, entryPoint: 'vs' },
      fragment: {
        module: splatModule,
        entryPoint: 'fs',
        targets: [
          {
            format: this.format,
            // 遠→近にソートしてあるので通常の "over" で正しい
            // α は 'one' で足す。フラグメントが返すのはストレート α なので、
            // 色は src-alpha で乗じて事前乗算にするのが正しいが、α まで
            // src-alpha を掛けると a·a になって毎回小さく積まれる。
            // α 0.77 のサーフェルが 0.59 しか積まれず、面が透けたままになる。
            blend: {
              color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha', operation: 'add' },
              alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
            },
          },
        ],
      },
      primitive: { topology: 'triangle-strip' },
    });
  }

  setSplats(data: ArrayBufferView, count: number): void {
    this.splatBuf?.destroy();
    this.sortedBuf?.destroy();

    const bytes = Math.max(count * SPLAT_BYTES, SPLAT_BYTES);
    this.splatBuf = this.device.createBuffer({
      size: Math.ceil(bytes / 4) * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(this.splatBuf, 0, data as ArrayBufferView & { buffer: ArrayBuffer });

    this.sortedBuf = this.device.createBuffer({
      size: Math.max(count * 4, 4),
      usage: GPUBufferUsage.STORAGE,
    });
    this.splatCount = count;

    this.computeBind = this.device.createBindGroup({
      layout: this.computeLayout,
      entries: [
        { binding: 0, resource: { buffer: this.splatBuf } },
        { binding: 1, resource: { buffer: this.camBuf } },
        { binding: 2, resource: { buffer: this.histBuf } },
        { binding: 3, resource: { buffer: this.offsetBuf } },
        { binding: 4, resource: { buffer: this.sortedBuf } },
        { binding: 5, resource: { buffer: this.indirectBuf } },
        { binding: 6, resource: { buffer: this.cursorBuf } },
      ],
    });
    this.renderBind = this.device.createBindGroup({
      layout: this.renderLayout,
      entries: [
        { binding: 0, resource: { buffer: this.splatBuf } },
        { binding: 1, resource: { buffer: this.camBuf } },
        { binding: 2, resource: { buffer: this.sortedBuf } },
      ],
    });
  }

  setCamera(view: ViewState): void {
    this.view = view;
  }

  resize(width: number, height: number): void {
    this.width = Math.max(1, Math.floor(width));
    this.height = Math.max(1, Math.floor(height));
    const canvas = this.ctx.canvas as HTMLCanvasElement;
    canvas.width = this.width;
    canvas.height = this.height;
  }

  /** LOD の間引き段を外から設定する。フレーム時間の監視結果で動かす（docs/06 §6.6）。 */
  setLodStride(stride: number): void {
    this.lodStride = Math.max(1, Math.floor(stride));
    this.stats.lodStride = this.lodStride;
  }

  private writeCamera(): void {
    const { yaw, pitch, distance, target } = this.view;
    const cp = Math.cos(pitch);
    // ワールドは X 右・**Y 下**・**Z 前（奥）**（6-splats.ts の toWorld）。
    // カメラは被写体の手前、つまり −Z 側に置いて +Z を向く。pitch を上げたら
    // 見下ろす向き、すなわち y が小さいほうへ動く。
    const eye: [number, number, number] = [
      target[0] + distance * cp * Math.sin(yaw),
      target[1] - distance * Math.sin(pitch),
      target[2] - distance * cp * Math.cos(yaw),
    ];
    const aspect = this.width / this.height;
    const fovY = 2 * Math.atan(0.5 / distance);
    const view = lookAt(eye, target, [0, -1, 0]);
    const proj = perspective(fovY, aspect, 0.01, 100);
    const viewProj = mul(proj, view);

    const buf = new ArrayBuffer(CAMERA_BYTES);
    const f = new Float32Array(buf);
    const u = new Uint32Array(buf);
    f.set(viewProj, 0);
    f.set(view, 16);
    f.set(eye, 32);
    f[35] = this.height / (2 * Math.tan(fovY / 2)); // focalPx
    f[36] = this.width;
    f[37] = this.height;
    // ソートのバケットは「このフレームのカメラからの距離」で張る（docs/09 §V22）。
    // 生成時のカメラ（距離 1.0）の深度レンジを使うと、寄ったり引いたりした
    // 途端に全部が 1 バケットに潰れ、ソートが効かなくなる。
    f[38] = Math.max(1e-3, distance - SCENE_RADIUS);
    f[39] = distance + SCENE_RADIUS;
    f[40] = this.cullCos;
    f[41] = this.filter2d;
    u[42] = this.splatCount;
    u[43] = this.lodStride;
    this.device.queue.writeBuffer(this.camBuf, 0, buf);
  }

  render(): void {
    if (!this.splatBuf || !this.computeBind || !this.renderBind || this.splatCount === 0) return;
    const t0 = performance.now();
    this.writeCamera();

    const enc = this.device.createCommandEncoder();
    const groups = Math.ceil(this.splatCount / this.lodStride / 256);

    const cp = enc.beginComputePass();
    cp.setBindGroup(0, this.computeBind);
    const run = (name: string, n: number) => {
      const pipeline = this.computePipelines[name];
      if (!pipeline) throw new Error(`パイプラインがありません: ${name}`);
      cp.setPipeline(pipeline);
      cp.dispatchWorkgroups(n);
    };
    run('clearHistogram', BUCKETS / 256);
    run('histogramPass', Math.max(1, groups));
    run('scanPass', 1);
    run('initCursor', BUCKETS / 256);
    run('scatterPass', Math.max(1, groups));
    cp.end();

    const rp = enc.beginRenderPass({
      colorAttachments: [
        {
          view: this.ctx.getCurrentTexture().createView(),
          clearValue: { r: 0, g: 0, b: 0, a: 0 },
          loadOp: 'clear',
          storeOp: 'store',
        },
      ],
    });
    rp.setPipeline(this.renderPipeline);
    rp.setBindGroup(0, this.renderBind);
    rp.drawIndirect(this.indirectBuf, 0);
    rp.end();

    // 可視数は統計用にだけ読む。描画自体は indirect なので読み戻しを待たない。
    if (!this.readbackBusy) {
      enc.copyBufferToBuffer(this.indirectBuf, 0, this.readbackBuf, 0, 16);
    }
    this.device.queue.submit([enc.finish()]);

    if (!this.readbackBusy) {
      this.readbackBusy = true;
      void this.readbackBuf
        .mapAsync(GPUMapMode.READ)
        .then(() => {
          this.stats.drawnSplats = new Uint32Array(this.readbackBuf.getMappedRange().slice(0))[1] ?? 0;
          this.readbackBuf.unmap();
        })
        .catch(() => {})
        .finally(() => {
          this.readbackBusy = false;
        });
    }
    this.stats.frameMs = performance.now() - t0;
  }

  /** 投入済みの作業が終わるまで待つ。WebGPU はこれが最も実測に近い。 */
  async flush(): Promise<void> {
    await this.device.queue.onSubmittedWorkDone();
  }

  /**
   * キャンバステクスチャをバッファへコピーして読む。
   *
   * copyTextureToBuffer の bytesPerRow は 256 の倍数でなければならないので、
   * 詰め物を入れて転送し、行ごとに切り出して詰め直す。
   * 既定のキャンバス形式は多くの環境で bgra8unorm なので、その場合は
   * 赤と青を入れ替えて RGBA として返す。
   */
  async readPixels(): Promise<Uint8Array> {
    const w = this.width;
    const h = this.height;
    const bytesPerRow = Math.ceil(w * 4 / 256) * 256;
    const buf = this.device.createBuffer({
      size: bytesPerRow * h,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    try {
      const enc = this.device.createCommandEncoder();
      enc.copyTextureToBuffer(
        { texture: this.ctx.getCurrentTexture() },
        { buffer: buf, bytesPerRow, rowsPerImage: h },
        { width: w, height: h, depthOrArrayLayers: 1 },
      );
      this.device.queue.submit([enc.finish()]);
      await buf.mapAsync(GPUMapMode.READ);
      const src = new Uint8Array(buf.getMappedRange());
      const out = new Uint8Array(w * h * 4);
      const bgra = this.format.startsWith('bgra');
      for (let y = 0; y < h; y++) {
        const s = y * bytesPerRow;
        const d = y * w * 4;
        for (let x = 0; x < w * 4; x += 4) {
          const r = src[s + x] as number;
          const g = src[s + x + 1] as number;
          const b = src[s + x + 2] as number;
          out[d + x] = bgra ? b : r;
          out[d + x + 1] = g;
          out[d + x + 2] = bgra ? r : b;
          out[d + x + 3] = src[s + x + 3] as number;
        }
      }
      buf.unmap();
      return out;
    } finally {
      buf.destroy();
    }
  }

  dispose(): void {
    this.splatBuf?.destroy();
    this.sortedBuf?.destroy();
    this.camBuf.destroy();
    this.histBuf.destroy();
    this.offsetBuf.destroy();
    this.cursorBuf.destroy();
    this.indirectBuf.destroy();
    this.readbackBuf.destroy();
  }
}

// --- 最小限の行列演算（列優先、WGSL の mat4x4<f32> と同じ並び） -----------------

type Mat4 = Float32Array;
type Vec3 = readonly [number, number, number];

function lookAt(eye: Vec3, center: Vec3, up: Vec3): Mat4 {
  const z = norm(sub(eye, center));
  const x = norm(cross(up, z));
  const y = cross(z, x);
  return new Float32Array([
    x[0], y[0], z[0], 0,
    x[1], y[1], z[1], 0,
    x[2], y[2], z[2], 0,
    -dot(x, eye), -dot(y, eye), -dot(z, eye), 1,
  ]);
}

function perspective(fovY: number, aspect: number, near: number, far: number): Mat4 {
  const f = 1 / Math.tan(fovY / 2);
  const nf = 1 / (near - far);
  return new Float32Array([
    f / aspect, 0, 0, 0,
    0, f, 0, 0,
    0, 0, far * nf, -1,
    0, 0, far * near * nf, 0,
  ]);
}

function mul(a: Mat4, b: Mat4): Mat4 {
  const o = new Float32Array(16);
  for (let c = 0; c < 4; c++) {
    for (let r = 0; r < 4; r++) {
      let s = 0;
      for (let k = 0; k < 4; k++) s += (a[k * 4 + r] as number) * (b[c * 4 + k] as number);
      o[c * 4 + r] = s;
    }
  }
  return o;
}

const sub = (a: Vec3, b: Vec3): Vec3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const cross = (a: Vec3, b: Vec3): Vec3 => [
  a[1] * b[2] - a[2] * b[1],
  a[2] * b[0] - a[0] * b[2],
  a[0] * b[1] - a[1] * b[0],
];
const dot = (a: Vec3, b: Vec3): number => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
function norm(a: Vec3): Vec3 {
  const l = Math.hypot(a[0], a[1], a[2]) || 1;
  return [a[0] / l, a[1] / l, a[2] / l];
}
