/**
 * WebGL2 による splat レンダラ（決定 D19）。
 *
 * PoC-1 で「WebGPU が使えない実機が現実に存在する」ことが分かったため、
 * WGSL 版と並ぶ互換経路として用意する。見え方は WGSL 版と揃える。
 *
 * WebGPU 版との違いは2点だけ。
 *
 *   1. コンピュートシェーダが無いので、**ソートと背面カリングを CPU で行う**。
 *      WGSL 版と同じ「深度バケットの計数ソート」を JS で実装する。O(n) なので
 *      42万個でも数ミリ秒で済む。
 *   2. ストレージバッファが無いので、スプラットを RGBA32UI テクスチャに置き、
 *      ソート結果だけをインスタンス属性で渡す。毎回スプラット全体（10MB）を
 *      アップロードし直すより、番号だけ（1.7MB）送るほうがはるかに軽い。
 *
 * カメラの回転が閾値未満なら再ソートしない（docs/06 §6.5）。静止時と
 * ゆっくりした操作ではソートコストが実質ゼロになる。
 */
import splatVert from '../glsl/splat.vert?raw';
import splatFrag from '../glsl/splat.frag?raw';
import {
  SCENE_RADIUS,
  SPLAT_BYTES,
  type RenderStats,
  type SplatRenderer,
  type ViewState,
  defaultView,
} from '../SplatRenderer';
import { decodeOct } from '../../codec/pack';

/** 深度バケット数。WGSL 版と同じ。 */
const BUCKETS = 8192;
/** 再ソートする回転角の閾値（ラジアン）。約3°。 */
const RESORT_ANGLE = 0.052;

export interface Webgl2RendererOptions {
  canvas: HTMLCanvasElement;
  /** 背面カリングの閾値の cos。既定 −0.15（真横で切るとシルエットが欠けるため）。 */
  cullCos?: number;
  /** 2D 低域フィルタの強さ（画素単位）。既定 1.0。 */
  filter2d?: number;
}

export class Webgl2SplatRenderer implements SplatRenderer {
  private readonly gl: WebGL2RenderingContext;
  private readonly cullCos: number;
  private readonly filter2d: number;

  private program: WebGLProgram | null = null;
  private vao: WebGLVertexArrayObject | null = null;
  private splatTex: WebGLTexture | null = null;
  private cornerBuf: WebGLBuffer | null = null;
  private indexBuf: WebGLBuffer | null = null;
  private uniforms: Record<string, WebGLUniformLocation | null> = {};

  /** CPU 側に保持するスプラットの生データ。ソートで読む。 */
  private splatData: DataView | null = null;
  private splatCount = 0;
  private texWidth = 0;

  // ソートの作業領域。毎フレーム確保し直さない。
  private histogram = new Uint32Array(BUCKETS);
  private cursor = new Uint32Array(BUCKETS);
  private sortedIndices = new Uint32Array(0);
  private visibleCount = 0;
  private lastSortYaw = Number.NaN;
  private lastSortPitch = Number.NaN;

  private view: ViewState = defaultView();
  private width = 1;
  private height = 1;
  private lodStride = 1;

  readonly stats: RenderStats = { drawnSplats: 0, frameMs: 0, lodStride: 1 };

  constructor(opts: Webgl2RendererOptions) {
    const gl = opts.canvas.getContext('webgl2', {
      alpha: true,
      premultipliedAlpha: false,
      antialias: false,
      // 読み戻さないので保持しない。モバイルでのメモリ節約になる。
      preserveDrawingBuffer: false,
    });
    if (!gl) throw new Error('WebGL2 のコンテキストを取得できませんでした');
    this.gl = gl;
    this.cullCos = opts.cullCos ?? -0.15;
    this.filter2d = opts.filter2d ?? 1.0;
    this.buildProgram();
  }

  /** この端末で WebGL2 レンダラが動くか。 */
  static isSupported(): boolean {
    try {
      const c = document.createElement('canvas');
      const gl = c.getContext('webgl2');
      if (!gl) return false;
      // 整数テクスチャと texelFetch が要る。WebGL2 なら仕様上必ずあるが、
      // 実装によっては RGBA32UI が描画不可なことがあるので念のため確かめる。
      return gl.getParameter(gl.MAX_TEXTURE_SIZE) >= 2048;
    } catch {
      return false;
    }
  }

  private compile(type: number, source: string, label: string): WebGLShader {
    const gl = this.gl;
    const shader = gl.createShader(type);
    if (!shader) throw new Error(`${label} シェーダを作成できませんでした`);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
      const log = gl.getShaderInfoLog(shader);
      gl.deleteShader(shader);
      throw new Error(`${label} シェーダのコンパイルに失敗しました:\n${log}`);
    }
    return shader;
  }

  private buildProgram(): void {
    const gl = this.gl;
    const vs = this.compile(gl.VERTEX_SHADER, splatVert, '頂点');
    const fs = this.compile(gl.FRAGMENT_SHADER, splatFrag, 'フラグメント');
    const program = gl.createProgram();
    if (!program) throw new Error('プログラムを作成できませんでした');
    gl.attachShader(program, vs);
    gl.attachShader(program, fs);
    gl.linkProgram(program);
    gl.deleteShader(vs);
    gl.deleteShader(fs);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      const log = gl.getProgramInfoLog(program);
      gl.deleteProgram(program);
      throw new Error(`プログラムのリンクに失敗しました:\n${log}`);
    }
    this.program = program;
    for (const name of ['uSplats', 'uTexWidth', 'uViewProj', 'uViewport', 'uFilter2d']) {
      this.uniforms[name] = gl.getUniformLocation(program, name);
    }

    // クアッドの4隅（triangle-strip）。全インスタンスで共有する。
    this.cornerBuf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.cornerBuf);
    gl.bufferData(
      gl.ARRAY_BUFFER,
      new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]),
      gl.STATIC_DRAW,
    );
  }

  setSplats(data: ArrayBufferView, count: number): void {
    const gl = this.gl;
    this.splatCount = count;
    this.splatData = new DataView(data.buffer, data.byteOffset, data.byteLength);
    this.sortedIndices = new Uint32Array(count);
    this.lastSortYaw = Number.NaN;

    // 1スプラット = 2テクセル（RGBA32UI）。幅は 2 の倍数に揃える。
    const maxTex = gl.getParameter(gl.MAX_TEXTURE_SIZE) as number;
    const texels = count * 2;
    this.texWidth = Math.min(maxTex, 2048);
    const rows = Math.max(1, Math.ceil(texels / this.texWidth));
    if (rows > maxTex) {
      throw new Error(
        `スプラットが多すぎてテクスチャに収まりません（${count} 個、上限 ${maxTex * this.texWidth / 2} 個）`,
      );
    }

    // 24 バイトの並びをそのまま 2 テクセル（32 バイト）に写す。
    //   texel0: [posX, posY, posZ, normal]  texel1: [scale, color, 0, 0]
    const packed = new Uint32Array(this.texWidth * rows * 4);
    const src = new Uint32Array(data.buffer, data.byteOffset, Math.floor(data.byteLength / 4));
    const stride = SPLAT_BYTES / 4;
    for (let i = 0; i < count; i++) {
      const s = i * stride;
      const d = i * 8;
      packed[d] = src[s] ?? 0;
      packed[d + 1] = src[s + 1] ?? 0;
      packed[d + 2] = src[s + 2] ?? 0;
      packed[d + 3] = src[s + 3] ?? 0;
      packed[d + 4] = src[s + 4] ?? 0;
      packed[d + 5] = src[s + 5] ?? 0;
    }

    if (this.splatTex) gl.deleteTexture(this.splatTex);
    this.splatTex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, this.splatTex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(
      gl.TEXTURE_2D, 0, gl.RGBA32UI, this.texWidth, rows, 0,
      gl.RGBA_INTEGER, gl.UNSIGNED_INT, packed,
    );

    if (this.indexBuf) gl.deleteBuffer(this.indexBuf);
    this.indexBuf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.indexBuf);
    gl.bufferData(gl.ARRAY_BUFFER, count * 4, gl.DYNAMIC_DRAW);

    this.buildVao();
  }

  private buildVao(): void {
    const gl = this.gl;
    if (this.vao) gl.deleteVertexArray(this.vao);
    this.vao = gl.createVertexArray();
    gl.bindVertexArray(this.vao);

    const cornerLoc = gl.getAttribLocation(this.program as WebGLProgram, 'aCorner');
    gl.bindBuffer(gl.ARRAY_BUFFER, this.cornerBuf);
    gl.enableVertexAttribArray(cornerLoc);
    gl.vertexAttribPointer(cornerLoc, 2, gl.FLOAT, false, 0, 0);
    gl.vertexAttribDivisor(cornerLoc, 0);

    const indexLoc = gl.getAttribLocation(this.program as WebGLProgram, 'aIndex');
    gl.bindBuffer(gl.ARRAY_BUFFER, this.indexBuf);
    gl.enableVertexAttribArray(indexLoc);
    // 整数属性なので IPointer を使う。Pointer だと float に正規化されて壊れる。
    gl.vertexAttribIPointer(indexLoc, 1, gl.UNSIGNED_INT, 0, 0);
    gl.vertexAttribDivisor(indexLoc, 1);

    gl.bindVertexArray(null);
  }

  setCamera(view: ViewState): void {
    this.view = view;
  }

  setLodStride(stride: number): void {
    const next = Math.max(1, Math.floor(stride));
    if (next !== this.lodStride) this.lastSortYaw = Number.NaN;
    this.lodStride = next;
    this.stats.lodStride = next;
  }

  resize(width: number, height: number): void {
    this.width = Math.max(1, Math.floor(width));
    this.height = Math.max(1, Math.floor(height));
    const canvas = this.gl.canvas as HTMLCanvasElement;
    canvas.width = this.width;
    canvas.height = this.height;
  }

  /**
   * 背面カリングと深度バケットの計数ソートを CPU で行う（WGSL 版の compute パス相当）。
   *
   * 被写体が単位立方体に正規化されているので、基数ソート4パスではなく1パスの計数ソートで
   * 足りる。バケットは**このフレームのカメラからの距離** [d − R, d + R] で張る
   * （R は `SCENE_RADIUS`）。幅は被写体1mで約 0.22mm、サーフェル1個より細かい。
   *
   * v2.6.7 まではここに「生成時のカメラ（距離 1.0）から見た深度レンジ」を使っていて、
   * **距離 1.0 以外では全部が 1 バケットに潰れていた**。docs/09 §V22。
   */
  private sort(eye: readonly [number, number, number], viewMat: Float32Array): void {
    const data = this.splatData;
    if (!data) return;

    this.histogram.fill(0);
    const stride = SPLAT_BYTES;
    const sortNear = Math.max(1e-3, this.view.distance - SCENE_RADIUS);
    const range = Math.max(this.view.distance + SCENE_RADIUS - sortNear, 1e-6);
    const n = this.splatCount;
    const step = this.lodStride;

    // 1周目: 可視判定とヒストグラム
    const depths = new Float32Array(Math.ceil(n / step));
    const buckets = new Int32Array(Math.ceil(n / step));
    let k = 0;
    let visible = 0;
    for (let i = 0; i < n; i += step, k++) {
      const o = i * stride;
      const px = data.getFloat32(o, true);
      const py = data.getFloat32(o + 4, true);
      const pz = data.getFloat32(o + 8, true);

      const tx = eye[0] - px;
      const ty = eye[1] - py;
      const tz = eye[2] - pz;
      const d = Math.hypot(tx, ty, tz);
      if (d < 1e-6) { buckets[k] = -1; continue; }

      // サーフェルなので法線で表裏が決まる。通常の 3DGS には無い最適化。
      const [nx, ny, nz] = decodeOct(data.getUint32(o + 12, true));
      if ((nx * tx + ny * ty + nz * tz) / d < this.cullCos) { buckets[k] = -1; continue; }

      // ビュー空間の z（カメラ前方が負）。列優先なので3行目の成分を拾う。
      const vz =
        (viewMat[2] as number) * px +
        (viewMat[6] as number) * py +
        (viewMat[10] as number) * pz +
        (viewMat[14] as number);
      const depth = -vz;
      if (!(depth > 0)) { buckets[k] = -1; continue; }

      const t = Math.min(1, Math.max(0, (depth - sortNear) / range));
      // 遠い→手前 の順に並べたいので反転する
      const b = Math.min(BUCKETS - 1, ((1 - t) * (BUCKETS - 1)) | 0);
      buckets[k] = b;
      depths[k] = depth;
      this.histogram[b] = (this.histogram[b] as number) + 1;
      visible++;
    }

    // 排他的プレフィックス和
    let running = 0;
    for (let b = 0; b < BUCKETS; b++) {
      this.cursor[b] = running;
      running += this.histogram[b] as number;
    }

    // 2周目: 書き出し
    k = 0;
    for (let i = 0; i < n; i += step, k++) {
      const b = buckets[k] as number;
      if (b < 0) continue;
      const slot = this.cursor[b] as number;
      this.cursor[b] = slot + 1;
      this.sortedIndices[slot] = i;
    }

    this.visibleCount = visible;
    this.stats.drawnSplats = visible;

    const gl = this.gl;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.indexBuf);
    gl.bufferSubData(gl.ARRAY_BUFFER, 0, this.sortedIndices, 0, visible);
  }

  render(): void {
    const gl = this.gl;
    if (!this.program || !this.vao || this.splatCount === 0) return;
    const t0 = performance.now();

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
    const viewMat = lookAt(eye, target, [0, -1, 0]);
    const viewProj = mul(perspective(fovY, aspect, 0.01, 100), viewMat);

    // 回転が小さいうちは前回の順序を使い回す（docs/06 §6.5）
    const moved =
      Math.abs(yaw - this.lastSortYaw) > RESORT_ANGLE ||
      Math.abs(pitch - this.lastSortPitch) > RESORT_ANGLE ||
      Number.isNaN(this.lastSortYaw);
    if (moved) {
      this.sort(eye, viewMat);
      this.lastSortYaw = yaw;
      this.lastSortPitch = pitch;
    }

    gl.viewport(0, 0, this.width, this.height);
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.disable(gl.DEPTH_TEST);
    gl.enable(gl.BLEND);
    // 遠→近にソートしてあるので通常の "over" で正しい
    // α は ONE で足す。SRC_ALPHA を掛けると a·a になり、面が透けたままになる
    // （WGSL 版と同じ規約。src/render/backends/wgsl.ts を参照）。
    gl.blendFuncSeparate(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA, gl.ONE, gl.ONE_MINUS_SRC_ALPHA);

    gl.useProgram(this.program);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.splatTex);
    gl.uniform1i(this.uniforms['uSplats'] ?? null, 0);
    gl.uniform1i(this.uniforms['uTexWidth'] ?? null, this.texWidth);
    gl.uniformMatrix4fv(this.uniforms['uViewProj'] ?? null, false, viewProj);
    gl.uniform2f(this.uniforms['uViewport'] ?? null, this.width, this.height);
    gl.uniform1f(this.uniforms['uFilter2d'] ?? null, this.filter2d);

    gl.bindVertexArray(this.vao);
    gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, this.visibleCount);
    gl.bindVertexArray(null);

    this.stats.frameMs = performance.now() - t0;
  }

  /**
   * 描画の完了を待つ。
   *
   * `gl.finish()` はブロックするので、fence sync を使ってポーリングする。
   * fence が使えない場合だけ finish() に落とす。
   */
  async flush(): Promise<void> {
    const gl = this.gl;
    const sync = gl.fenceSync(gl.SYNC_GPU_COMMANDS_COMPLETE, 0);
    if (!sync) {
      gl.finish();
      return;
    }
    gl.flush();
    try {
      for (;;) {
        const status = gl.clientWaitSync(sync, 0, 0);
        if (status === gl.ALREADY_SIGNALED || status === gl.CONDITION_SATISFIED) return;
        if (status === gl.WAIT_FAILED) {
          gl.finish();
          return;
        }
        await new Promise<void>((resolve) => setTimeout(resolve, 0));
      }
    } finally {
      gl.deleteSync(sync);
    }
  }

  /**
   * 既定のフレームバッファから読む。
   *
   * `preserveDrawingBuffer: false` なので、合成が走ると中身は消える。
   * よって `render()` と同じタスクの中で（await を挟まずに）呼ぶこと。
   * gl.readPixels は左下原点なので、上下を入れ替えて返す。
   */
  async readPixels(): Promise<Uint8Array> {
    const gl = this.gl;
    const w = gl.drawingBufferWidth;
    const h = gl.drawingBufferHeight;
    const flipped = new Uint8Array(w * h * 4);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.readPixels(0, 0, w, h, gl.RGBA, gl.UNSIGNED_BYTE, flipped);
    const out = new Uint8Array(w * h * 4);
    const row = w * 4;
    for (let y = 0; y < h; y++) {
      out.set(flipped.subarray((h - 1 - y) * row, (h - y) * row), y * row);
    }
    return out;
  }

  dispose(): void {
    const gl = this.gl;
    if (this.vao) gl.deleteVertexArray(this.vao);
    if (this.splatTex) gl.deleteTexture(this.splatTex);
    if (this.cornerBuf) gl.deleteBuffer(this.cornerBuf);
    if (this.indexBuf) gl.deleteBuffer(this.indexBuf);
    if (this.program) gl.deleteProgram(this.program);
    this.splatData = null;
  }
}

// --- 行列演算（列優先。WGSL 版と同じ実装を共有したいが、依存を増やさないため複製） ---

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
