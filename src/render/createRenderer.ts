/**
 * 端末に応じてレンダラのバックエンドを選ぶ（決定 D19）。
 *
 * WebGPU があればそちら、無ければ WebGL2。どちらも無ければ生成できない。
 * 呼び出し側は `SplatRenderer` インタフェースだけを見る（docs/06 §6.2）。
 */
import type { SplatRenderer } from './SplatRenderer';
import { WgslSplatRenderer } from './backends/wgsl';
import { Webgl2SplatRenderer } from './backends/webgl2';

export type RendererBackend = 'webgpu' | 'webgl2';

export interface RendererChoice {
  readonly renderer: SplatRenderer;
  readonly backend: RendererBackend;
  /** WebGPU を試したが使えなかった場合の理由。UI に出す。 */
  readonly fallbackReason?: string;
}

export interface CreateRendererOptions {
  canvas: HTMLCanvasElement;
  /** すでに取得済みの WebGPU デバイスがあれば渡す。推論と共有するため。 */
  device?: GPUDevice;
  /** 強制的にバックエンドを指定する。PoC で両方を比べるのに使う。 */
  force?: RendererBackend;
}

/**
 * 使えるレンダラを1つ作る。
 *
 * WebGPU の判定は「アダプタが取れるか」まで見る。`navigator.gpu` があっても
 * アダプタが返らない端末が現実にある（PoC-1 の Android 10 実機がそうだった）。
 */
export async function createRenderer(opts: CreateRendererOptions): Promise<RendererChoice> {
  const wantWebgpu = opts.force !== 'webgl2';
  let reason: string | undefined;

  if (wantWebgpu) {
    try {
      let device = opts.device;
      if (!device) {
        if (!('gpu' in navigator) || !navigator.gpu) throw new Error('navigator.gpu がありません');
        const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
        if (!adapter) throw new Error('アダプタを取得できませんでした');
        device = await adapter.requestDevice();
      }
      return {
        renderer: new WgslSplatRenderer({ device, canvas: opts.canvas }),
        backend: 'webgpu',
      };
    } catch (e) {
      reason = e instanceof Error ? e.message : String(e);
      if (opts.force === 'webgpu') throw new Error(`WebGPU レンダラを作れませんでした: ${reason}`);
    }
  }

  if (!Webgl2SplatRenderer.isSupported()) {
    throw new Error(
      `描画できません。WebGPU は使えず（${reason ?? '未試行'}）、WebGL2 もありません。`,
    );
  }
  return {
    renderer: new Webgl2SplatRenderer({ canvas: opts.canvas }),
    backend: 'webgl2',
    ...(reason ? { fallbackReason: reason } : {}),
  };
}
