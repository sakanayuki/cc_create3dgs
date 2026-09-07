/**
 * 端末の能力判定と段階的縮退（docs/06 §6.8, docs/08 §8.2）。
 *
 * 決定 D5 で WebGPU を必須としたが、必須の意味は「WebGPU が無ければ生成できない」
 * であって「あれば必ず目標構成で動く」ではない。アダプタの limits によっては
 * 作業グリッドを落とす必要があるし、ORT の WebGPU EP が使えないこともある。
 */

/** 描画の重さで端末を3段に分ける。起動時ベンチの結果で決まる（リスク R12）。 */
export type DeviceTier = 'high' | 'mid' | 'low';

export interface AdapterReport {
  readonly supported: boolean;
  readonly reason?: string;
  readonly vendor?: string;
  readonly architecture?: string;
  readonly limits?: Record<string, number>;
  readonly features?: string[];
}

export interface Capability {
  readonly webgpu: AdapterReport;
  /** 作業グリッド。limits が足りなければ 1024 から落とす（決定 D17）。 */
  readonly workingGrid: 1024 | 768 | 512;
  /** SharedArrayBuffer が使えるか。GitHub Pages では常に false（COOP/COEP を返せない）。 */
  readonly crossOriginIsolated: boolean;
  /** WASM フォールバック時に使えるスレッド数。 */
  readonly wasmThreads: number;
  /**
   * WebGPU に `shader-f16` があるか。
   *
   * 無い実装では q4f16 のモデルが**黙って壊れる**（決定 D22 の補足、
   * src/runtime/modelCatalog.ts の manifestBackendKey）。
   */
  readonly shaderF16: boolean;
}

/** 本設計が必要とする limits の下限。これを割ると作業グリッドを落とす。 */
const REQUIRED_LIMITS = {
  maxStorageBufferBindingSize: 128 * 1024 * 1024,
  maxBufferSize: 256 * 1024 * 1024,
  maxTextureDimension2D: 2048,
  maxComputeWorkgroupStorageSize: 16384,
} as const;

function limitsOf(adapter: GPUAdapter): Record<string, number> {
  const out: Record<string, number> = {};
  // GPUSupportedLimits は列挙できないので、見たいものだけ拾う
  const keys = [
    'maxTextureDimension2D',
    'maxBufferSize',
    'maxStorageBufferBindingSize',
    'maxUniformBufferBindingSize',
    'maxComputeWorkgroupStorageSize',
    'maxComputeInvocationsPerWorkgroup',
    'maxComputeWorkgroupSizeX',
    'maxStorageBuffersPerShaderStage',
    'maxBindGroups',
  ];
  for (const k of keys) {
    const v = (adapter.limits as unknown as Record<string, number | undefined>)[k];
    if (typeof v === 'number') out[k] = v;
  }
  return out;
}

/** limits から妥当な作業グリッドを選ぶ。 */
export function gridForLimits(limits: Record<string, number> | undefined): 1024 | 768 | 512 {
  if (!limits) return 512;
  const tex = limits['maxTextureDimension2D'] ?? 0;
  const storage = limits['maxStorageBufferBindingSize'] ?? 0;
  if (tex >= 4096 && storage >= REQUIRED_LIMITS.maxStorageBufferBindingSize) return 1024;
  if (tex >= 2048 && storage >= 64 * 1024 * 1024) return 768;
  return 512;
}

export async function detectCapability(): Promise<Capability> {
  const isolated = typeof crossOriginIsolated !== 'undefined' && crossOriginIsolated;
  // GitHub Pages は COOP/COEP を返せないので SharedArrayBuffer が無く、
  // WASM は単スレッドに制限される。これが D5「WebGPU 必須」の技術的根拠。
  const wasmThreads = isolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;

  if (!('gpu' in navigator) || !navigator.gpu) {
    return {
      webgpu: { supported: false, reason: 'navigator.gpu がありません' },
      workingGrid: 512,
      crossOriginIsolated: isolated,
      wasmThreads,
      shaderF16: false,
    };
  }

  let adapter: GPUAdapter | null = null;
  try {
    adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  } catch (e) {
    return {
      webgpu: { supported: false, reason: `requestAdapter が失敗: ${String(e)}` },
      workingGrid: 512,
      crossOriginIsolated: isolated,
      wasmThreads,
      shaderF16: false,
    };
  }
  if (!adapter) {
    return {
      webgpu: { supported: false, reason: 'アダプタを取得できませんでした' },
      workingGrid: 512,
      crossOriginIsolated: isolated,
      wasmThreads,
      shaderF16: false,
    };
  }

  const limits = limitsOf(adapter);
  const info = (adapter as GPUAdapter & { info?: GPUAdapterInfo }).info;
  const report: AdapterReport = {
    supported: true,
    ...(info?.vendor ? { vendor: info.vendor } : {}),
    ...(info?.architecture ? { architecture: info.architecture } : {}),
    limits,
    features: [...adapter.features],
  };

  return {
    webgpu: report,
    workingGrid: gridForLimits(limits),
    crossOriginIsolated: isolated,
    wasmThreads,
    // q4f16 のモデルを使ってよいかの判断材料。無いと黙って壊れる。
    shaderF16: adapter.features.has('shader-f16'),
  };
}

/** 判定結果を人が読める日本語にする。UI とベンチ結果の両方で使う。 */
export function describeCapability(cap: Capability): string[] {
  const lines: string[] = [];
  if (!cap.webgpu.supported) {
    lines.push(`WebGPU: 利用できません（${cap.webgpu.reason ?? '理由不明'}）`);
  } else {
    const l = cap.webgpu.limits ?? {};
    lines.push(`WebGPU: 利用できます${cap.webgpu.vendor ? ` — ${cap.webgpu.vendor} ${cap.webgpu.architecture ?? ''}`.trimEnd() : ''}`);
    lines.push(`  maxTextureDimension2D: ${l['maxTextureDimension2D'] ?? '?'}`);
    lines.push(`  maxStorageBufferBindingSize: ${fmtBytes(l['maxStorageBufferBindingSize'])}`);
    lines.push(`  maxBufferSize: ${fmtBytes(l['maxBufferSize'])}`);
  }
  lines.push(`作業グリッド: ${cap.workingGrid}²`);
  lines.push(`crossOriginIsolated: ${cap.crossOriginIsolated}（WASM スレッド数 ${cap.wasmThreads}）`);
  return lines;
}

function fmtBytes(n: number | undefined): string {
  if (typeof n !== 'number') return '?';
  return `${(n / 1024 / 1024).toFixed(0)} MB`;
}
