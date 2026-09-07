/**
 * モデルの所在解決（決定 D7, D22）。
 *
 * 実測に基づく分岐なので、ここを間違えると「動くけれど遅い」という
 * 気付きにくい形で出る。PoC-1 では WASM の q4f16 が uint8 の 1.5 倍
 * 遅かった（19.8s 対 12.9s）。
 */
import { beforeEach, describe, expect, it } from 'vitest';
import {
  fileForBackend,
  loadManifest,
  manifestBackendKey,
  resetManifestCache,
  resolveModel,
  totalBytesFor,
  type Manifest,
  type ManifestEntry,
} from '../../src/runtime/modelCatalog';

const entry = (over: Partial<ManifestEntry> = {}): ManifestEntry => ({
  id: 'depth-anything-v3-small',
  role: 'depth',
  byBackend: { webgpu: 'depth.q4f16.onnx', wasm: 'depth.uint8.onnx' },
  variants: [
    { mode: 'q4f16', file: 'depth.q4f16.onnx', bytes: 22_000_000 },
    { mode: 'uint8', file: 'depth.uint8.onnx', bytes: 27_000_000 },
  ],
  ...over,
});

describe('バックエンド別のファイル選択', () => {
  it('WebGPU は q4f16、WASM は uint8', () => {
    expect(fileForBackend(entry(), 'webgpu')).toBe('depth.q4f16.onnx');
    expect(fileForBackend(entry(), 'wasm')).toBe('depth.uint8.onnx');
  });

  it('片方しか無ければそれを使う（動かないよりまし）', () => {
    const e = entry({ byBackend: { wasm: 'only.uint8.onnx' } });
    expect(fileForBackend(e, 'webgpu')).toBe('only.uint8.onnx');
  });

  it('旧形式（byBackend なし）でも代表ファイルに落ちる', () => {
    const base = entry({ file: 'legacy.onnx' });
    const { byBackend: _omit, ...legacy } = base;
    expect(fileForBackend(legacy, 'webgpu')).toBe('legacy.onnx');
  });

  it('エラー項目は選ばない', () => {
    const e = entry({ error: '元モデルが見つかりません' });
    expect(fileForBackend(e, 'webgpu')).toBeNull();
  });
});

describe('取得元の優先順', () => {
  beforeEach(() => resetManifestCache());

  const manifest: Manifest = { models: [entry()] };
  const okFetch = (async () =>
    new Response(JSON.stringify(manifest), { status: 200 })) as unknown as typeof fetch;
  const missingFetch = (async () => new Response('', { status: 404 })) as unknown as typeof fetch;

  it('同一オリジンを先に、HuggingFace を後に置く', async () => {
    await loadManifest(okFetch);
    const sources = await resolveModel('depth-anything-v3-small', 'wasm', {
      repo: 'onnx-community/depth-anything-v3-small',
      file: 'onnx/model.onnx',
      extra: 'onnx/model.onnx_data',
    });
    expect(sources).toHaveLength(2);
    expect(sources[0]!.url).toContain('models/depth.uint8.onnx');
    expect(sources[0]!.url).not.toContain('huggingface');
    expect(sources[1]!.url).toContain('huggingface.co');
    // 外部データも一緒に渡る（DA3 は model.onnx_data を伴う）
    expect(sources[1]!.externalData?.path).toBe('model.onnx_data');
  });

  it('マニフェストが無ければ HuggingFace だけ', async () => {
    await loadManifest(missingFetch);
    const sources = await resolveModel('modnet', 'webgpu', {
      repo: 'Xenova/modnet',
      file: 'onnx/model_uint8.onnx',
    });
    expect(sources).toHaveLength(1);
    expect(sources[0]!.url).toContain('huggingface.co');
  });

  it('マニフェストは1度しか取りに行かない', async () => {
    let calls = 0;
    const counting = (async () => {
      calls++;
      return new Response(JSON.stringify(manifest), { status: 200 });
    }) as unknown as typeof fetch;
    await loadManifest(counting);
    await loadManifest(counting);
    await loadManifest(counting);
    expect(calls).toBe(1);
  });
});

describe('ダウンロード量', () => {
  it('バックエンドに応じた実際の量を数える', () => {
    const m: Manifest = { models: [entry()] };
    expect(totalBytesFor(m, 'webgpu', ['depth-anything-v3-small'])).toBe(22_000_000);
    expect(totalBytesFor(m, 'wasm', ['depth-anything-v3-small'])).toBe(27_000_000);
  });

  it('同じファイルを共有するモデルを二重に数えない', () => {
    // CNN 系は両バックエンドとも uint8 なので、ファイルは1つしか無い
    const shared: ManifestEntry = {
      id: 'modnet',
      role: 'matte',
      byBackend: { webgpu: 'modnet.uint8.onnx', wasm: 'modnet.uint8.onnx' },
      variants: [{ mode: 'uint8', file: 'modnet.uint8.onnx', bytes: 6_600_000 }],
    };
    const m: Manifest = { models: [shared] };
    expect(totalBytesFor(m, 'webgpu', ['modnet', 'modnet'])).toBe(6_600_000);
  });
});

describe('shader-f16 が無い WebGPU', () => {
  it('q4f16 を避けて uint8 側を選ぶ', () => {
    // q4f16 は shader-f16 拡張が要る。無い実装では ORT の f16 シェーダが
    // コンパイルに失敗するが例外にならず、出力が壊れたまま「成功」する。
    // 実測では DA3 の intrinsics が全て 0 で返った（CPU では fx=887.5）。
    expect(manifestBackendKey('webgpu', true)).toBe('webgpu');
    expect(manifestBackendKey('webgpu', false)).toBe('wasm');
  });

  it('WASM 経路は影響を受けない', () => {
    expect(manifestBackendKey('wasm', true)).toBe('wasm');
    expect(manifestBackendKey('wasm', false)).toBe('wasm');
  });

  it('実際に uint8 のファイルが返る', async () => {
    resetManifestCache();
    const manifest: Manifest = { models: [entry()] };
    await loadManifest((async () =>
      new Response(JSON.stringify(manifest), { status: 200 })) as unknown as typeof fetch);

    const hf = { repo: 'onnx-community/depth-anything-v3-small', file: 'onnx/model.onnx' };
    const withF16 = await resolveModel('depth-anything-v3-small', 'webgpu', hf, true);
    const noF16 = await resolveModel('depth-anything-v3-small', 'webgpu', hf, false);
    expect(withF16[0]!.url).toContain('depth.q4f16.onnx');
    expect(noF16[0]!.url).toContain('depth.uint8.onnx');
  });
});
