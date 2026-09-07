/**
 * モデルの所在を解決する（決定 D7, D22）。
 *
 * CI が量子化したものを同一オリジンに置く（`models/manifest.json`）。
 * そこに無ければ HuggingFace から直接取る。前者が本番の経路で、後者は
 * モデルをまだ焼き込んでいない環境（開発中の Pages など）のための保険。
 *
 * 量子化方式は実行プロバイダで変える。PoC-1 の実測で、WASM では
 * q4f16 が最遅（19.8s）・uint8 が最速（12.9s）だった。MatMulNBits に
 * WASM の速い経路が無いため。WebGPU では逆に q4f16 が小さくて速い。
 */
import type { Backend, ModelSource } from './OrtSession';

export interface ManifestVariant {
  readonly mode: string;
  readonly file: string;
  readonly bytes?: number;
  readonly sha256?: string;
}

export interface ManifestEntry {
  readonly id: string;
  readonly role: 'depth' | 'matte' | 'inpaint';
  readonly inputSize?: readonly number[];
  /** バックエンド名 → ファイル名。CI が書く（決定 D22）。 */
  readonly byBackend?: Readonly<Record<string, string>>;
  readonly variants?: readonly ManifestVariant[];
  /** 旧形式との互換。byBackend が無いときに使う。 */
  readonly file?: string;
  readonly mode?: string;
  readonly bytes?: number;
  readonly error?: string;
}

export interface Manifest {
  readonly profile?: string;
  readonly backends?: readonly string[];
  readonly models: readonly ManifestEntry[];
}

/**
 * マニフェストのどのバックエンド欄を引くか決める。
 *
 * q4f16 は WebGPU の **`shader-f16` 拡張**を必要とする。この拡張が無い
 * 実装では ORT の f16 シェーダがコンパイルに失敗するが、**例外にならない**。
 * 出力が壊れたまま推論が「成功」してしまう。実測では Depth Anything 3 の
 * `intrinsics` が全て 0 で返り（CPU では fx=887.5）、焦点距離が取れないまま
 * 画角 55° の仮定に落ちていた。深度そのものも当てにならない。
 *
 * したがって、`shader-f16` が無い WebGPU では q4f16 を選ばない。
 * uint8 側（WASM 用に置いてあるもの）を使う。実行プロバイダは WebGPU の
 * ままでよく、変えるのは「どのファイルを落とすか」だけである。
 */
export function manifestBackendKey(backend: Backend, shaderF16: boolean): string {
  return backend === 'webgpu' && !shaderF16 ? 'wasm' : backend;
}

/** HuggingFace の直リンク。同一オリジンに無いときの保険。 */
export interface HfFallback {
  readonly repo: string;
  readonly file: string;
  /** 外部データ（model.onnx_data）。 */
  readonly extra?: string;
}

const HF = 'https://huggingface.co';
const hfUrl = (repo: string, path: string): string => `${HF}/${repo}/resolve/main/${path}`;

let cached: Manifest | null | undefined;

function baseUrl(): string {
  return import.meta.env.BASE_URL ?? '/';
}

/**
 * マニフェストを1度だけ読む。
 *
 * 無い場合（まだモデルを焼いていない Pages など）は null を覚えて、
 * 以降は取りに行かない。生成のたびに 404 を1回踏むのは無駄。
 */
export async function loadManifest(fetchImpl: typeof fetch = fetch): Promise<Manifest | null> {
  if (cached !== undefined) return cached;
  try {
    const res = await fetchImpl(`${baseUrl()}models/manifest.json`, { cache: 'no-cache' });
    cached = res.ok ? ((await res.json()) as Manifest) : null;
  } catch {
    cached = null;
  }
  return cached;
}

/** テストのために覚えた内容を捨てる。 */
export function resetManifestCache(): void {
  cached = undefined;
}

/**
 * マニフェストの1項目から、そのバックエンド向けのファイル名を選ぶ。
 *
 * byBackend が無い（旧形式）ときは代表ファイルに落とす。選べなければ null。
 */
export function fileForBackend(entry: ManifestEntry, backend: Backend): string | null {
  if (entry.error) return null;
  const byBackend = entry.byBackend;
  if (byBackend) {
    const direct = byBackend[backend];
    if (direct) return direct;
    // 指定のバックエンドが無ければ、他のバックエンド向けでも動きはする。
    const any = Object.values(byBackend)[0];
    if (any) return any;
  }
  return entry.file ?? null;
}

/**
 * 使うモデルの取得元を、優先順に返す。
 *
 * 先頭から順に試して、最初に読めたものを使う（`OrtSession.loadWithLadder`
 * と同じ考え方）。同一オリジンを先に置くのは決定 D7。
 */
export async function resolveModel(
  id: string,
  backend: Backend,
  fallback: HfFallback,
  /** WebGPU に `shader-f16` があるか。無ければ q4f16 を避ける。 */
  shaderF16 = true,
): Promise<ModelSource[]> {
  const out: ModelSource[] = [];
  const manifest = await loadManifest();
  const entry = manifest?.models.find((m) => m.id === id);
  const key = manifestBackendKey(backend, shaderF16);
  const file = entry ? fileForBackend(entry, key as Backend) : null;
  if (file) out.push({ id, url: `${baseUrl()}models/${file}` });

  out.push({
    id,
    url: hfUrl(fallback.repo, fallback.file),
    ...(fallback.extra
      ? {
          externalData: {
            url: hfUrl(fallback.repo, fallback.extra),
            path: fallback.extra.split('/').pop() ?? fallback.extra,
          },
        }
      : {}),
  });
  return out;
}

/** マニフェストが示す、そのバックエンドでの総ダウンロード量（バイト）。 */
export function totalBytesFor(manifest: Manifest, backend: Backend, ids: readonly string[]): number {
  const files = new Map<string, number>();
  for (const id of ids) {
    const entry = manifest.models.find((m) => m.id === id);
    if (!entry) continue;
    const file = fileForBackend(entry, backend);
    if (!file) continue;
    const variant = entry.variants?.find((v) => v.file === file);
    files.set(file, variant?.bytes ?? entry.bytes ?? 0);
  }
  let total = 0;
  for (const b of files.values()) total += b;
  return total;
}
