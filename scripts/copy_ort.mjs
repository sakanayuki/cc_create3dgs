// ONNX Runtime Web の wasm 一式を public/ort/ に複製する。
//
// GitHub Pages は同一オリジン配信（決定 D7）なので、ORT の wasm を CDN から
// 引かずに自分で配る。COOP/COEP を返せない環境ではスレッドが無効になるが、
// "threaded" ビルドは crossOriginIsolated が false のとき単スレッドで動く。
import { copyFile, mkdir, readdir } from 'node:fs/promises';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = dirname(dirname(fileURLToPath(import.meta.url)));
const src = join(root, 'node_modules', 'onnxruntime-web', 'dist');
const dst = join(root, 'public', 'ort');

// jsep = WebGPU EP 付き。素の wasm は WASM フォールバック用（docs/08 §8.2 の3段縮退）。
const WANTED = [
  'ort-wasm-simd-threaded.jsep.wasm',
  'ort-wasm-simd-threaded.jsep.mjs',
  'ort-wasm-simd-threaded.wasm',
  'ort-wasm-simd-threaded.mjs',
];

await mkdir(dst, { recursive: true });
const available = new Set(await readdir(src));
let copied = 0;
for (const name of WANTED) {
  if (!available.has(name)) {
    console.warn(`[copy_ort] 見つかりません: ${name}`);
    continue;
  }
  await copyFile(join(src, name), join(dst, name));
  copied++;
}
console.log(`[copy_ort] ${copied}/${WANTED.length} 個を public/ort/ に複製しました`);
