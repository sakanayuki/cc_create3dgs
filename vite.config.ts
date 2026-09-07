/// <reference types="vitest/config" />
import { defineConfig, type Plugin } from 'vite';
import { resolve } from 'node:path';

/**
 * ONNX Runtime の wasm がバンドルに二重に入るのを防ぐ。
 *
 * ort の ESM は wasm を `new URL(...)` で参照するので Vite が assets/ に複製する。
 * だが本プロジェクトは scripts/copy_ort.mjs で public/ort/ に置き、
 * `ort.env.wasm.wasmPaths` でそちらを指している（決定 D7: 同一オリジン配信）。
 * 放置すると 27.8 MB がそのまま無駄になるので、バンドル側を落とす。
 */
function dropBundledWasm(): Plugin {
  return {
    name: 'photosplat:drop-bundled-wasm',
    generateBundle(_options, bundle) {
      for (const name of Object.keys(bundle)) {
        if (name.endsWith('.wasm')) {
          this.warn(`重複する wasm をバンドルから除外しました: ${name}（public/ort/ から配信）`);
          delete bundle[name];
        }
      }
    },
  };
}

// GitHub Pages は https://<user>.github.io/<repo>/ に配信されるため base が要る。
// ローカル開発とテストでは '/' を使う。
const base = process.env.PAGES_BASE ?? '/';

export default defineConfig({
  base,
  plugins: [dropBundledWasm()],
  build: {
    target: 'es2022',
    // モデルは public/models/ に CI が置く。Vite はそれを dist/models/ にコピーする。
    assetsInlineLimit: 4096,
    rollupOptions: {
      input: {
        main: resolve(__dirname, 'index.html'),
        // PoC-1 実機検証ハーネス。本体とは独立に配信する。
        poc: resolve(__dirname, 'poc.html'),
      },
    },
  },
  worker: { format: 'es' },
  // .wgsl をソースコードとして読み込む
  assetsInclude: ['**/*.wgsl'],
  test: {
    // E2E は Playwright が担当するので Vitest からは外す
    include: ['tests/unit/**/*.test.ts'],
  },
});
