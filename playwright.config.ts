import { defineConfig, devices } from '@playwright/test';
import { existsSync } from 'node:fs';

/**
 * ブラウザの実体。CI では `playwright install` が入れたものを使うが、
 * 環境が Chromium を別の場所に持っている場合（開発コンテナなど）はそれを使う。
 */
const bundledChromium = process.env.PLAYWRIGHT_EXECUTABLE_PATH ?? '/opt/pw-browsers/chromium';
const executablePath = existsSync(bundledChromium) ? bundledChromium : undefined;

/**
 * E2E は「結果の正しさ」だけを見る。速度は見ない。
 * GitHub ランナーに GPU は無く、そこで測った時間は実機について何も語らないため
 * （docs/07 §7.4, §7.5）。
 */
export default defineConfig({
  testDir: './tests/e2e',
  outputDir: './tests/results/playwright',
  fullyParallel: false,
  workers: 1,
  // ソフトウェア実装は実機の20〜50倍遅いので長めに取る。
  // app.spec.ts は中で 8 分待つので、取り込みの時間ぶん余裕を足して 12 分にする。
  timeout: 12 * 60 * 1000,
  reporter: [['list'], ['json', { outputFile: 'tests/results/e2e.json' }]],
  use: {
    baseURL: 'http://127.0.0.1:4173',
    trace: 'retain-on-failure',
  },
  projects: [
    {
      name: 'chromium',
      use: {
        ...devices['Desktop Chrome'],
        launchOptions: {
          ...(executablePath ? { executablePath } : {}),
          args: [
            '--enable-unsafe-webgpu',
            '--enable-features=Vulkan,UseSkiaRenderer',
            '--use-angle=swiftshader',
            '--use-vulkan=swiftshader',
            '--disable-vulkan-surface',
          ],
        },
      },
    },
  ],
  webServer: {
    // 必ずビルドしてから配信する。dist が古いままだと、直したはずのコードを
    // 検証しないまま通ってしまう（実際に一度それで空振りした）。
    //
    // **`--host 127.0.0.1` を明示する。** これが無いと vite preview は
    // `localhost` に bind する。GitHub ランナーの `localhost` は `::1` と
    // `127.0.0.1` の両方に解決し、Node は先頭（多くは `::1`）を掴む。
    // すると下の `url`（127.0.0.1）には応答が返らず、
    // 「Timed out waiting 120000ms from config.webServer」で必ず落ちる。
    // 開発コンテナでは `localhost` が IPv4 だけに解決するので再現しない。
    // 待つ側と配信する側で名前が食い違わないよう、両方とも数字で書く。
    command: 'npm run build && npm run preview -- --host 127.0.0.1 --port 4173 --strictPort',
    url: 'http://127.0.0.1:4173',
    reuseExistingServer: !process.env.CI,
    // 冷えたランナーでは npm ci の直後にビルドが走る。手元では 12 秒だが余裕を持たせる。
    timeout: 180_000,
    // **既定では stdout が捨てられる。** 起動しなかったとき、ログに
    // 「タイムアウトした」以外の手がかりが何も残らない。実際に一度それで
    // 原因の切り分けができなかったので、両方とも拾う。
    stdout: 'pipe',
    stderr: 'pipe',
  },
});
