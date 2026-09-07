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
  timeout: 10 * 60 * 1000, // ソフトウェア実装は実機の20〜50倍遅いので長めに取る
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
    command: 'npm run preview -- --port 4173 --strictPort',
    url: 'http://127.0.0.1:4173',
    reuseExistingServer: !process.env.CI,
    timeout: 120_000,
  },
});
