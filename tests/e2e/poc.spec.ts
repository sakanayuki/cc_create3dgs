/**
 * PoC-1 ハーネスと本体シェルの E2E。
 *
 * WebGPU が使えるかは実行環境に依存する（開発コンテナの headless Chromium では
 * `navigator.gpu` が露出しなかった）。そこで**能力検出で分岐**し、
 * 使えない場合は「未検証」として記録する。黙って通過させない。
 */
import { expect, test } from '@playwright/test';
import { mkdirSync, writeFileSync } from 'node:fs';
import { dirname } from 'node:path';

const RESULTS = 'tests/results/metrics.json';
const metrics: Record<string, unknown> = { webgpuAvailable: null };

function save(): void {
  mkdirSync(dirname(RESULTS), { recursive: true });
  writeFileSync(RESULTS, JSON.stringify(metrics, null, 2), 'utf8');
}

test.describe('本体シェル', () => {
  test('トップページが表示され、端末の能力判定が走る', async ({ page }) => {
    const errors: string[] = [];
    page.on('pageerror', (e) => errors.push(e.message));

    await page.goto('/');
    await expect(page.locator('h1')).toContainText('PhotoSplat');
    await expect(page.locator('a.btn')).toContainText('PoC-1');

    // main.ts が能力判定を終えて注記を足すのを待つ
    await expect(page.locator('.card.accent p.mono')).toContainText('この端末:', { timeout: 30_000 });
    expect(errors, `ページ内で例外が起きました: ${errors.join(' / ')}`).toEqual([]);
  });
});

test.describe('PoC-1 ハーネス', () => {
  test('能力カードとモデル一覧が描画される', async ({ page }) => {
    const errors: string[] = [];
    page.on('pageerror', (e) => errors.push(e.message));

    await page.goto('/poc.html');
    await expect(page.locator('h1')).toContainText('PoC-1');

    const pill = page.locator('#cap .pill');
    await expect(pill).toBeVisible({ timeout: 30_000 });
    const capText = await pill.textContent();
    metrics['webgpuAvailable'] = capText?.includes('利用可') ?? false;

    const items = page.locator('#models .item');
    await expect(items).toHaveCount(7);
    await expect(items.first()).toContainText('Depth Anything 3 Small');

    await expect(page.locator('#out')).toContainText('WebGPU');
    expect(errors, `ページ内で例外が起きました: ${errors.join(' / ')}`).toEqual([]);
    save();
  });

  test('描画ベンチ（WebGPU があるときのみ実測）', async ({ page }) => {
    // CI はソフトウェア実装で実機の20〜50倍遅い。ここで見るのは「壊れていないこと」だけで、
    // 速度は見ない（docs/07 §7.4）。実機の数値は poc.html を直接開いて測る。
    await page.goto('/poc.html?frames=8&counts=20000,424000');
    await expect(page.locator('#cap .pill')).toBeVisible({ timeout: 30_000 });

    const hasWebGpu = await page.evaluate(async () => {
      if (!('gpu' in navigator) || !navigator.gpu) return false;
      const a = await navigator.gpu.requestAdapter().catch(() => null);
      return !!a;
    });
    metrics['webgpuAvailable'] = hasWebGpu;

    if (!hasWebGpu) {
      // 黙って通過させず、未検証であることを記録に残す
      metrics['renderBench'] = 'skipped: WebGPU 未提供';
      save();
      test.info().annotations.push({
        type: 'warning',
        description:
          'WebGPU が無いため描画ベンチ（R12）を検証できませんでした。実機の poc.html で確認してください。',
      });
      test.skip(true, 'WebGPU が利用できない環境です');
      return;
    }

    await page.locator('#runRender').click();
    await expect(page.locator('#renderOut .pill')).toHaveCount(2, { timeout: 8 * 60 * 1000 });

    const results = (await page.evaluate(() => {
      const pre = document.getElementById('out');
      return pre ? JSON.parse(pre.textContent ?? '{}') : {};
    })) as Record<string, unknown>;

    const bench = results['描画ベンチ'] as { splatCount: number; frameMs: number; drawnSplats: number }[];
    expect(bench.length).toBe(2);

    const std = bench.find((b) => b.splatCount === 424_000);
    expect(std, '標準プリセット（42万）の結果がありません').toBeTruthy();
    // 背面カリングが効いていること。全部描いていたら実装が壊れている。
    expect(std!.drawnSplats).toBeGreaterThan(0);
    expect(std!.drawnSplats).toBeLessThan(424_000);

    metrics['renderBench'] = bench;
    metrics['gaussianCount'] = std!.splatCount;
    save();
  });
});
