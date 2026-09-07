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

    // 最初は「選ぶ」画面。他の2つは隠れている。
    await expect(page.locator('#pick')).toBeVisible();
    await expect(page.locator('#work')).toBeHidden();
    await expect(page.locator('#view')).toBeHidden();
    await expect(page.locator('#photo')).toBeVisible();

    // 能力判定が終わって、どの経路で動くかが出る
    await expect(page.locator('#env')).toContainText('この端末:', { timeout: 30_000 });
    await expect(page.locator('#env')).not.toContainText('判定中', { timeout: 30_000 });
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
    // ソフトウェア実装では 42万サーフェルの1フレームに数十秒かかる。ここで見るのは
    // 「レンダラが動き、背面カリングが効いていること」だけなので規模を落とす。
    // 実機の 42万での fps は poc.html を直接開いて測る（R12）。
    await page.goto('/poc.html?frames=6&counts=20000,60000');
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

    // ピルの数ではなく結果 JSON を見る。失敗時にもピルが出るので数え間違えるため。
    const readBench = () =>
      page.evaluate(() => {
        const pre = document.getElementById('out');
        if (!pre) return null;
        try {
          return (JSON.parse(pre.textContent ?? '{}')['描画ベンチ'] ?? null) as
            | { splatCount: number; frameMs: number; drawnSplats: number }[]
            | null;
        } catch {
          return null;
        }
      });

    await expect
      .poll(async () => (await readBench())?.length ?? 0, { timeout: 8 * 60 * 1000, intervals: [2000] })
      .toBe(2);

    const bench = (await readBench()) ?? [];
    expect(bench.map((b) => b.splatCount)).toEqual([20_000, 60_000]);

    const big = bench.find((b) => b.splatCount === 60_000);
    expect(big, '60,000 個の結果がありません').toBeTruthy();

    const cullRatio = 1 - big!.drawnSplats / 60_000;
    // アサーションの前に記録する。失敗しても数値を残したいため。
    metrics['renderBench'] = bench;
    metrics['cullRatio'] = cullRatio;
    save();

    // カリングが動いていること。0% なら法線かカリングが壊れており、
    // 100% なら何も描けていない。ベンチの球殻は視錐台からはみ出すので、
    // 背面カリング（約50%）に視錐台カリングが上乗せされる。
    expect(big!.drawnSplats).toBeGreaterThan(0);
    expect(cullRatio).toBeGreaterThan(0.25);
    expect(cullRatio).toBeLessThan(0.98);
  });
});
