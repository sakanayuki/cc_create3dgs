/**
 * 複数枚モードの入口（docs/12 §12.16.1）。
 *
 * **「機能を作ったのに画面から辿り着けない」を二度とやらないための検査。**
 * 実際に、実装してブランチに push したまま PR を出し忘れ、公開サイトを見た
 * 依頼者に「どこにあるか分からない」と2度言わせた。
 *
 * ここでは生成は走らせない。モデルの取得が要り、CI では落とせないため
 * （docs/07 §7.4 と同じ事情）。**入口が出て、選べて、押せるところまで**を見る。
 */
import { expect, test } from '@playwright/test';

test('プリセットで複数枚モードに切り替えると、枠が3つ出る', async ({ page }) => {
  await page.goto('/');

  // 既定は1枚モード
  await expect(page.locator('#photo')).toBeVisible();
  await expect(page.locator('#multi-pick')).toBeHidden();

  await page.getByRole('radio', { name: /複数枚/ }).check();
  await expect(page.locator('#multi-pick')).toBeVisible();
  await expect(page.locator('#photo')).toBeHidden();
  for (const id of ['photo-front', 'photo-right', 'photo-left']) {
    await expect(page.locator(`#${id}`)).toBeVisible();
  }

  // 1枚モードへ戻せる
  await page.getByRole('radio', { name: '標準' }).check();
  await expect(page.locator('#photo')).toBeVisible();
  await expect(page.locator('#multi-pick')).toBeHidden();
});

test('2枚そろうまでボタンは押せない', async ({ page }) => {
  await page.goto('/');
  await page.getByRole('radio', { name: /複数枚/ }).check();

  const go = page.locator('#multi-go');
  await expect(go).toBeDisabled();
  await expect(go).toHaveText(/2枚以上/);

  await page.locator('#photo-front').setInputFiles(`${process.cwd()}/tests/test29.jpeg`);
  await expect(go).toBeDisabled();

  await page.locator('#photo-right').setInputFiles(`${process.cwd()}/tests/test29_right.jpeg`);
  await expect(go).toBeEnabled();
  await expect(go).toHaveText(/2枚から作る/);

  await page.locator('#photo-left').setInputFiles(`${process.cwd()}/tests/test29_left.jpeg`);
  await expect(go).toHaveText(/3枚から作る/);
});
