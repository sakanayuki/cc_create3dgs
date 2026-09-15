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

/**
 * 「できました」の画面にも進捗を置く（docs/12 §12.16.3）。
 *
 * 1枚目の下書きを見せた時点で画面は `#view` へ移る。進捗バーが `#work` にしか
 * 無かったので、残り2枚の生成と位置合わせのあいだ**進捗がどこにも出なかった**。
 * 実機で「正面写真の出力までで完了してしまった」と報告された。
 *
 * ここで生成は走らせない（モデルの取得が要る）。**置き場所があること**だけを見る。
 */
test('「できました」の画面にも進捗の置き場所がある', async ({ page }) => {
  await page.goto('/');

  // #work の中ではなく #view の中にあること。ここが要点。
  await expect(page.locator('#view #viewProgress')).toHaveCount(1);
  await expect(page.locator('#view #viewProgress .bar#bar2')).toHaveCount(1);
  await expect(page.locator('#view #stage2')).toHaveCount(1);

  // 走っていないあいだは畳んである。
  await expect(page.locator('#viewProgress')).toBeHidden();
});
