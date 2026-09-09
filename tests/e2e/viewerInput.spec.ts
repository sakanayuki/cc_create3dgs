/**
 * プレビューを指（マウス）で回せるか、実ブラウザで確かめる。
 *
 * 角度の計算そのものは単体テストで見ているが、実機で「動かない」と
 * 言われたときに壊れていたのは、そこへ値を渡すまでの繋ぎだった。
 * 本物の DOM で、本物のポインタイベントを流して確かめる。
 */
import { expect, test, type Page } from '@playwright/test';

/** PoC ページに Viewer を1つ立てて、キャンバスの矩形を返す。 */
async function mountViewer(page: Page): Promise<{ x: number; y: number; w: number; h: number }> {
  await page.goto('/poc.html');
  await page.waitForFunction(() => Boolean(window.__photosplat));
  return page.evaluate(async () => {
    const api = window.__photosplat;
    if (!api) throw new Error('__photosplat がページに露出していません');
    const canvas = document.createElement('canvas');
    canvas.id = 'e2e-view';
    canvas.style.cssText =
      'position:fixed;left:20px;top:20px;width:320px;height:320px;touch-action:none;z-index:9999';
    document.body.appendChild(canvas);

    const viewer = new api.Viewer({ canvas, introAnimation: false });
    await viewer.init();
    const count = 4000;
    viewer.setSplats(api.makeProceduralSplats(count), count);
    (window as unknown as Record<string, unknown>)['__v'] = viewer;

    const r = canvas.getBoundingClientRect();
    return { x: r.x, y: r.y, w: r.width, h: r.height };
  });
}

const yawOf = (page: Page): Promise<number> =>
  page.evaluate(() => {
    const v = (window as unknown as Record<string, unknown>)['__v'] as { getView(): { yaw: number } };
    return v.getView().yaw;
  });

const pitchOf = (page: Page): Promise<number> =>
  page.evaluate(() => {
    const v = (window as unknown as Record<string, unknown>)['__v'] as {
      getView(): { pitch: number };
    };
    return v.getView().pitch;
  });

test('マウスの横ドラッグで回る', async ({ page }) => {
  const r = await mountViewer(page);
  expect(await yawOf(page)).toBeCloseTo(0, 6);

  const cx = r.x + r.w / 2;
  const cy = r.y + r.h / 2;
  await page.mouse.move(cx, cy);
  await page.mouse.down();
  // 少しずつ動かす。実際の操作と同じように、多数の pointermove が飛ぶ。
  for (let i = 1; i <= 12; i++) await page.mouse.move(cx + i * 8, cy);
  const dragging = await yawOf(page);
  await page.mouse.up();

  // 96px = キャンバス幅の 0.3。快適範囲の 1.2 倍ぶん回るはず（約 54°）だが、
  // ラバーバンドが掛かるので「充分に回った」ことだけを見る。
  expect(Math.abs((dragging * 180) / Math.PI)).toBeGreaterThan(20);
  // 離しても戻らない
  expect(await yawOf(page)).toBeCloseTo(dragging, 6);
});

test('縦ドラッグで上下に回る', async ({ page }) => {
  const r = await mountViewer(page);
  const cx = r.x + r.w / 2;
  const cy = r.y + r.h / 2;
  await page.mouse.move(cx, cy);
  await page.mouse.down();
  for (let i = 1; i <= 10; i++) await page.mouse.move(cx, cy + i * 6);
  const p = await pitchOf(page);
  await page.mouse.up();
  expect(Math.abs((p * 180) / Math.PI)).toBeGreaterThan(8);
  // 横は動いていない
  expect(Math.abs(await yawOf(page))).toBeLessThan(1e-6);
});

test('掴み直しても飛ばず、続きから回る', async ({ page }) => {
  const r = await mountViewer(page);
  const cx = r.x + r.w / 2;
  const cy = r.y + r.h / 2;

  await page.mouse.move(cx, cy);
  await page.mouse.down();
  await page.mouse.move(cx + 30, cy);
  await page.mouse.up();
  const first = await yawOf(page);
  expect(Math.abs(first)).toBeGreaterThan(1e-3);

  // 別の場所で掴んで、動かさずに離す → 変わらない
  await page.mouse.move(r.x + 20, r.y + 20);
  await page.mouse.down();
  await page.mouse.up();
  expect(await yawOf(page)).toBeCloseTo(first, 6);

  // そこから続きが始まる
  await page.mouse.move(cx, cy);
  await page.mouse.down();
  await page.mouse.move(cx + 30, cy);
  await page.mouse.up();
  expect(Math.abs(await yawOf(page))).toBeGreaterThan(Math.abs(first) * 1.5);
});

test('タッチのポインタでも回る', async ({ page }) => {
  const r = await mountViewer(page);
  // スマホのスワイプ相当。pointerType を touch にして流す。
  await page.evaluate(
    ({ x, y }) => {
      const el = document.getElementById('e2e-view') as HTMLCanvasElement;
      // setPointerCapture は合成イベントの pointerId でも動くよう、実装を差し替える
      el.setPointerCapture = (): void => {};
      const fire = (type: string, cx: number, cy: number): void => {
        el.dispatchEvent(
          new PointerEvent(type, {
            pointerId: 7,
            pointerType: 'touch',
            isPrimary: true,
            clientX: cx,
            clientY: cy,
            bubbles: true,
          }),
        );
      };
      fire('pointerdown', x, y);
      for (let i = 1; i <= 12; i++) fire('pointermove', x + i * 8, y);
      fire('pointerup', x + 96, y);
    },
    { x: r.x + r.w / 2, y: r.y + r.h / 2 },
  );
  expect(Math.abs(((await yawOf(page)) * 180) / Math.PI)).toBeGreaterThan(20);
});
