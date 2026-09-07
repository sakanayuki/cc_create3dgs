/**
 * パイプラインで作ったスプラットが、立体として描けるかを見る E2E。
 *
 * 単体テストは各段を別々に検算するが、座標系の取り違えのように
 * 「繋いでみて初めて分かる」不具合はそれでは出ない。ここでは
 * 深度較正 → 法線推定 → 適応サンプリング → シェル組み立て → 描画
 * を通しで走らせ、画面に出た絵を読み戻して確かめる。
 */
import { expect, test, type Page } from '@playwright/test';
import { mkdirSync, writeFileSync } from 'node:fs';
import { dirname } from 'node:path';

const RESULTS = 'tests/results/pipeline.json';
const results: Record<string, unknown> = {};

function save(): void {
  mkdirSync(dirname(RESULTS), { recursive: true });
  writeFileSync(RESULTS, JSON.stringify(results, null, 2), 'utf8');
}

interface Shot {
  backend: string;
  count: number;
  frontCount: number;
  drawn: number;
  litPixels: number;
  coverage: number;
  centroid: [number, number] | null;
  bbox: [number, number, number, number] | null;
  meanColor: [number, number, number] | null;
}

async function shoot(page: Page, yaw: number): Promise<Shot> {
  return page.evaluate(async (y: number) => {
    const api = window.__photosplat;
    if (!api) throw new Error('__photosplat がページに露出していません');
    const built = api.buildPipelineSplats(192, 0.3);
    const size = 256;
    const canvas = document.createElement('canvas');
    canvas.width = size;
    canvas.height = size;
    document.body.appendChild(canvas);
    try {
      const { renderer, backend } = await api.createRenderer({ canvas });
      try {
        renderer.resize(size, size);
        renderer.setSplats(built.data, built.count);
        renderer.setDepthRange(built.nearZ, built.farZ);
        renderer.setCamera({ yaw: y, pitch: 0.1, distance: 1.0, target: [0, 0, 0] });
        for (let i = 0; i < 2; i++) {
          renderer.render();
          await renderer.flush();
        }
        renderer.render();
        const px = await renderer.readPixels();

        let lit = 0;
        let sx = 0;
        let sy = 0;
        let sr = 0;
        let sg = 0;
        let sb = 0;
        let x0 = size;
        let y0 = size;
        let x1 = -1;
        let y1 = -1;
        for (let yy = 0; yy < size; yy++) {
          for (let xx = 0; xx < size; xx++) {
            const i = (yy * size + xx) * 4;
            const a = px[i + 3] as number;
            const lum = (px[i] as number) + (px[i + 1] as number) + (px[i + 2] as number);
            if (a <= 8 && lum <= 12) continue;
            lit++;
            sx += xx;
            sy += yy;
            sr += px[i] as number;
            sg += px[i + 1] as number;
            sb += px[i + 2] as number;
            if (xx < x0) x0 = xx;
            if (yy < y0) y0 = yy;
            if (xx > x1) x1 = xx;
            if (yy > y1) y1 = yy;
          }
        }
        const n = Math.max(lit, 1);
        return {
          backend,
          count: built.count,
          frontCount: built.frontCount,
          drawn: renderer.stats.drawnSplats,
          litPixels: lit,
          coverage: Math.round((lit / (size * size)) * 10_000) / 10_000,
          centroid: lit > 0 ? [sx / n / size, sy / n / size] : null,
          bbox: lit > 0 ? [x0 / size, y0 / size, (x1 + 1) / size, (y1 + 1) / size] : null,
          meanColor: lit > 0 ? [Math.round(sr / n), Math.round(sg / n), Math.round(sb / n)] : null,
        } as unknown as Shot;
      } finally {
        renderer.dispose();
      }
    } finally {
      canvas.remove();
    }
  }, yaw);
}

test.describe('パイプライン通しの描画', () => {
  test('生成したスプラットが立体として描ける', async ({ page }) => {
    const errors: string[] = [];
    page.on('pageerror', (e) => errors.push(e.message));
    await page.goto('/poc.html');
    await expect(page.locator('#cap .pill')).toBeVisible({ timeout: 30_000 });

    const front = await shoot(page, 0);
    results['正面'] = front;
    save();

    expect(front.count, 'スプラットが1個も作られていません').toBeGreaterThan(1000);
    expect(front.frontCount).toBeGreaterThan(0);
    expect(front.count, '背面シェルが作られていません').toBeGreaterThan(front.frontCount);

    // 正面から見たら背面シェルは1個も描かれないはず。ここが崩れるときは
    // 座標系（y と z の反転）か法線の向きを取り違えている。
    expect(
      front.drawn,
      `正面なのに背面シェルが描かれています（描画 ${front.drawn} / 前面 ${front.frontCount}）`,
    ).toBeLessThanOrEqual(front.frontCount);
    expect(front.drawn / front.frontCount, '前面シェルがほとんど描かれていません').toBeGreaterThan(0.9);

    // 被写体は縦長の楕円体（横 0.30 / 縦 0.40）。その形が画面に出ているか。
    expect(front.litPixels).toBeGreaterThan(5000);
    const [bx0, by0, bx1, by1] = front.bbox ?? [0, 0, 0, 0];
    const w = bx1 - bx0;
    const h = by1 - by0;
    expect(h / w, `縦横比 ${(h / w).toFixed(2)}（縦長のはず）`).toBeGreaterThan(1.1);
    expect(h / w).toBeLessThan(1.6);

    const [cx, cy] = front.centroid ?? [0, 0];
    expect(cx).toBeGreaterThan(0.4);
    expect(cx).toBeLessThan(0.6);
    expect(cy).toBeGreaterThan(0.35);
    expect(cy).toBeLessThan(0.65);

    expect(errors, `ページ内で例外が起きました: ${errors.join(' / ')}`).toEqual([]);
  });

  test('視点を回すとシルエットが変わる（板ではない）', async ({ page }) => {
    await page.goto('/poc.html');
    await expect(page.locator('#cap .pill')).toBeVisible({ timeout: 30_000 });

    const front = await shoot(page, 0);
    const side = await shoot(page, 0.5);
    results['斜め'] = side;
    save();

    expect(side.litPixels).toBeGreaterThan(1000);

    // 回すと投影幅が狭くなる。平らな板を貼っているだけならここは変わらない。
    const widthOf = (s: Shot): number => {
      const [x0, , x1] = s.bbox ?? [0, 0, 0, 0];
      return x1 - x0;
    };
    expect(widthOf(side), '回しても幅が変わりません').toBeLessThan(widthOf(front) * 0.98);

    // 斜めからは背面シェルの一部が見えるようになる
    expect(side.drawn).toBeGreaterThan(0);
  });
});
