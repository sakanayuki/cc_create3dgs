/**
 * 描画バックエンドが「本当にピクセルを出しているか」を確かめる E2E。
 *
 * フレーム時間や drawnSplats はレンダラの自己申告なので、それだけでは
 * 「真っ黒を高速に描く」バグを見逃す。ここではキャンバスを読み戻し、
 * 手続き生成の球殻が画面中央に丸く出ていることまで確かめる。
 *
 * WebGL2 は D19 のフォールバック経路で、実機（WebGPU の無い Android）では
 * こちらしか動かない。CI の SwiftShader は遅いが、正しさは同じように見られる。
 */
import { expect, test, type Page } from '@playwright/test';
import { mkdirSync, writeFileSync } from 'node:fs';
import { dirname } from 'node:path';

const RESULTS = 'tests/results/render-probe.json';
const probes: Record<string, unknown> = {};

function save(): void {
  mkdirSync(dirname(RESULTS), { recursive: true });
  writeFileSync(RESULTS, JSON.stringify(probes, null, 2), 'utf8');
}

interface Probe {
  backend: string;
  litPixels: number;
  coverage: number;
  drawnSplats: number;
  centroid: [number, number] | null;
  bbox: [number, number, number, number] | null;
  meanColor: [number, number, number] | null;
}

async function probe(page: Page, backend: 'webgpu' | 'webgl2'): Promise<Probe> {
  return page.evaluate(async (b) => {
    const api = window.__photosplat;
    if (!api) throw new Error('__photosplat がページに露出していません');
    // キャンバスは毎回作り直す。1枚のキャンバスに WebGPU と WebGL2 の
    // 両方のコンテキストは取れないため（2度目の getContext が null を返す）。
    const canvas = document.createElement('canvas');
    canvas.width = 256;
    canvas.height = 256;
    document.body.appendChild(canvas);
    try {
      return (await api.probePixels(canvas, 20_000, b, 256)) as unknown as Probe;
    } finally {
      canvas.remove();
    }
  }, backend);
}

/**
 * 球殻を正面から見た絵として妥当か。
 *
 * 「何か描けている」だけでは、画面の隅に1本線が出ていても通ってしまう。
 * 被写体は原点中心の球殻なので、重心は中央、外接矩形はほぼ正方形で
 * 画面をはみ出さない範囲に収まるはずである。
 */
function expectLooksLikeShell(p: Probe): void {
  expect(p.litPixels, '1ピクセルも描かれていません').toBeGreaterThan(1000);
  expect(p.coverage, '画面がほぼ塗り潰されています（クリアかブレンドの異常）').toBeLessThan(0.9);
  expect(p.drawnSplats, '背面カリングが全部落としています').toBeGreaterThan(0);

  const [cx, cy] = p.centroid ?? [NaN, NaN];
  expect(cx, `重心が中央にありません: ${cx}`).toBeGreaterThan(0.35);
  expect(cx).toBeLessThan(0.65);
  expect(cy, `重心が中央にありません: ${cy}`).toBeGreaterThan(0.35);
  expect(cy).toBeLessThan(0.65);

  const [x0, y0, x1, y1] = p.bbox ?? [NaN, NaN, NaN, NaN];
  const w = x1 - x0;
  const h = y1 - y0;
  expect(w, `描画領域が細すぎます: ${w}×${h}`).toBeGreaterThan(0.15);
  expect(h, `描画領域が細すぎます: ${w}×${h}`).toBeGreaterThan(0.15);
  // 球殻なので縦横比は 1 に近い。y を 1.25 倍に伸ばしてあるので余裕を持たせる。
  expect(w / h).toBeGreaterThan(0.4);
  expect(w / h).toBeLessThan(2.5);

  // 手続き生成の色は灰〜青寄りの中間色。真っ黒なら色が渡っていない。
  const [r, g, bl] = p.meanColor ?? [0, 0, 0];
  expect(r + g + bl, `平均色が暗すぎます: ${r},${g},${bl}`).toBeGreaterThan(60);
}

/**
 * ワールドの向きが画面のどこに出るか（docs/06 §6.2）。
 *
 * 上下・左右の取り違えは、単体テストも既存の描画検査も素通りする。とくに
 * 軸を 1 つだけ反転すると行列式が −1 になって**鏡像**になるが、対称な
 * 被写体では見た目が変わらない。非対称な目印を置いて実際に描いて読む。
 */
test.describe('ワールドの向き', () => {
  for (const backend of ['webgl2', 'webgpu'] as const) {
    test(`${backend}: 上は −Y、右は +X、鏡像にならない`, async ({ page }) => {
      await page.goto('/poc.html');
      const r = await page.evaluate(async (b) => {
        const api = window.__photosplat;
        if (!api) throw new Error('__photosplat がページに露出していません');
        const canvas = document.createElement('canvas');
        canvas.width = 192;
        canvas.height = 192;
        document.body.appendChild(canvas);
        try {
          return (await api.probeOrientation(canvas, b, 192)) as unknown as {
            backend: string;
            found: Record<string, { x: number; y: number } | null>;
          };
        } finally {
          canvas.remove();
        }
      }, backend);

      const top = r.found['top'];
      const bottom = r.found['bottom'];
      const right = r.found['right'];
      expect(top, '−Y の目印が描かれていません').not.toBeNull();
      expect(bottom, '+Y の目印が描かれていません').not.toBeNull();
      expect(right, '+X の目印が描かれていません').not.toBeNull();
      const t = top as { x: number; y: number };
      const b2 = bottom as { x: number; y: number };
      const rt = right as { x: number; y: number };

      expect(t.y, `−Y が画面の上半分に出ていません (y=${t.y.toFixed(2)})`).toBeLessThan(0.45);
      expect(b2.y, `+Y が画面の下半分に出ていません (y=${b2.y.toFixed(2)})`).toBeGreaterThan(0.55);
      // ここが鏡像の検査。+X は画面の右へ出なければならない。
      expect(rt.x, `+X が画面の右半分に出ていません (x=${rt.x.toFixed(2)})`).toBeGreaterThan(0.55);
    });
  }
});

test.describe('描画のピクセル検証', () => {
  test('WebGL2 が実際にピクセルを描く（D19 フォールバック）', async ({ page }) => {
    const errors: string[] = [];
    page.on('pageerror', (e) => errors.push(e.message));
    await page.goto('/poc.html');
    await expect(page.locator('#cap .pill')).toBeVisible({ timeout: 30_000 });

    const supported = await page.evaluate(
      () => !!document.createElement('canvas').getContext('webgl2'),
    );
    expect(supported, 'この環境には WebGL2 がありません').toBe(true);

    const p = await probe(page, 'webgl2');
    probes['webgl2'] = p;
    save();

    expect(p.backend).toBe('webgl2');
    expectLooksLikeShell(p);
    expect(errors, `ページ内で例外が起きました: ${errors.join(' / ')}`).toEqual([]);
  });

  test('WebGPU が実際にピクセルを描く', async ({ page }) => {
    const errors: string[] = [];
    page.on('pageerror', (e) => errors.push(e.message));
    await page.goto('/poc.html');
    await expect(page.locator('#cap .pill')).toBeVisible({ timeout: 30_000 });

    const hasWebGpu = await page.evaluate(async () => {
      if (!('gpu' in navigator) || !navigator.gpu) return false;
      return !!(await navigator.gpu.requestAdapter().catch(() => null));
    });
    if (!hasWebGpu) {
      // 黙って通過させず、未検証であることを残す
      probes['webgpu'] = 'skipped: WebGPU 未提供';
      save();
      test.skip(true, 'WebGPU が利用できない環境です');
      return;
    }

    const p = await probe(page, 'webgpu');
    probes['webgpu'] = p;
    save();

    expect(p.backend).toBe('webgpu');
    expectLooksLikeShell(p);
    expect(errors, `ページ内で例外が起きました: ${errors.join(' / ')}`).toEqual([]);
  });

  /**
   * 2つのバックエンドが同じ絵を出すか。
   *
   * 片方だけを見ても「それらしい絵」なら通ってしまう。実際、WGSL 側は
   * 構造体の整列規則でスプラット配列のストライドが 24 バイトではなく 32 バイトに
   * なっており、3個に1個だけ正しく読めていた。球は球に見えたまま画面全体に
   * 薄い靄が乗るだけだったので、片側の見た目では気付けなかった。
   * 独立した2実装を突き合わせれば、この種の壊れ方は数字で出る。
   */
  test('WebGPU と WebGL2 が同じ絵を出す', async ({ page }) => {
    await page.goto('/poc.html');
    await expect(page.locator('#cap .pill')).toBeVisible({ timeout: 30_000 });

    const hasWebGpu = await page.evaluate(async () => {
      if (!('gpu' in navigator) || !navigator.gpu) return false;
      return !!(await navigator.gpu.requestAdapter().catch(() => null));
    });
    if (!hasWebGpu) {
      probes['agreement'] = 'skipped: WebGPU 未提供';
      save();
      test.skip(true, 'WebGPU が利用できない環境です');
      return;
    }

    const gpu = await probe(page, 'webgpu');
    const gl = await probe(page, 'webgl2');
    probes['agreement'] = { webgpu: gpu, webgl2: gl };
    save();

    // カリング判定は両者とも同じ式なので、描画数はぴったり合うはず。
    // 浮動小数の丸めで境界のサーフェルが数個ずれる余地だけ残す。
    const diff = Math.abs(gpu.drawnSplats - gl.drawnSplats) / Math.max(gl.drawnSplats, 1);
    expect(diff, `描画数が食い違います: WebGPU ${gpu.drawnSplats} / WebGL2 ${gl.drawnSplats}`).toBeLessThan(0.02);
    // 塗られた面積も合うはず。片方だけに靄が乗っていればここで出る。
    expect(Math.abs(gpu.coverage - gl.coverage)).toBeLessThan(0.05);
  });
});
