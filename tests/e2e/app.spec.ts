/**
 * 本体アプリの通し（docs/02, docs/03 §3.10）。
 *
 * この環境のブラウザは HuggingFace に到達できないので、モデルの取得は
 * 失敗する。それでも確かめられること、というより**確かめなければならない**
 * ことがある。「押したのに何も起きない」状態にならないことだ。
 * 成功するか、理由を出して失敗するか、必ずどちらかに落ち着くのを見る。
 */
import { expect, test } from '@playwright/test';

/** 小さな PNG を1枚その場で作る。外部ファイルに依存させない。 */
async function makePhoto(): Promise<{ name: string; mimeType: string; buffer: Buffer }> {
  // 32×32 の単色 PNG（zlib 無圧縮ブロックで手組み）
  const w = 32;
  const h = 32;
  const raw = Buffer.alloc((w * 3 + 1) * h);
  for (let y = 0; y < h; y++) {
    const o = y * (w * 3 + 1);
    raw[o] = 0;
    for (let x = 0; x < w; x++) {
      raw[o + 1 + x * 3] = 200;
      raw[o + 2 + x * 3] = 150;
      raw[o + 3 + x * 3] = 120;
    }
  }
  const { deflateSync } = await import('node:zlib');
  const idat = deflateSync(raw);
  const crcTable: number[] = [];
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    crcTable[n] = c >>> 0;
  }
  const crc = (buf: Buffer): number => {
    let c = 0xffffffff;
    for (const b of buf) c = (crcTable[(c ^ b) & 0xff] as number) ^ (c >>> 8);
    return (c ^ 0xffffffff) >>> 0;
  };
  const chunk = (type: string, data: Buffer): Buffer => {
    const len = Buffer.alloc(4);
    len.writeUInt32BE(data.length);
    const body = Buffer.concat([Buffer.from(type, 'ascii'), data]);
    const c = Buffer.alloc(4);
    c.writeUInt32BE(crc(body));
    return Buffer.concat([len, body, c]);
  };
  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(w, 0);
  ihdr.writeUInt32BE(h, 4);
  ihdr[8] = 8; // bit depth
  ihdr[9] = 2; // truecolor
  const png = Buffer.concat([
    Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
    chunk('IHDR', ihdr),
    chunk('IDAT', idat),
    chunk('IEND', Buffer.alloc(0)),
  ]);
  return { name: 'test.png', mimeType: 'image/png', buffer: png };
}

test.describe('本体アプリ', () => {
  test('写真を選ぶと生成画面に移り、必ず結果か理由に行き着く', async ({ page }) => {
    const errors: string[] = [];
    page.on('pageerror', (e) => errors.push(e.message));

    await page.goto('/');
    await expect(page.locator('#env')).not.toContainText('判定中', { timeout: 30_000 });

    await page.locator('#photo').setInputFiles(await makePhoto());

    // まず「つくっています」に移る
    await expect(page.locator('#work')).toBeVisible({ timeout: 15_000 });
    await expect(page.locator('#pick')).toBeHidden();

    // 進捗が動き出す（押しただけで固まらない）
    await expect(page.locator('#stage')).not.toHaveText('準備しています', { timeout: 60_000 });

    // 成功して見る画面に行くか、理由を出して止まるか。必ずどちらか。
    //
    // **待つ時間は 8 分。** ここは速度を測る検査ではない（playwright.config.ts の
    // 冒頭のとおり、E2E は結果の正しさだけを見る）。それなのに 5 分で切っていたため、
    // ランナーの速さのぶれで落ちるようになっていた。CI ではモデルを HuggingFace から
    // 約 46MB 落とし、そのうえで SwiftShader の WASM で推論する。実測 4.3 分の回が
    // あり、余裕がほとんど無かった（2026-09-11 に main で 5.0 分に届いて落ちた）。
    //
    // **落ちたときは、どこで止まったかを書く。** 前回は「成功も失敗もしていません」
    // としか出ず、モデルの取得で止まったのか推論で止まったのか分からなかった。
    const done = page.locator('#view');
    const failed = page.locator('#workError');
    await expect(async () => {
      const ok = await done.isVisible();
      const ng = await failed.isVisible();
      const stage = (await page.locator('#stage').textContent().catch(() => '')) ?? '';
      expect(ok || ng, `生成画面のまま、成功も失敗もしていません（いま: ${stage.trim()}）`).toBe(
        true,
      );
    }).toPass({ timeout: 8 * 60 * 1000, intervals: [2000] });

    if (await failed.isVisible()) {
      // 失敗するなら、何が起きたかを必ず書く。空のカードは出さない。
      const text = (await failed.textContent()) ?? '';
      expect(text.trim().length, '失敗したのに理由が空です').toBeGreaterThan(20);
      expect(text).toContain('生成できませんでした');
      // 選び直せること
      await page.locator('#retry').click();
      await expect(page.locator('#pick')).toBeVisible();
    } else {
      // 閲覧画面はプレビュー（下書き）の時点で出るので、統計が埋まるのは
      // その後になる。仕上げが終わるまで待つ。
      // 仕上げ中は「下書きです」と出ていること。終われば消えること。
      await expect(page.locator('#stats')).toContainText('ガウシアン', { timeout: 10 * 60 * 1000 });
      await expect(page.locator('#refining')).toBeHidden();
      await expect(page.locator('#save')).toBeVisible();
    }

    // 未捕捉の例外を残したまま終わらない
    expect(errors, `ページ内で例外が起きました: ${errors.join(' / ')}`).toEqual([]);
  });
});
