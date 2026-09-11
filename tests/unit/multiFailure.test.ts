/**
 * 複数枚モードが失敗したときの言い方（docs/12 §12.16.4）。
 *
 * 実機で「1枚目のプレビューは出たのに、2枚目で
 * `InvalidStateError: The source image could not be decoded.` で止まる」
 * という報告を受けた。生の例外だけでは、3枚のうちどの写真の話なのか、
 * 写真が悪いのか端末が力尽きたのかが、利用者にも開発者にも分からない。
 *
 * ここで見るのは**言い方**である。推論は要らないので単体で回せる。
 */
import { afterEach, describe, expect, it, vi } from 'vitest';
import { ensureDecodable, explainFailure } from '../../src/pipeline/generateMulti';

const g = globalThis as unknown as { createImageBitmap?: unknown };
const original = g.createImageBitmap;

afterEach(() => {
  if (original === undefined) delete g.createImageBitmap;
  else g.createImageBitmap = original;
});

describe('開けない写真を、生成を始める前に見つける', () => {
  it('開けない枠の名前とファイル名を言う', async () => {
    g.createImageBitmap = vi.fn(async (blob: Blob) => {
      if ((blob as File).name === 'right.heic') {
        throw new DOMException('The source image could not be decoded.', 'InvalidStateError');
      }
      return { close: (): void => {} };
    });

    const photos = [
      { slot: 'front' as const, blob: new File([], 'front.jpg'), name: 'front.jpg' },
      { slot: 'right' as const, blob: new File([], 'right.heic'), name: 'right.heic' },
    ];

    await expect(ensureDecodable(photos)).rejects.toThrow(/右向き（right\.heic）/);
    // 原因は決めつけない。ここで落ちても資源不足のことがある（PR #7 の指摘）。
    await expect(ensureDecodable(photos)).rejects.toThrow(/HEIC/);
    await expect(ensureDecodable(photos)).rejects.toThrow(/資源/);
  });

  it('開けた bitmap は必ず閉じる（確認そのものが資源を食い潰さないため）', async () => {
    const close = vi.fn();
    g.createImageBitmap = vi.fn(async () => ({ close }));

    await ensureDecodable([
      { slot: 'front' as const, blob: new Blob() },
      { slot: 'left' as const, blob: new Blob() },
    ]);

    expect(close).toHaveBeenCalledTimes(2);
  });

  it('createImageBitmap が無い環境では何もしない', async () => {
    delete g.createImageBitmap;
    await expect(
      ensureDecodable([
        { slot: 'front' as const, blob: new Blob() },
        { slot: 'right' as const, blob: new Blob() },
      ]),
    ).resolves.toBeUndefined();
  });
});

describe('途中で落ちたときの言い方', () => {
  it('一度開けた写真が後から開けなくなったら、資源のほうを疑う', () => {
    const e = new DOMException('The source image could not be decoded.', 'InvalidStateError');
    const msg = explainFailure(e);
    // ここへ来る時点で ensureDecodable は通っている。ファイルのせいにしない。
    expect(msg).toMatch(/資源/);
    expect(msg).toMatch(/2枚/);
    expect(msg).not.toMatch(/壊れ/);
  });

  it('それ以外の例外はそのまま見せる（原因を隠さない）', () => {
    expect(explainFailure(new Error('モデルを取得できませんでした'))).toContain(
      'モデルを取得できませんでした',
    );
  });
});
