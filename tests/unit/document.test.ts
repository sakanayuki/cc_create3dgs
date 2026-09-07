/**
 * `PhotoSplatDocument` ⇄ `.pgs` の往復（docs/05 §5.2）。
 *
 * 画像符号化はブラウザの OffscreenCanvas に任せる設計なので、Node では
 * 「符号化せずそのまま持つ」スタブコーデックを差し込んで、コンテナと
 * プレーンの詰め替えの正しさだけを見る。実際の PNG/WebP は E2E で通る。
 */
import { describe, expect, it } from 'vitest';
import { decodeDocument, encodeDocument } from '../../src/codec/document';
import type { ImageCodec } from '../../src/codec/images';
import type { PhotoSplatDocument } from '../../src/doc/PhotoSplatDocument';
import { unpackPgs } from '../../src/codec/pgs';

/** RGBA をそのまま持つスタブ。先頭に幅と高さを付けて往復できるようにする。 */
const stubCodec: ImageCodec = {
  async encode(rgba, width, height) {
    const out = new Uint8Array(8 + rgba.length);
    new DataView(out.buffer).setUint32(0, width, true);
    new DataView(out.buffer).setUint32(4, height, true);
    out.set(rgba, 8);
    return out;
  },
  async decode(bytes) {
    const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    const width = dv.getUint32(0, true);
    const height = dv.getUint32(4, true);
    return { rgba: new Uint8ClampedArray(bytes.subarray(8)), width, height };
  },
};

function makeDoc(size = 16): PhotoSplatDocument {
  const n = size * size;
  const half = (size / 2) * (size / 2);
  const frontDepth = new Uint16Array(n);
  const frontColor = new Uint8ClampedArray(n * 3);
  const alpha = new Uint8ClampedArray(n);
  for (let i = 0; i < n; i++) {
    frontDepth[i] = Math.round((0.3 + 0.5 * (i / n)) * 65535);
    frontColor[i * 3] = i % 256;
    frontColor[i * 3 + 1] = (i * 3) % 256;
    frontColor[i * 3 + 2] = (i * 7) % 256;
    // 中央付近だけ被写体にする
    const x = i % size;
    const y = Math.floor(i / size);
    alpha[i] = Math.hypot(x - size / 2, y - size / 2) < size / 3 ? 255 : 0;
  }
  return {
    width: size,
    height: size,
    camera: { focalPx: 982, cx: size / 2, cy: size / 2, focalSource: 'da3' },
    depthRange: { nearZ: 0.679, farZ: 1.321 },
    frontDepth,
    frontColor,
    alpha,
    backColor: new Uint8ClampedArray(half * 3).fill(90),
    inpaint: new Uint8ClampedArray(half * 4).fill(60),
    backShell: {
      enabled: true, thicknessT: 0.321, profile: 'ellipsoid', density: 4,
      colorMode: 'edge-extend', shadeBase: 0.5, shadeRange: 0.1,
    },
    skirt: {
      enabled: true, edgeThreshold: 0.018, lengthScale: 0.85,
      opacityFalloff: 'linear', colorSource: 'inpaint',
    },
    sampling: { enabled: true, thetaDepth: 0.006, thetaColor: 4, maxCell: 8 },
    meta: {
      createdAt: '2026-09-07T00:00:00Z',
      app: 'PhotoSplat 0.1.0',
      preset: 'standard',
      subjectMode: 'person',
      models: { depth: 'depth-anything-v3-small@q4f16' },
      gaussianCount: { front: 293600, back: 104858, skirt: 25000 },
    },
  };
}

describe('ドキュメントの往復', () => {
  it('メタデータとパラメータを保つ', async () => {
    const doc = makeDoc();
    const back = await decodeDocument(await encodeDocument(doc, stubCodec), stubCodec);

    expect(back.width).toBe(doc.width);
    expect(back.camera).toEqual(doc.camera);
    expect(back.depthRange).toEqual(doc.depthRange);
    expect(back.backShell).toEqual(doc.backShell);
    expect(back.skirt).toEqual(doc.skirt);
    expect(back.sampling).toEqual(doc.sampling);
    expect(back.meta).toEqual(doc.meta);
  });

  it('色・α・背面色・補完テクスチャをそのまま戻す', async () => {
    const doc = makeDoc();
    const back = await decodeDocument(await encodeDocument(doc, stubCodec), stubCodec);
    expect([...back.frontColor]).toEqual([...doc.frontColor]);
    expect([...back.alpha]).toEqual([...doc.alpha]);
    expect([...back.backColor]).toEqual([...doc.backColor]);
    expect([...(back.inpaint ?? [])]).toEqual([...(doc.inpaint ?? [])]);
  });

  it('深度は 12bit 量子化の範囲でしか劣化しない', async () => {
    const doc = makeDoc();
    const back = await decodeDocument(await encodeDocument(doc, stubCodec), stubCodec);
    const step = 1 << (16 - 12);
    for (let i = 0; i < doc.frontDepth.length; i++) {
      expect(Math.abs((back.frontDepth[i] as number) - (doc.frontDepth[i] as number)))
        .toBeLessThanOrEqual(step / 2);
    }
  });

  it('10bit（軽量プリセット）でも往復し、量子化幅どおりに劣化する', async () => {
    const doc = makeDoc();
    const bytes = await encodeDocument(doc, stubCodec, { depthBits: 10 });
    expect((unpackPgs(bytes).manifest as { depthBits: number }).depthBits).toBe(10);
    const back = await decodeDocument(bytes, stubCodec);
    const step = 1 << (16 - 10);
    for (let i = 0; i < doc.frontDepth.length; i++) {
      expect(Math.abs((back.frontDepth[i] as number) - (doc.frontDepth[i] as number)))
        .toBeLessThanOrEqual(step / 2);
    }
  });
});

describe('保存しないものを保存していないこと', () => {
  it('チャンクは深度2枚・色・α・背面色・補完テクスチャの6つだけ', async () => {
    // 位置・法線・スケール・厚み・スカート幾何・サンプリング分割は
    // すべて決定的な導出物なので入っていてはいけない（docs/04 §4.5.3）
    const file = unpackPgs(await encodeDocument(makeDoc(), stubCodec));
    expect([...file.chunks.keys()].sort()).toEqual(['ALFA', 'BCOL', 'COLR', 'DPTH', 'DPTL', 'INPT']);
    expect(file.unknownChunks).toEqual([]);
  });

  it('補完テクスチャが無ければ INPT を書かず、スカートの色を引き伸ばしに落とす', async () => {
    const doc = { ...makeDoc(), inpaint: null };
    const bytes = await encodeDocument(doc, stubCodec);
    const file = unpackPgs(bytes);
    expect(file.chunks.has('INPT')).toBe(false);
    expect((file.manifest as { skirt: { colorSource: string } }).skirt.colorSource).toBe('stretch');

    const back = await decodeDocument(bytes, stubCodec);
    expect(back.inpaint).toBeNull();
    expect(back.skirt.colorSource).toBe('stretch');
  });
});

describe('壊れた入力', () => {
  it('必須チャンクが欠けたファイルを拒否する', async () => {
    const bytes = await encodeDocument(makeDoc(), stubCodec);
    const file = unpackPgs(bytes);
    const chunks = new Map(file.chunks);
    chunks.delete('BCOL');
    const { packPgs } = await import('../../src/codec/pgs');
    const broken = packPgs(file.manifest, chunks);
    await expect(decodeDocument(broken, stubCodec)).rejects.toThrow(/必須チャンクが欠けています.*BCOL/s);
  });

  it('マニフェストに必須項目が無ければ拒否する', async () => {
    const bytes = await encodeDocument(makeDoc(), stubCodec);
    const file = unpackPgs(bytes);
    const { packPgs } = await import('../../src/codec/pgs');
    const broken = packPgs({ version: 1 }, file.chunks);
    await expect(decodeDocument(broken, stubCodec)).rejects.toThrow(/マニフェストに必須項目がありません/);
  });

  it('長さの噛み合わないドキュメントを書き出そうとしたら止める', async () => {
    const doc = { ...makeDoc(), alpha: new Uint8ClampedArray(5) };
    await expect(encodeDocument(doc, stubCodec)).rejects.toThrow(/ドキュメントが不正です[\s\S]*alpha/);
  });
});
