/** `.pgs` コンテナの往復と、壊れた入力に対する振る舞い（docs/05 §5.2）。 */
import { describe, expect, it } from 'vitest';
import { PGS_MAGIC, missingRequiredChunks, packPgs, unpackPgs } from '../../src/codec/pgs';

const sampleManifest = {
  version: 1,
  grid: { width: 1024, height: 1024 },
  camera: { focalPx: 982, cx: 512, cy: 512, focalSource: 'da3' },
  depthRange: { nearZ: 0.679, farZ: 1.321 },
  depthBits: 12,
};

const chunk = (n: number, seed: number) =>
  Uint8Array.from({ length: n }, (_, i) => (i * seed + 7) % 256);

function fullChunks(): Map<string, Uint8Array> {
  return new Map<string, Uint8Array>([
    ['DPTH', chunk(1000, 3)],
    ['DPTL', chunk(700, 5)],
    ['COLR', chunk(1500, 7)],
    ['ALFA', chunk(300, 11)],
    ['BCOL', chunk(400, 13)],
    ['INPT', chunk(250, 17)],
  ]);
}

describe('.pgs コンテナ', () => {
  it('マニフェストとチャンクを往復する', () => {
    const chunks = fullChunks();
    const file = unpackPgs(packPgs(sampleManifest, chunks));

    expect(file.manifest).toEqual(sampleManifest);
    expect([...file.chunks.keys()]).toEqual([...chunks.keys()]);
    for (const [id, expected] of chunks) {
      expect([...(file.chunks.get(id) ?? [])]).toEqual([...expected]);
    }
    expect(missingRequiredChunks(file)).toEqual([]);
  });

  it('マジックとバージョンをヘッダに書く', () => {
    const bytes = packPgs(sampleManifest, fullChunks());
    expect(String.fromCharCode(...bytes.subarray(0, 8))).toBe(PGS_MAGIC);
    const dv = new DataView(bytes.buffer);
    expect(dv.getUint16(8, true)).toBe(1); // major
  });

  it('空のチャンク集合でも往復する', () => {
    const file = unpackPgs(packPgs({ a: 1 }, new Map()));
    expect(file.manifest).toEqual({ a: 1 });
    expect(file.chunks.size).toBe(0);
  });

  it('長さ0のペイロードを扱える', () => {
    const file = unpackPgs(packPgs({}, new Map([['THMB', new Uint8Array(0)]])));
    expect(file.chunks.get('THMB')?.length).toBe(0);
  });

  it('未知のチャンクは読み飛ばして呼び出し側に伝える（前方互換）', () => {
    const chunks = fullChunks();
    chunks.set('XPRM', chunk(64, 19)); // 将来のバージョンが足したチャンク
    const file = unpackPgs(packPgs(sampleManifest, chunks));
    expect(file.unknownChunks).toEqual(['XPRM']);
    expect(file.chunks.get('XPRM')?.length).toBe(64);
    expect(missingRequiredChunks(file)).toEqual([]); // 必須は揃っている
  });

  it('必須チャンクの欠けを検出する', () => {
    const chunks = fullChunks();
    chunks.delete('BCOL');
    chunks.delete('ALFA');
    const file = unpackPgs(packPgs(sampleManifest, chunks));
    expect(missingRequiredChunks(file).sort()).toEqual(['ALFA', 'BCOL']);
  });

  it('マルチバイト文字を含むマニフェストを壊さない', () => {
    const m = { メモ: '人物モード・縁色の行方向伸長', preset: '高品質' };
    expect(unpackPgs(packPgs(m, new Map())).manifest).toEqual(m);
  });

  describe('壊れた入力', () => {
    it('マジックが違えば拒否する', () => {
      const bytes = packPgs(sampleManifest, fullChunks());
      bytes[0] = 0x58;
      expect(() => unpackPgs(bytes)).toThrow(/マジック/);
    });

    it('短すぎるファイルを拒否する', () => {
      expect(() => unpackPgs(new Uint8Array(8))).toThrow(/短すぎ/);
    });

    it('メジャーバージョンが新しければ拒否する', () => {
      const bytes = packPgs(sampleManifest, fullChunks());
      new DataView(bytes.buffer).setUint16(8, 99, true);
      expect(() => unpackPgs(bytes)).toThrow(/このバージョンでは読めません/);
    });

    it('チャンク長が壊れていれば拒否する', () => {
      const bytes = packPgs(sampleManifest, fullChunks());
      const manifestLen = new DataView(bytes.buffer).getUint32(12, true);
      // 最初のチャンクの長さを膨らませる
      new DataView(bytes.buffer).setUint32(16 + manifestLen + 4, 0xffffff, true);
      expect(() => unpackPgs(bytes)).toThrow(/長さが壊れています/);
    });

    it('末尾に余分なバイトがあれば拒否する', () => {
      const bytes = packPgs(sampleManifest, fullChunks());
      const padded = new Uint8Array(bytes.length + 3);
      padded.set(bytes);
      expect(() => unpackPgs(padded)).toThrow(/末尾/);
    });

    it('マニフェストが JSON でなければ理由を添えて拒否する', () => {
      const bytes = packPgs(sampleManifest, new Map());
      bytes[16] = 0x7b; // '{' で始まるが壊れた JSON にする
      bytes[17] = 0x7b;
      expect(() => unpackPgs(bytes)).toThrow(/マニフェストを解釈できません/);
    });

    it('4文字でないチャンク ID を書こうとしたら止める', () => {
      expect(() => packPgs({}, new Map([['TOOLONG', new Uint8Array(1)]]))).toThrow(/4文字/);
    });
  });
});
