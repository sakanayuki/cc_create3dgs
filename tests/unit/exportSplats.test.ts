/**
 * 書き出し（docs/05 §5.3）。
 *
 * 24 バイトのスプラット列を .spz / .ply に直す経路。ここが壊れると
 * 「アプリでは見えるのに、持ち出したファイルが真っ黒」になる。
 * 実際 v1 の設計では SPZ の色を sRGB のまま入れていて、白と黒が飽和していた。
 */
import { describe, expect, it } from 'vitest';
import { encodeOct, packHalf2, packRgba8 } from '../../src/codec/pack';
import { SPZ_FRACTIONAL_BITS, spzColorToLinear } from '../../src/codec/spz';
import {
  EXPORT_FORMATS,
  SPLAT_STRIDE,
  splatsToPly,
  splatsToSplat,
  splatsToSpzRaw,
  toSplatFile,
  type ExportFormat,
} from '../../src/ui/export';
import { SPLAT_BYTES } from '../../src/render/SplatRenderer';

/** 既知の値を持つスプラットを1個だけ作る。 */
function oneSplat(
  pos: [number, number, number],
  nrm: [number, number, number],
  scale: [number, number],
  rgba: [number, number, number, number],
): Uint8Array {
  const buf = new ArrayBuffer(SPLAT_BYTES);
  const f = new Float32Array(buf);
  const u = new Uint32Array(buf);
  f[0] = pos[0];
  f[1] = pos[1];
  f[2] = pos[2];
  u[3] = encodeOct(nrm[0], nrm[1], nrm[2]);
  u[4] = packHalf2(scale[0], scale[1]);
  u[5] = packRgba8(rgba[0], rgba[1], rgba[2], rgba[3]);
  return new Uint8Array(buf);
}

describe('.spz 書き出し', () => {
  it('位置が固定小数点で往復する', () => {
    const data = oneSplat([0.25, -0.125, 0.5], [0, 0, 1], [0.01, 0.01], [128, 128, 128, 255]);
    const spz = splatsToSpzRaw(data, 1);
    const dv = new DataView(spz.buffer, spz.byteOffset);
    const scale = 1 << SPZ_FRACTIONAL_BITS;
    const read = (o: number): number => {
      const v = spz[o]! | (spz[o + 1]! << 8) | (spz[o + 2]! << 16);
      // 24bit 符号付き
      return ((v << 8) >> 8) / scale;
    };
    expect(dv.getUint32(8, true)).toBe(1);
    expect(read(16)).toBeCloseTo(0.25, 3);
    expect(read(19)).toBeCloseTo(-0.125, 3);
    expect(read(22)).toBeCloseTo(0.5, 3);
  });

  it('白と黒が飽和しない', () => {
    // SPZ の色は SH の DC 項であって sRGB ではない。素通しすると
    // 黒が 0、白が 255 に張り付いて情報が消える（v1 設計の誤り）。
    for (const [name, v] of [['黒', 0], ['白', 255]] as const) {
      const data = oneSplat([0, 0, 0], [0, 0, 1], [0.01, 0.01], [v, v, v, 255]);
      const spz = splatsToSpzRaw(data, 1);
      const colorOffset = 16 + 9 + 1; // 位置9B + α1B のあと
      const u8 = spz[colorOffset] as number;
      expect(u8, `${name} が端に張り付いています: ${u8}`).toBeGreaterThan(0);
      expect(u8, `${name} が端に張り付いています: ${u8}`).toBeLessThan(255);
    }
  });

  it('明るい色ほど大きなリニア値になる', () => {
    const linearOf = (v: number): number => {
      const data = oneSplat([0, 0, 0], [0, 0, 1], [0.01, 0.01], [v, v, v, 255]);
      const spz = splatsToSpzRaw(data, 1);
      return spzColorToLinear(spz[16 + 9 + 1] as number);
    };
    expect(linearOf(200)).toBeGreaterThan(linearOf(100));
    expect(linearOf(100)).toBeGreaterThan(linearOf(20));
  });

  it('サーフェルは法線方向だけ極端に薄い', () => {
    const data = oneSplat([0, 0, 0], [0, 0, 1], [0.02, 0.02], [128, 128, 128, 255]);
    const spz = splatsToSpzRaw(data, 1);
    const scaleOffset = 16 + 9 + 1 + 3;
    const decode = (o: number): number => (spz[o] as number) / 16 - 10;
    const s0 = decode(scaleOffset);
    const s2 = decode(scaleOffset + 2);
    expect(s2).toBeLessThan(s0 - 3); // 2軸に対して桁違いに薄い
  });
});

describe('.ply 書き出し', () => {
  it('ヘッダと本体の大きさが整合する', () => {
    const data = oneSplat([0, 0, 0], [0, 0, 1], [0.01, 0.01], [200, 150, 100, 255]);
    const ply = splatsToPly(data, 1);
    const text = new TextDecoder().decode(ply.subarray(0, 512));
    expect(text).toContain('ply');
    expect(text).toContain('element vertex 1');
    const headerEnd = text.indexOf('end_header\n') + 'end_header\n'.length;
    // 17 個の float32 = 68 バイト
    expect(ply.length - headerEnd).toBe(68);
  });

  it('不透明度がロジットで入る', () => {
    const half = oneSplat([0, 0, 0], [0, 0, 1], [0.01, 0.01], [128, 128, 128, 128]);
    const opaque = oneSplat([0, 0, 0], [0, 0, 1], [0.01, 0.01], [128, 128, 128, 255]);
    const read = (d: Uint8Array): number => {
      const ply = splatsToPly(d, 1);
      const text = new TextDecoder().decode(ply.subarray(0, 512));
      const off = text.indexOf('end_header\n') + 'end_header\n'.length;
      const dv = new DataView(ply.buffer, ply.byteOffset + off);
      return dv.getFloat32(9 * 4, true); // x,y,z,nx,ny,nz,dc0..2 の次
    };
    // α = 0.5 → ロジット 0 付近、α ≈ 1 → 大きな正
    expect(Math.abs(read(half))).toBeLessThan(0.1);
    expect(read(opaque)).toBeGreaterThan(3);
  });
});

describe('色空間（3DGS の DC 項）', () => {
  /**
   * 3DGS の SH DC 項は「学習に使った画像の画素値」を表す。ラスタライザの出力
   * `SH_C0·dc + 0.5` を、読み込んだ PNG の値（sRGB を 255 で割っただけ）と
   * 直接比べて学習するので、dc が符号化しているのは sRGB/255 である。
   *
   * ここで sRGB→リニア変換を挟むと、両端（0 と 255）は一致するので飽和検査は
   * 通ってしまうが、中間調が大きくずれる。中間の灰色 128 が他のビューアで
   * 55 として表示されていた。
   */
  const SH_C0 = 0.28209479177387814;

  const dcOf = (v: number): number => {
    const data = oneSplat([0, 0, 0], [0, 0, 1], [0.01, 0.01], [v, v, v, 255]);
    const ply = splatsToPly(data, 1);
    const text = new TextDecoder().decode(ply.subarray(0, 512));
    const off = text.indexOf('end_header\n') + 'end_header\n'.length;
    return new DataView(ply.buffer, ply.byteOffset + off).getFloat32(6 * 4, true);
  };

  it('中間調が往復する', () => {
    for (const v of [64, 128, 192]) {
      // ビューア側の復元式に通すと元の画素値に戻ること
      const shown = (SH_C0 * dcOf(v) + 0.5) * 255;
      expect(shown, `画素値 ${v}`).toBeCloseTo(v, 0);
    }
  });

  it('両端も往復する', () => {
    expect((SH_C0 * dcOf(0) + 0.5) * 255).toBeCloseTo(0, 0);
    expect((SH_C0 * dcOf(255) + 0.5) * 255).toBeCloseTo(255, 0);
  });

  it('明るいほど dc が大きい', () => {
    expect(dcOf(200)).toBeGreaterThan(dcOf(128));
    expect(dcOf(128)).toBeGreaterThan(dcOf(50));
  });
});

describe('.splat 書き出し', () => {
  /** 参照実装（antimatter15/splat の convert.py）に合わせた並び。 */
  it('1個 32 バイト', () => {
    const data = oneSplat([0.1, 0.2, 0.3], [0, 0, 1], [0.02, 0.03], [200, 150, 100, 220]);
    const out = splatsToSplat(data, 1);
    expect(out.length).toBe(SPLAT_STRIDE);
  });

  it('位置とスケールが線形の float32 で入る', () => {
    const data = oneSplat([0.1, -0.2, 0.3], [0, 0, 1], [0.02, 0.03], [200, 150, 100, 220]);
    const out = splatsToSplat(data, 1);
    const f = new Float32Array(out.buffer, out.byteOffset, 6);
    expect(f[0]).toBeCloseTo(0.1, 3);
    expect(f[1]).toBeCloseTo(-0.2, 3);
    expect(f[2]).toBeCloseTo(0.3, 3);
    // .splat のスケールは対数ではなく線形
    expect(f[3]).toBeCloseTo(0.02, 3);
    expect(f[4]).toBeCloseTo(0.03, 3);
    expect(f[5]).toBeLessThan(0.001); // 法線方向は極小（サーフェル）
  });

  it('色は画素値がそのまま入る（DC を経由しない）', () => {
    const data = oneSplat([0, 0, 0], [0, 0, 1], [0.02, 0.02], [200, 150, 100, 220]);
    const out = splatsToSplat(data, 1);
    expect([out[24], out[25], out[26], out[27]]).toEqual([200, 150, 100, 220]);
  });

  it('回転は (q/|q|)·128+128 で、順序は w, x, y, z', () => {
    // 法線が +z のとき、接平面の基底は回転なし付近になる
    const data = oneSplat([0, 0, 0], [0, 0, 1], [0.02, 0.02], [128, 128, 128, 255]);
    const out = splatsToSplat(data, 1);
    const q = [out[28]!, out[29]!, out[30]!, out[31]!].map((v) => (v - 128) / 128);
    const len = Math.hypot(...q);
    expect(len, `正規化されていません: ${q.join(', ')}`).toBeCloseTo(1, 1);
  });

  it('大きく不透明なものから並ぶ（参照実装と同じ順）', () => {
    const big = oneSplat([0, 0, 0], [0, 0, 1], [0.1, 0.1], [255, 0, 0, 255]);
    const small = oneSplat([1, 1, 1], [0, 0, 1], [0.001, 0.001], [0, 255, 0, 255]);
    const both = new Uint8Array(big.length + small.length);
    both.set(small, 0);
    both.set(big, small.length);
    const out = splatsToSplat(both, 2);
    // 先頭には大きいほう（赤）が来る
    expect(out[24]).toBe(255);
    expect(out[25]).toBe(0);
  });
});

describe('形式の選択', () => {
  const data = oneSplat([0, 0, 0], [0, 0, 1], [0.02, 0.02], [180, 140, 120, 255]);

  it('3つとも書き出せて、中身が違う', async () => {
    const blobs = await Promise.all(
      (['spz', 'ply', 'splat'] as ExportFormat[]).map((f) => toSplatFile(data, 1, f)),
    );
    const sizes = blobs.map((b) => b.size);
    expect(sizes.every((n) => n > 0), `サイズ: ${sizes.join(', ')}`).toBe(true);
    // .ply はヘッダを持つので最も大きい
    expect(sizes[1]).toBeGreaterThan(sizes[2] as number);
  });

  it('拡張子と説明が揃っている', () => {
    for (const [key, info] of Object.entries(EXPORT_FORMATS)) {
      expect(info.extension, key).toBe(key);
      expect(info.label, key).toBe(`.${key}`);
      expect(info.note.length, key).toBeGreaterThan(5);
    }
  });

  it('.splat の実体は 32 バイト × 個数', async () => {
    const blob = await toSplatFile(data, 1, 'splat');
    expect(blob.size).toBe(SPLAT_STRIDE);
  });
});
