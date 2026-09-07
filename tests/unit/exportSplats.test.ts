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
import { splatsToPly, splatsToSpzRaw } from '../../src/ui/export';
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
