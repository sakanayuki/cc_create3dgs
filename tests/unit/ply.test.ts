/** `.ply` 書き出しの構造と、サーフェル→クォータニオン変換（docs/05 §5.4）。 */
import { describe, expect, it } from 'vitest';
import {
  PlyGaussianOut,
  SH_C0,
  SURFEL_THICKNESS_LOG,
  dcToLinear,
  encodePly,
  linearToDc,
  plyByteLength,
  plyHeader,
  quatFromNormal,
} from '../../src/codec/ply';

const headerOf = (bytes: Uint8Array) => {
  const text = new TextDecoder().decode(bytes.subarray(0, 600));
  return text.slice(0, text.indexOf('end_header\n') + 'end_header\n'.length);
};

describe('PLY ヘッダ', () => {
  it('INRIA 標準の並びで17プロパティを宣言する', () => {
    const h = plyHeader(42);
    expect(h.startsWith('ply\nformat binary_little_endian 1.0\nelement vertex 42\n')).toBe(true);
    expect((h.match(/property float /g) ?? []).length).toBe(17);
    expect(h).toContain('property float nx');
    expect(h).toContain('property float f_dc_2');
    expect(h).toContain('property float rot_3');
    expect(h.endsWith('end_header\n')).toBe(true);
  });

  it('SH 1〜3次（f_rest_*）を出力しない', () => {
    // 単一画像に視点依存の情報は無いので持たせない（docs/04 §4.2）
    expect(plyHeader(1)).not.toContain('f_rest');
  });
});

describe('PLY 本体', () => {
  it('宣言した頂点数ぶんのバイト列を出す', () => {
    const n = 100;
    const bytes = encodePly(n, (i, out) => {
      out.x = i;
    });
    expect(bytes.length).toBe(plyByteLength(n));
    expect(bytes.length - headerOf(bytes).length).toBe(n * 17 * 4);
  });

  it('書いた値をリトルエンディアンの float32 で読み戻せる', () => {
    const bytes = encodePly(3, (i, out) => {
      out.x = i + 0.5;
      out.y = -i;
      out.z = 100 * i;
      out.nx = 0; out.ny = 0; out.nz = 1;
      out.opacityLogit = 2.5;
      out.rotW = 1;
    });
    const headerLen = headerOf(bytes).length;
    const dv = new DataView(bytes.buffer, headerLen);
    for (let i = 0; i < 3; i++) {
      const base = i * 17 * 4;
      expect(dv.getFloat32(base, true)).toBeCloseTo(i + 0.5, 5);
      expect(dv.getFloat32(base + 4, true)).toBeCloseTo(-i, 5);
      expect(dv.getFloat32(base + 8, true)).toBeCloseTo(100 * i, 3);
      expect(dv.getFloat32(base + 20, true)).toBeCloseTo(1, 5); // nz
      expect(dv.getFloat32(base + 36, true)).toBeCloseTo(2.5, 5); // opacity
      expect(dv.getFloat32(base + 52, true)).toBeCloseTo(1, 5); // rot_0 (w)
    }
  });

  it('法線フィールドに実際の値を書く（標準では常に0の無駄なフィールド）', () => {
    const bytes = encodePly(1, (_i, out) => {
      out.nx = 0.6; out.ny = -0.8; out.nz = 0;
    });
    const dv = new DataView(bytes.buffer, headerOf(bytes).length);
    expect(dv.getFloat32(12, true)).toBeCloseTo(0.6, 5);
    expect(dv.getFloat32(16, true)).toBeCloseTo(-0.8, 5);
    expect(dv.getFloat32(20, true)).toBeCloseTo(0, 5);
  });

  it('PlyGaussianOut.set で一括代入できる', () => {
    const slot = new PlyGaussianOut();
    slot.set({
      x: 1, y: 2, z: 3, nx: 0, ny: 1, nz: 0,
      dc: [0.1, 0.2, 0.3], opacityLogit: 1.5,
      logScale: [-5, -5, SURFEL_THICKNESS_LOG], rot: [1, 0, 0, 0],
    });
    expect(slot.x).toBe(1);
    expect(slot.dc2).toBeCloseTo(0.3);
    expect(slot.logScale2).toBe(SURFEL_THICKNESS_LOG);
  });
});

describe('SH DC と sRGB の変換', () => {
  it('往復する', () => {
    for (const c of [0, 0.25, 0.5, 0.75, 1]) {
      expect(dcToLinear(linearToDc(c))).toBeCloseTo(c, 6);
    }
  });

  it('3DGS の慣習どおり 0.5 が DC=0 に対応する', () => {
    expect(linearToDc(0.5)).toBeCloseTo(0, 9);
    expect(dcToLinear(0)).toBeCloseTo(0.5, 9);
    expect(SH_C0).toBeCloseTo(0.28209479, 7);
  });
});

describe('法線からクォータニオン', () => {
  const rotate = (q: readonly [number, number, number, number], v: readonly [number, number, number]) => {
    const [w, x, y, z] = q;
    // v' = v + 2w(q_v × v) + 2 q_v × (q_v × v)
    const cx = y * v[2] - z * v[1];
    const cy = z * v[0] - x * v[2];
    const cz = x * v[1] - y * v[0];
    const ccx = y * cz - z * cy;
    const ccy = z * cx - x * cz;
    const ccz = x * cy - y * cx;
    return [
      v[0] + 2 * w * cx + 2 * ccx,
      v[1] + 2 * w * cy + 2 * ccy,
      v[2] + 2 * w * cz + 2 * ccz,
    ] as const;
  };

  it('+Z を法線方向へ回す', () => {
    const normals: [number, number, number][] = [
      [0, 0, 1], [1, 0, 0], [0, 1, 0], [0, 0, -1],
      [0.577, 0.577, 0.577], [-0.6, 0.8, 0], [0.3, -0.4, 0.866],
    ];
    for (const n of normals) {
      const len = Math.hypot(...n);
      const un: [number, number, number] = [n[0] / len, n[1] / len, n[2] / len];
      const got = rotate(quatFromNormal(...un), [0, 0, 1]);
      expect(Math.hypot(got[0] - un[0], got[1] - un[1], got[2] - un[2])).toBeLessThan(1e-4);
    }
  });

  it('正規化されたクォータニオンを返す', () => {
    for (const n of [[0, 0, 1], [1, 0, 0], [0, 0, -1], [0.5, 0.5, 0.707]] as const) {
      const q = quatFromNormal(n[0], n[1], n[2]);
      expect(Math.hypot(...q)).toBeCloseTo(1, 5);
    }
  });

  it('正規化されていない法線を渡しても単位クォータニオンを返す', () => {
    // 法線は深度マップの平面フィット由来なので微小にずれる
    for (const n of [[0, 0, 3], [2, 0, 0], [0.5, 0.5, 0.707], [1e-14, 0, 1e-14]] as const) {
      const q = quatFromNormal(n[0], n[1], n[2]);
      expect(Math.hypot(...q)).toBeCloseTo(1, 6);
      expect(q.every(Number.isFinite)).toBe(true);
    }
  });

  it('180度反転（法線が -Z）でも壊れない', () => {
    const q = quatFromNormal(0, 0, -1);
    expect(Number.isFinite(q[0])).toBe(true);
    expect(Math.hypot(...q)).toBeCloseTo(1, 5);
    const got = rotate(q, [0, 0, 1]);
    expect(got[2]).toBeCloseTo(-1, 4);
  });
});

describe('サーフェルの厚み', () => {
  it('ゼロではなく exp(-8) を使う（行列を特異にしないため）', () => {
    const thickness = Math.exp(SURFEL_THICKNESS_LOG);
    expect(thickness).toBeGreaterThan(0);
    expect(thickness).toBeCloseTo(0.000335, 6);
    // 被写体サイズ(1)の 0.03% 程度。視覚的には平たいまま。
    expect(thickness).toBeLessThan(0.001);
  });
});
