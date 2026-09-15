/**
 * 「人体としてあり得る形か」の物差し（docs/11 §11.5、docs/13 §13.3）。
 *
 * 合格ラインの判定に使う数なので、**答えが分かっている形**で検算する。
 */
import { describe, expect, it } from 'vitest';
import { SPLAT_BYTES } from '../../src/render/SplatRenderer';
import { encodeOct, packHalf2, packRgba8 } from '../../src/codec/pack';
import {
  crossSectionBulge,
  crossSectionDent,
  frontDepthMap,
} from '../helpers/bodyShape';

/** z(x, y) で決まる面を、格子状のスプラットにする。 */
const SIDE = 200;
/** 点が地図より密になるようにしておく（疎だと区間が 1 画素ずつに割れる）。 */
function surface(
  shape: (u: number, v: number) => number,
  n = 400,
): { data: Uint8Array; count: number } {
  const buf = new ArrayBuffer(n * n * SPLAT_BYTES);
  const f = new Float32Array(buf);
  const u32 = new Uint32Array(buf);
  const stride = SPLAT_BYTES / 4;
  let k = 0;
  for (let iy = 0; iy < n; iy++) {
    for (let ix = 0; ix < n; ix++) {
      const u = ix / (n - 1); // 0..1（横）
      const v = iy / (n - 1); // 0..1（縦）
      const o = k * stride;
      f[o] = u - 0.5;
      f[o + 1] = v - 0.5;
      f[o + 2] = shape(u, v);
      u32[o + 3] = encodeOct(0, 0, -1);
      u32[o + 4] = packHalf2(0.004, 0.004);
      u32[o + 5] = packRgba8(200, 200, 200, 255);
      k++;
    }
  }
  return { data: new Uint8Array(buf), count: n * n };
}

describe('横断面の物差し', () => {
  it('平らな面は、飛び出しも凹みも 0', () => {
    const { data, count } = surface(() => 0);
    const map = frontDepthMap(data, count, SIDE);
    expect(crossSectionBulge(map).p99).toBeCloseTo(0, 6);
    expect(crossSectionDent(map).max).toBeCloseTo(0, 6);
  });

  it('凸な面（円柱）は、凹みが 0', () => {
    // 半径 0.5 の円柱。手前が凸なので凹みは出ない。
    const { data, count } = surface((u) => -Math.sqrt(Math.max(0, 0.25 - (u - 0.5) ** 2)));
    const map = frontDepthMap(data, count, SIDE);
    const dent = crossSectionDent(map);
    expect(dent.max, `凸なのに凹みが ${dent.max.toFixed(4)} 出ています`).toBeLessThan(0.01);
  });

  it('溝の深さを体幅比で当てる', () => {
    // 体幅 1.0 の面の中央に、深さ 0.20 の V 溝を掘る。
    const depth = 0.2;
    const { data, count } = surface((u) => (Math.abs(u - 0.5) < 0.15 ? depth * (1 - Math.abs(u - 0.5) / 0.15) : 0));
    const map = frontDepthMap(data, count, SIDE);
    const dent = crossSectionDent(map);
    expect(dent.max, `凹み ${dent.max.toFixed(3)}（期待 0.20 前後）`).toBeGreaterThan(0.17);
    expect(dent.max).toBeLessThan(0.23);
  });

  it('手前への峰の高さを体幅比で当てる', () => {
    // 幅の 6% の細い峰を、体幅の 0.10 ぶん手前へ出す。
    const { data, count } = surface((u) => (Math.abs(u - 0.5) < 0.03 ? -0.1 : 0));
    const map = frontDepthMap(data, count, SIDE);
    const b = crossSectionBulge(map);
    expect(b.max, `飛び出し ${b.max.toFixed(3)}（期待 0.10 弱）`).toBeGreaterThan(0.06);
    expect(b.max).toBeLessThan(0.12);
  });

  it('尺度に依らない（全体を 3 倍しても同じ数）', () => {
    const shape = (u: number): number => (Math.abs(u - 0.5) < 0.15 ? 0.2 * (1 - Math.abs(u - 0.5) / 0.15) : 0);
    const one = surface(shape);
    const a = frontDepthMap(one.data, one.count, SIDE);
    const big = surface((u) => shape(u) * 3);
    const b = frontDepthMap(big.data, big.count, SIDE);
    expect(crossSectionDent(a).max * 3).toBeCloseTo(crossSectionDent(b).max, 2);
  });

  it('短すぎる区間は数に入れない', () => {
    // 幅 10 画素ぶんしか無い面。既定の minRun(40) には届かない。
    const { data, count } = surface((u) => (Math.abs(u - 0.5) < 0.15 ? 0.2 : 0), 30);
    const map = frontDepthMap(data, count, 30);
    expect(crossSectionDent(map).samples).toBe(0);
  });
});
