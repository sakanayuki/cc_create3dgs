/**
 * ⑦ 適応サンプリング（docs/04 §4.3.1 ②、docs/05 §5.2.3）。
 *
 * 最重要は「書き出し→読み直しで同じ分割になること」。`.pgs` はセルの情報を
 * 保存せず読込時に引き直すので、ここが崩れると復元したものが別物になる。
 */
import { describe, expect, it } from 'vitest';
import {
  adaptiveSample,
  estimateReduction,
  flattenToPlanes,
  REDUCTION_TARGETS,
  SAMPLING_PRESETS,
  samplingReduction,
  solveSamplingParams,
} from '../../src/pipeline/7-sample';

const S = 64;

interface Planes {
  depth: Float32Array;
  color: Uint8ClampedArray;
  alpha: Uint8ClampedArray;
}

function blank(size = S): Planes {
  return {
    depth: new Float32Array(size * size),
    color: new Uint8ClampedArray(size * size * 4),
    alpha: new Uint8ClampedArray(size * size),
  };
}

/** 決定的な擬似乱数。テストが実行ごとに変わらないようにする。 */
function rng(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 0x100000000;
  };
}

describe('分割の基準', () => {
  it('一様な面は最大セルに統合される', () => {
    const p = blank();
    p.alpha.fill(255);
    p.depth.fill(0.5);
    for (let i = 0; i < S * S; i++) {
      p.color[i * 4] = 100;
      p.color[i * 4 + 1] = 120;
      p.color[i * 4 + 2] = 140;
      p.color[i * 4 + 3] = 255;
    }
    const map = adaptiveSample(p.depth, p.color, p.alpha, S, S);
    expect(map.cellCount).toBe((S / 8) * (S / 8)); // 8×8 セルで埋まる
    expect(samplingReduction(map)).toBeCloseTo(1 - 1 / 64, 6);
  });

  it('市松模様は統合されない（群間分散が下の階層で立つ）', () => {
    const p = blank();
    p.alpha.fill(255);
    for (let y = 0; y < S; y++) {
      for (let x = 0; x < S; x++) {
        const i = y * S + x;
        p.depth[i] = (x + y) % 2 === 0 ? 0.4 : 0.6;
        const c = (x + y) % 2 === 0 ? 40 : 220;
        p.color[i * 4] = c;
        p.color[i * 4 + 1] = c;
        p.color[i * 4 + 2] = c;
        p.color[i * 4 + 3] = 255;
      }
    }
    const map = adaptiveSample(p.depth, p.color, p.alpha, S, S);
    expect(map.cellCount).toBe(S * S); // 1画素ずつ残る
    expect(samplingReduction(map)).toBe(0);
  });

  it('高品質プリセットは統合しない', () => {
    const p = blank();
    p.alpha.fill(255);
    p.depth.fill(0.5);
    const map = adaptiveSample(p.depth, p.color, p.alpha, S, S, SAMPLING_PRESETS.high);
    expect(map.cellCount).toBe(S * S);
  });

  it('シルエットをまたぐセルは1画素まで割る', () => {
    const p = blank();
    p.depth.fill(0.5);
    // 左半分だけ被写体。境界は x = 32。
    for (let y = 0; y < S; y++) for (let x = 0; x < 32; x++) p.alpha[y * S + x] = 255;

    const map = adaptiveSample(p.depth, p.color, p.alpha, S, S);
    // 境界に接するセル（x = 24..31 の列）は 8×8 のまま統合されてよい。
    // またいでいるセルが1つも無いことを確かめる。
    for (let c = 0; c < map.cellCount; c++) {
      const x0 = map.x[c] as number;
      const y0 = map.y[c] as number;
      const sz = map.size[c] as number;
      let inside = 0;
      let outside = 0;
      for (let y = y0; y < y0 + sz; y++) {
        for (let x = x0; x < x0 + sz; x++) {
          if ((p.alpha[y * S + x] as number) >= 128) inside++;
          else outside++;
        }
      }
      expect(inside > 0 && outside > 0, `セル ${c} がシルエットをまたいでいます`).toBe(false);
    }
  });

  it('被写体の外にはセルを作らない', () => {
    const p = blank();
    p.depth.fill(0.5);
    for (let y = 20; y < 40; y++) for (let x = 20; x < 40; x++) p.alpha[y * S + x] = 255;
    const map = adaptiveSample(p.depth, p.color, p.alpha, S, S);
    expect(map.cellId[0]).toBe(-1);
    expect(map.subjectPixels).toBe(400);
  });
});

describe('書き出し→読み直しの一致（docs/05 §5.2.3）', () => {
  /**
   * これが崩れると、`.pgs` に書いたガウシアン集合と、読み込んで復元される
   * ガウシアン集合が別物になる。乱数で作った 40 枚で確かめる。
   */
  it('乱数で作った画像でも分割が完全に一致する', () => {
    for (let seed = 1; seed <= 40; seed++) {
      const r = rng(seed);
      const p = blank(32);
      const size = 32;
      // 滑らかな面と細かい模様が混ざった画像
      for (let y = 0; y < size; y++) {
        for (let x = 0; x < size; x++) {
          const i = y * size + x;
          p.alpha[i] = r() < 0.85 ? 255 : 0;
          p.depth[i] = 0.3 + 0.4 * Math.sin(x * 0.2 + seed) + 0.01 * r();
          const base = 128 + 60 * Math.cos(y * 0.15 + seed);
          const c = base + (r() < 0.3 ? 20 * r() : 0);
          p.color[i * 4] = c;
          p.color[i * 4 + 1] = c * 0.9;
          p.color[i * 4 + 2] = c * 1.1;
          p.color[i * 4 + 3] = 255;
        }
      }

      const first = adaptiveSample(p.depth, p.color, p.alpha, size, size);
      flattenToPlanes(first, p.color, p.alpha, p.depth);
      const second = adaptiveSample(p.depth, p.color, p.alpha, size, size);

      expect(second.cellCount, `seed ${seed}: セル数が違います`).toBe(first.cellCount);
      expect(Array.from(second.size), `seed ${seed}: セルの大きさが違います`).toEqual(
        Array.from(first.size),
      );
      expect(Array.from(second.x), `seed ${seed}: セルの位置が違います`).toEqual(
        Array.from(first.x),
      );
    }
  });

  it('群内分散だけが大きいセルでも一致する（旧基準が壊れた反例）', () => {
    // 群内 0.9θ・群間 0.6θ。全分散なら 1.5θ > θ で分割されるが、
    // 統合後は 0.6θ ≤ θ になって統合されてしまう組み合わせ。
    const th = SAMPLING_PRESETS.standard.depthVar;
    const within = Math.sqrt(0.9 * th);
    const between = Math.sqrt(0.6 * th);
    const size = 8;
    const p = blank(size);
    p.alpha.fill(255);
    for (let y = 0; y < size; y++) {
      for (let x = 0; x < size; x++) {
        const i = y * size + x;
        const qMean = 0.5 + ((x >> 2) === (y >> 2) ? between : -between);
        p.depth[i] = qMean + ((x + y) % 2 === 0 ? within : -within);
        p.color[i * 4] = 128;
        p.color[i * 4 + 1] = 128;
        p.color[i * 4 + 2] = 128;
        p.color[i * 4 + 3] = 255;
      }
    }
    const first = adaptiveSample(p.depth, p.color, p.alpha, size, size);
    flattenToPlanes(first, p.color, p.alpha, p.depth);
    const second = adaptiveSample(p.depth, p.color, p.alpha, size, size);
    expect(Array.from(second.size)).toEqual(Array.from(first.size));
  });
});

describe('統合率', () => {
  it('プリセットが強いほどよく統合する', () => {
    const p = blank();
    p.alpha.fill(255);
    const r = rng(7);
    for (let y = 0; y < S; y++) {
      for (let x = 0; x < S; x++) {
        const i = y * S + x;
        p.depth[i] = 0.5 + 0.004 * Math.sin(x * 0.4) + 0.002 * r();
        const c = 128 + 20 * Math.sin(y * 0.3) + 6 * r();
        p.color[i * 4] = c;
        p.color[i * 4 + 1] = c;
        p.color[i * 4 + 2] = c;
        p.color[i * 4 + 3] = 255;
      }
    }
    const light = samplingReduction(adaptiveSample(p.depth, p.color, p.alpha, S, S, SAMPLING_PRESETS.light));
    const std = samplingReduction(adaptiveSample(p.depth, p.color, p.alpha, S, S, SAMPLING_PRESETS.standard));
    const high = samplingReduction(adaptiveSample(p.depth, p.color, p.alpha, S, S, SAMPLING_PRESETS.high));
    expect(light).toBeGreaterThanOrEqual(std);
    expect(std).toBeGreaterThanOrEqual(high);
    expect(high).toBe(0);
  });
});

describe('統合率から閾値を逆算する', () => {
  /** 楕円体の人物風。滑らかな陰影に細かいざらつきを載せる。 */
  function portrait(size = 128): Planes {
    const p = blank(size);
    const r = rng(12345);
    for (let y = 0; y < size; y++) {
      for (let x = 0; x < size; x++) {
        const i = y * size + x;
        const dx = (x - size / 2) / (size * 0.3);
        const dy = (y - size / 2) / (size * 0.4);
        const rr = dx * dx + dy * dy;
        if (rr > 1) continue;
        p.alpha[i] = 255;
        p.depth[i] = 0.5 - 0.25 * Math.sqrt(1 - rr) + 0.004 * Math.sin(x * 0.9) * Math.sin(y * 0.8);
        const shade = 150 + 70 * (1 - rr) + 8 * (r() - 0.5);
        p.color[i * 4] = shade;
        p.color[i * 4 + 1] = shade * 0.85;
        p.color[i * 4 + 2] = shade * 0.78;
        p.color[i * 4 + 3] = 255;
      }
    }
    return p;
  }

  it('プリセットの目標統合率に当たる', () => {
    const p = portrait();
    const size = 128;
    for (const target of [REDUCTION_TARGETS.light, REDUCTION_TARGETS.standard]) {
      const params = solveSamplingParams(p.depth, p.color, p.alpha, size, size, target);
      const got = samplingReduction(adaptiveSample(p.depth, p.color, p.alpha, size, size, params));
      // 見積りはタイルを間引いているので、全走査とは数%ずれる
      expect(Math.abs(got - target), `目標 ${target} に対して実測 ${got}`).toBeLessThan(0.06);
    }
  });

  it('目標 0 なら統合しない設定を返す', () => {
    const p = portrait(64);
    const params = solveSamplingParams(p.depth, p.color, p.alpha, 64, 64, 0);
    const map = adaptiveSample(p.depth, p.color, p.alpha, 64, 64, params);
    expect(samplingReduction(map)).toBe(0);
  });

  it('逆算した閾値でも書き出し→読み直しが一致する', () => {
    const size = 64;
    const p = portrait(size);
    const params = solveSamplingParams(p.depth, p.color, p.alpha, size, size, 0.3);
    const first = adaptiveSample(p.depth, p.color, p.alpha, size, size, params);
    flattenToPlanes(first, p.color, p.alpha, p.depth);
    const second = adaptiveSample(p.depth, p.color, p.alpha, size, size, params);
    expect(Array.from(second.size)).toEqual(Array.from(first.size));
  });

  it('見積りは全走査とおおむね一致する', () => {
    const size = 128;
    const p = portrait(size);
    const full = samplingReduction(
      adaptiveSample(p.depth, p.color, p.alpha, size, size, SAMPLING_PRESETS.standard),
    );
    const est = estimateReduction(p.depth, p.color, p.alpha, size, size, SAMPLING_PRESETS.standard, 4);
    expect(Math.abs(est - full)).toBeLessThan(0.1);
  });
});
