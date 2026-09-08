/**
 * ① 被写体抽出の後処理（docs/03 §3.3）。
 *
 * ここで見るのは「4つの処理が、狙った副作用だけを起こすか」。
 * 特に、髪の毛のような本物の半透明部分を潰していないかを確かめる。
 */
import { describe, expect, it } from 'vitest';
import {
  applyMatteGate,
  boxFilter,
  connectedComponents,
  fillInteriorHoles,
  guidedFilterColor,
  refineMatte,
  removeSmallComponents,
  softenEdge,
} from '../../src/pipeline/1-matte';

const SIZE = 32;

/** 中央に正方形の被写体があるマット。 */
function squareAlpha(size = SIZE, half = 8): Uint8ClampedArray {
  const a = new Uint8ClampedArray(size * size);
  const c = size / 2;
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      if (Math.abs(x - c) < half && Math.abs(y - c) < half) a[y * size + x] = 255;
    }
  }
  return a;
}

/** α と同じ形を白、背景を黒にしたガイド画像。 */
function guideFrom(alpha: ArrayLike<number>, size = SIZE): Uint8ClampedArray {
  const rgb = new Uint8ClampedArray(size * size * 4);
  for (let i = 0; i < size * size; i++) {
    const v = (alpha[i] as number) >= 128 ? 230 : 20;
    rgb[i * 4] = v;
    rgb[i * 4 + 1] = v;
    rgb[i * 4 + 2] = v;
    rgb[i * 4 + 3] = 255;
  }
  return rgb;
}

describe('箱フィルタ', () => {
  it('一様な入力は変わらない（端でも）', () => {
    const src = new Float32Array(SIZE * SIZE).fill(7);
    const out = boxFilter(src, SIZE, SIZE, 4);
    for (let i = 0; i < out.length; i++) expect(out[i]).toBeCloseTo(7, 5);
  });

  it('総和を保つ（面積 × 平均）', () => {
    const src = new Float32Array(SIZE * SIZE);
    src[SIZE * 10 + 10] = 100;
    const out = boxFilter(src, SIZE, SIZE, 2);
    // 半径2の窓は 5×5 = 25 画素。中心から ±2 の範囲に 100/25 ずつ散る。
    expect(out[SIZE * 10 + 10]).toBeCloseTo(4, 5);
    expect(out[SIZE * 10 + 13]).toBeCloseTo(0, 5);
  });
});

describe('ガイデッドフィルタ', () => {
  it('ガイドと一致する α はほぼそのまま残る', () => {
    const alpha = squareAlpha();
    const out = guidedFilterColor(alpha, guideFrom(alpha), SIZE, SIZE, { radius: 4, eps: 1e-4 });
    const c = SIZE / 2;
    expect(out[c * SIZE + c]).toBeGreaterThan(240);
    expect(out[0]).toBeLessThan(15);
  });

  it('ぼけた α の境界を、ガイドの輪郭に合わせて締める', () => {
    // マット推定モデルは縮小した入力で推論するので、返る α の境界は数画素ぼける。
    // ガイド（＝写真そのもの）は鋭い。このぼけを締めるのがこのフィルタの仕事。
    const truth = squareAlpha();
    const blurred = new Uint8ClampedArray(SIZE * SIZE);
    const b = boxFilter(truth, SIZE, SIZE, 3);
    for (let i = 0; i < blurred.length; i++) blurred[i] = Math.round(b[i] as number);

    const out = guidedFilterColor(blurred, guideFrom(truth), SIZE, SIZE, { radius: 4, eps: 1e-4 });

    // 中央の行で 10%〜90% を横切る幅を測る。狭いほど境界が締まっている。
    const transitionWidth = (m: ArrayLike<number>): number => {
      const row = SIZE / 2;
      let lo = -1;
      let hi = -1;
      for (let x = 0; x < SIZE / 2; x++) {
        const v = m[row * SIZE + x] as number;
        if (lo < 0 && v > 25) lo = x;
        if (hi < 0 && v > 230) hi = x;
      }
      return hi - lo;
    };
    expect(transitionWidth(out)).toBeLessThan(transitionWidth(blurred));
  });

  it('ガイドが背景と言い切る側へ α を漏らさない', () => {
    // 縁の半透明が背景側へ伸びると、そこにガウシアンが立って
    // 被写体から離れた薄い膜になる（docs/03 §3.5.3 の rubber sheet と同じ問題）。
    const truth = squareAlpha();
    const fat = new Uint8ClampedArray(SIZE * SIZE);
    for (let y = 0; y < SIZE; y++) {
      for (let x = 0; x < SIZE; x++) {
        // 真の被写体より 3px 太った α（背景側へはみ出している）
        const c = SIZE / 2;
        if (Math.abs(x - c) < 11 && Math.abs(y - c) < 11) fat[y * SIZE + x] = 255;
      }
    }
    const out = guidedFilterColor(fat, guideFrom(truth), SIZE, SIZE, { radius: 4, eps: 1e-4 });
    const row = SIZE / 2;
    // 真の輪郭より外側（左）では、はみ出しが十分に削られていること
    expect(out[row * SIZE + (SIZE / 2 - 10)] as number).toBeLessThan(
      fat[row * SIZE + (SIZE / 2 - 10)] as number,
    );
  });
});

describe('連結成分', () => {
  it('離れた2つの塊を別ラベルにする', () => {
    const a = new Uint8ClampedArray(SIZE * SIZE);
    a[SIZE * 2 + 2] = 255;
    a[SIZE * 20 + 20] = 255;
    const { sizes } = connectedComponents(a, SIZE, SIZE);
    expect(sizes.length).toBe(3); // [0番は未使用, 塊A, 塊B]
    expect(sizes[1]).toBe(1);
    expect(sizes[2]).toBe(1);
  });

  it('斜めに接する画素は同じ塊とみなす（8近傍）', () => {
    const a = new Uint8ClampedArray(SIZE * SIZE);
    a[SIZE * 5 + 5] = 255;
    a[SIZE * 6 + 6] = 255;
    const { sizes } = connectedComponents(a, SIZE, SIZE);
    expect(sizes.length).toBe(2);
    expect(sizes[1]).toBe(2);
  });
});

describe('小成分除去', () => {
  it('最大成分の 2% 未満の島を消し、本体は残す', () => {
    const a = squareAlpha(); // 16×16 = 256 画素
    a[SIZE * 1 + 1] = 255; // 1 画素の島（0.4%）
    const out = removeSmallComponents(a, SIZE, SIZE, 0.02);
    expect(out[SIZE * 1 + 1]).toBe(0);
    expect(out[SIZE * (SIZE / 2) + SIZE / 2]).toBe(255);
  });

  it('閾値以上の島は残す', () => {
    const a = squareAlpha();
    // 3×3 = 9 画素は 256 の 3.5% なので残る
    for (let y = 1; y <= 3; y++) for (let x = 1; x <= 3; x++) a[y * SIZE + x] = 255;
    const out = removeSmallComponents(a, SIZE, SIZE, 0.02);
    expect(out[SIZE * 2 + 2]).toBe(255);
  });
});

describe('内部の穴埋め', () => {
  it('被写体の中の穴を埋め、外の背景は触らない', () => {
    const a = squareAlpha();
    const c = SIZE / 2;
    a[c * SIZE + c] = 0;
    a[c * SIZE + c + 1] = 0;
    const out = fillInteriorHoles(a, SIZE, SIZE);
    expect(out[c * SIZE + c]).toBe(255);
    expect(out[c * SIZE + c + 1]).toBe(255);
    expect(out[0]).toBe(0);
  });

  it('外へ通じた切り欠きは埋めない', () => {
    const a = squareAlpha();
    const c = SIZE / 2;
    // 左辺から中心まで、幅1の溝を掘る
    for (let x = 0; x <= c; x++) a[c * SIZE + x] = 0;
    const out = fillInteriorHoles(a, SIZE, SIZE);
    expect(out[c * SIZE + c]).toBe(0);
  });
});

describe('輪郭の軟化', () => {
  it('階段状の輪郭を均す', () => {
    const a = squareAlpha();
    const out = softenEdge(a, SIZE, SIZE, 2);
    const c = SIZE / 2;
    // 境界（x = c - 8）付近に中間値ができる
    const edge = out[c * SIZE + (c - 8)] as number;
    expect(edge).toBeGreaterThan(0);
    expect(edge).toBeLessThan(255);
  });

  it('被写体の中心と遠い背景は触らない', () => {
    const a = squareAlpha();
    const out = softenEdge(a, SIZE, SIZE, 2);
    const c = SIZE / 2;
    expect(out[c * SIZE + c]).toBe(255);
    expect(out[0]).toBe(0);
  });
});

describe('後処理ひとまとめ', () => {
  it('島を消し穴を埋めても、本体の形は保たれる', () => {
    const truth = squareAlpha();
    const a = new Uint8ClampedArray(truth);
    a[SIZE * 1 + 1] = 255; // 消えてほしい島
    const c = SIZE / 2;
    a[c * SIZE + c] = 0; // 埋まってほしい穴

    const out = refineMatte(a, guideFrom(truth), SIZE, SIZE);
    expect(out[SIZE * 1 + 1]).toBeLessThan(128);
    expect(out[c * SIZE + c]).toBeGreaterThan(200);

    // 面積が大きく変わっていないこと（±15%）
    const area = (m: ArrayLike<number>): number => {
      let n = 0;
      for (let i = 0; i < SIZE * SIZE; i++) if ((m[i] as number) >= 128) n++;
      return n;
    };
    const ratio = area(out) / area(truth);
    expect(ratio).toBeGreaterThan(0.85);
    expect(ratio).toBeLessThan(1.15);
  });
});

describe('領域の門でマットを絞る（v2.5、実写での破綻にもとづく）', () => {
  const W = 40;
  const H = 40;

  /** 人物（左）と、それに触れていない敷物（右下）。 */
  function scene(): { fine: Uint8ClampedArray; gate: Uint8ClampedArray } {
    const fine = new Uint8ClampedArray(W * H);
    const gate = new Uint8ClampedArray(W * H);
    for (let y = 6; y < 26; y++) {
      for (let x = 6; x < 16; x++) {
        fine[y * W + x] = 255;
        gate[y * W + x] = 255;
      }
    }
    // 敷物: 輪郭モデルだけが被写体と誤認する。門は認めない。
    for (let y = 28; y < 36; y++) for (let x = 20; x < 36; x++) fine[y * W + x] = 254;
    return { fine, gate };
  }

  it('門が認めない領域を落とす', () => {
    const { fine, gate } = scene();
    const out = applyMatteGate(fine, gate, W, H, 0);
    expect(out[30 * W + 28], '敷物が残っています').toBe(0);
    expect(out[15 * W + 10], '人物が消えています').toBe(255);
  });

  it('門の中では輪郭モデルの値をそのまま通す（半透明も保つ）', () => {
    const { fine, gate } = scene();
    // 髪のような半端な α。門は 255 なので触られないはず。
    fine[10 * W + 7] = 90;
    const out = applyMatteGate(fine, gate, W, H, 0);
    expect(out[10 * W + 7]).toBe(90);
  });

  it('門を膨らませるので、輪郭モデルのはみ出しぶんを削らない', () => {
    const { fine, gate } = scene();
    // 輪郭モデルは門より 2px 外まで被写体を取る（髪の生え際など）。
    for (let y = 6; y < 26; y++) for (let x = 4; x < 6; x++) fine[y * W + x] = 200;

    const tight = applyMatteGate(fine, gate, W, H, 0);
    expect(tight[15 * W + 4], '膨らませないと縁が削れる').toBe(0);

    const grown = applyMatteGate(fine, gate, W, H, 4);
    expect(grown[15 * W + 4], '膨らませた門は縁を残す').toBe(200);
    expect(grown[30 * W + 28], '膨らませても敷物までは届かない').toBe(0);
  });

  it('門が全面を認めるなら何も変わらない', () => {
    const { fine } = scene();
    const all = new Uint8ClampedArray(W * H).fill(255);
    const out = applyMatteGate(fine, all, W, H, 4);
    for (let i = 0; i < out.length; i++) expect(out[i]).toBe(fine[i]);
  });
});
