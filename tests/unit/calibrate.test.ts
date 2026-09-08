/**
 * 深度較正（docs/03 §3.5）。
 *
 * 検証の要は「シルエット法線の事前分布からシフト b を推定できるか」。
 * 真の深度が分かっている合成球を作り、既知の a と b で逆深度に変換してから
 * 推定器に渡し、b を取り戻せるかを見る。
 */
import { describe, expect, it } from 'vitest';
import {
  calibrate,
  calmSilhouetteRim,
  enhanceRelief,
  flattenImplausibleBands,
  estimateNormals,
  solveAffine,
  pullBoundaryDepthInward,
  silhouetteBand,
  subjectShortSide,
} from '../../src/pipeline/3-calibrate';

const SIZE = 96;
const FOCAL = 92; // SIZE 相当の画角

/** 中心 (0,0,z0)・半径 r の球をピンホールカメラで見たときの深度とマット。 */
function sphereScene(z0 = 1.0, r = 0.28, size = SIZE, focal = FOCAL) {
  const depth = new Float32Array(size * size);
  const alpha = new Uint8ClampedArray(size * size);
  const c = size / 2;
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const i = y * size + x;
      // 視線 d = ((x−c)/f, (y−c)/f, 1) を正規化せずに使う
      const dx = (x - c) / focal;
      const dy = (y - c) / focal;
      // |t·d − (0,0,z0)|² = r²  →  t²|d|² − 2·t·z0 + z0² − r² = 0
      const A = dx * dx + dy * dy + 1;
      const B = -2 * z0;
      const C = z0 * z0 - r * r;
      const disc = B * B - 4 * A * C;
      if (disc < 0) {
        depth[i] = z0 + r * 1.6; // 背景は奥に置く
        alpha[i] = 0;
      } else {
        const t = (-B - Math.sqrt(disc)) / (2 * A);
        depth[i] = t; // 手前側の交点までの距離（z 成分は t·1 = t）
        alpha[i] = 255;
      }
    }
  }
  return { depth, alpha, size, focal, z0, r };
}

/**
 * 局所強調（③）の平滑化半径。calibrate と同じ式で出す。
 * 球の投影直径 = 2·r·focal（この場面では 51.5px）に 0.04 を掛けて丸める。
 */
const RELIEF_BLUR_PX = Math.max(1, Math.round(2 * 0.28 * FOCAL * 0.04));

/**
 * 被写体マスクを k 画素だけ内側へ削り、「平滑化窓がマスクに切られない画素」を返す。
 * 縁の非対称な窓の影響を切り分けるために使う。
 */
function eroded(
  alpha: ArrayLike<number>,
  width: number,
  height: number,
  k: number,
): Uint8Array {
  const out = new Uint8Array(width * height);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      if ((alpha[y * width + x] as number) < 128) continue;
      let ok = 1;
      for (let dy = -k; dy <= k && ok; dy++) {
        for (let dx = -k; dx <= k; dx++) {
          const nx = x + dx;
          const ny = y + dy;
          if (nx < 0 || ny < 0 || nx >= width || ny >= height) { ok = 0; break; }
          if ((alpha[ny * width + nx] as number) < 128) { ok = 0; break; }
        }
      }
      out[y * width + x] = ok;
    }
  }
  return out;
}

describe('被写体の短辺', () => {
  it('球の投影直径をおおよそ返す', () => {
    const { alpha, size, focal, z0, r } = sphereScene();
    const got = subjectShortSide(alpha, size, size);
    const expected = (2 * r * focal) / z0;
    expect(got).toBeGreaterThan(expected * 0.85);
    expect(got).toBeLessThan(expected * 1.2);
  });

  it('被写体が無ければ 0', () => {
    expect(subjectShortSide(new Uint8ClampedArray(64), 8, 8)).toBe(0);
  });
});

describe('法線推定（5×5 平面フィット）', () => {
  // 向きの規約: カメラを原点・視線を +z としたとき、手前を向いた面の法線は −z 成分を持つ。
  // 描画の背面カリングが dot(n, eye − pos) > 閾値 で判定するので、この向きでなければならない。

  it('視線に垂直な平面では法線が視点の側（−z）を向く', () => {
    const n = 48;
    const depth = new Float32Array(n * n).fill(1.0);
    const normals = estimateNormals(depth, n, n, FOCAL, 0.05);
    const i = (24 * n + 24) * 3;
    expect(normals[i + 2]).toBeLessThan(-0.99);
  });

  it('傾いた平面で勾配に応じて法線が傾く', () => {
    const n = 48;
    const depth = new Float32Array(n * n);
    for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) depth[y * n + x] = 1.0 + 0.004 * x;
    const normals = estimateNormals(depth, n, n, FOCAL, 0.5);
    const i = (24 * n + 24) * 3;
    // x 方向に深くなる面。外向き法線は +x 側へ傾く
    expect(normals[i]).toBeGreaterThan(0.1);
    expect(normals[i + 2]).toBeLessThan(0);
  });

  it('球の表面で法線が球の外を向く', () => {
    const { depth, size, focal } = sphereScene();
    const normals = estimateNormals(depth, size, size, focal, 0.05);
    const c = size / 2;
    const i = (c * size + (c + 12)) * 3;
    expect(normals[i]).toBeGreaterThan(0.15);
    expect(normals[i + 2]).toBeLessThan(0);
    expect(Math.hypot(normals[i] as number, normals[i + 1] as number, normals[i + 2] as number))
      .toBeCloseTo(1, 4);
  });

  it('球の左右で法線の x 符号が反転し、対称になる', () => {
    const { depth, size, focal } = sphereScene();
    const normals = estimateNormals(depth, size, size, focal, 0.05);
    const c = size / 2;
    const right = normals[(c * size + (c + 12)) * 3] as number;
    const left = normals[(c * size + (c - 12)) * 3] as number;
    expect(right).toBeGreaterThan(0);
    expect(left).toBeLessThan(0);
    expect(Math.abs(right + left)).toBeLessThan(0.05);
  });

  it('深度の不連続をまたぐ画素をフィットから外す', () => {
    // 左右で深度が大きく違う階段。境界で法線が横倒しにならないこと
    const n = 48;
    const depth = new Float32Array(n * n);
    for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) depth[y * n + x] = x < 24 ? 1.0 : 1.5;
    const normals = estimateNormals(depth, n, n, FOCAL, 0.05);
    const i = (24 * n + 23) * 3;
    expect(normals[i + 2]).toBeLessThan(-0.9);
  });
});

describe('シルエット帯', () => {
  it('境界から内側 3〜8px の画素だけを拾う', () => {
    const { alpha, size } = sphereScene();
    const band = silhouetteBand(alpha, size, size, 3, 8);
    expect(band.length).toBeGreaterThan(100);
    // 拾った画素はすべて被写体内部
    for (const i of band) expect(alpha[i]).toBe(255);
    // 中心（最も内側）は含まれない
    expect([...band]).not.toContain((size / 2) * size + size / 2);
  });
});

describe('逆深度の係数を閉形式で解く（solveAffine）', () => {
  it('真の (a, b) を誤差なく復元する', () => {
    const { depth, alpha, size } = sphereScene();
    const a = 0.8;
    const b = 0.35;

    // 真の深度から逆深度を作る: d = (1/z − b) / a
    const inv: number[] = [];
    const z: number[] = [];
    for (let i = 0; i < depth.length; i++) {
      if ((alpha[i] as number) < 128) continue;
      inv.push((1 / (depth[i] as number) - b) / a);
      z.push(depth[i] as number);
    }
    inv.sort((p, q) => p - q);
    z.sort((p, q) => p - q);
    const q = (arr: number[], t: number) => arr[Math.round(t * (arr.length - 1))] as number;

    // 中央値を 1 に正規化した尺度で目標幅を作る
    const scale = 1 / q(z, 0.5);
    const targetSpan = (q(z, 0.95) - q(z, 0.05)) * scale;

    const got = solveAffine(q(inv, 0.05), q(inv, 0.5), q(inv, 0.95), targetSpan);
    expect(got).not.toBeNull();
    expect(got!.a).toBeCloseTo(a / scale, 4);
    expect(got!.b).toBeCloseTo(b / scale, 4);
  });

  it('中央値が 1.0、幅が目標値になる', () => {
    const d5 = 1.0;
    const d50 = 1.2;
    const d95 = 1.3;
    for (const S of [0.1, 0.2, 0.35]) {
      const r = solveAffine(d5, d50, d95, S);
      expect(r, `S=${S} で解が無い`).not.toBeNull();
      const z = (d: number) => 1 / (r!.a * d + r!.b);
      expect(z(d50)).toBeCloseTo(1.0, 6);
      expect(z(d5) - z(d95)).toBeCloseTo(S, 6);
    }
  });

  it('立体感スライダの全域（0.3〜1.2）で解が存在する', () => {
    const { depth, alpha, size, focal } = sphereScene();
    const inv: number[] = [];
    for (let i = 0; i < depth.length; i++) {
      if ((alpha[i] as number) >= 128) inv.push(1 / (depth[i] as number));
    }
    inv.sort((p, q) => p - q);
    const q = (t: number) => inv[Math.round(t * (inv.length - 1))] as number;
    const shortSide = subjectShortSide(alpha, size, size);

    for (const ratio of [0.3, 0.65, 1.2]) {
      const S = (shortSide / focal) * ratio;
      expect(solveAffine(q(0.05), q(0.5), q(0.95), S), `ratio=${ratio}`).not.toBeNull();
    }
  });

  it('解が存在しない入力では null を返す（例外にしない）', () => {
    expect(solveAffine(1, 1, 1, 0.1)).toBeNull(); // 幅ゼロ
    expect(solveAffine(1, 1.1, 1.2, 0)).toBeNull(); // 目標幅ゼロ
    expect(solveAffine(1.2, 1.1, 1.0, 0.1)).toBeNull(); // 順序が逆
  });

  it('深度がすべて正になる解だけを返す', () => {
    for (const S of [0.05, 0.1, 0.3, 0.8]) {
      const r = solveAffine(0.9, 1.1, 1.4, S);
      if (!r) continue;
      expect(r.a * 0.9 + r.b).toBeGreaterThan(0);
      expect(r.a * 1.4 + r.b).toBeGreaterThan(0);
    }
  });
});

describe('較正（全体）', () => {
  it('深度出力（DA3 経路）で 0..65535 に正規化する', () => {
    const { depth, alpha, size, focal } = sphereScene();
    const r = calibrate({ raw: depth, width: size, height: size, alpha, kind: 'depth', focalPx: focal });
    expect(r.shift).toBe(0);
    expect(r.farZ).toBeGreaterThan(r.nearZ);
    let min = 65535;
    let max = 0;
    for (let i = 0; i < r.depth.length; i++) {
      if ((alpha[i] as number) < 128) continue;
      min = Math.min(min, r.depth[i] as number);
      max = Math.max(max, r.depth[i] as number);
    }
    expect(min).toBeLessThan(3000);
    expect(max).toBeGreaterThan(62000);
  });

  it('奥行き比（立体感スライダ）が深度レンジに反映される', () => {
    const { depth, alpha, size, focal } = sphereScene();
    const base = { raw: depth, width: size, height: size, alpha, kind: 'depth' as const, focalPx: focal };
    const flat = calibrate({ ...base, depthToWidthRatio: 0.3 });
    const deep = calibrate({ ...base, depthToWidthRatio: 1.2 });
    const span = (r: { nearZ: number; farZ: number }) => r.farZ - r.nearZ;
    expect(span(deep) / span(flat)).toBeGreaterThan(3);
  });

  it('逆深度出力（V2 経路）でも解けて正規化される', () => {
    const { depth, alpha, size, focal } = sphereScene();
    const inv = new Float32Array(depth.length);
    for (let i = 0; i < depth.length; i++) inv[i] = 1 / (depth[i] as number);
    const r = calibrate({ raw: inv, width: size, height: size, alpha, kind: 'inverse-depth', focalPx: focal });
    expect(Number.isFinite(r.shift)).toBe(true);
    expect(r.farZ).toBeGreaterThan(r.nearZ);
  });

  it('DA3 経路と V2 経路が同じ形を出す（同じ被写体・同じ立体感）', () => {
    // 同じ球を、深度として渡した場合と逆深度として渡した場合で
    // 正規化後の深度マップが一致するはず。どちらの経路も
    // 「中央値を 1、幅を目標値」に揃えているため。
    const { depth, alpha, size, focal } = sphereScene();
    const inv = new Float32Array(depth.length);
    for (let i = 0; i < depth.length; i++) inv[i] = 1 / (depth[i] as number);

    const base = { width: size, height: size, alpha, focalPx: focal, depthToWidthRatio: 0.65 };
    const a = calibrate({ ...base, raw: depth, kind: 'depth' });
    const b = calibrate({ ...base, raw: inv, kind: 'inverse-depth' });

    // 内部と縁を分けて見る。局所強調（③）の平滑化窓は、被写体の縁では
    // マスクに切られて非対称になる。深度経路（線形）と逆深度経路（1/x）は
    // 残差の形が少し違うので、その差が縁だけで拡大する。実測でも
    // 不一致は外周 10% のリングに全部入っていて、内部は桁違いに小さい。
    // そこで内部は厳しく、全体は分位で見る。
    const isInterior = eroded(alpha, size, size, RELIEF_BLUR_PX);

    let worstInside = 0;
    let subjectPixels = 0;
    let disagreeing = 0;
    for (let i = 0; i < depth.length; i++) {
      if ((alpha[i] as number) < 128) continue;
      const d = Math.abs((a.depth[i] as number) - (b.depth[i] as number)) / 65535;
      subjectPixels++;
      if (d > 0.05) disagreeing++;
      if (isInterior[i]) worstInside = Math.max(worstInside, d);
    }

    // 被写体の内部では、2 つの経路はほぼ一致する（実測 0.009）。
    expect(worstInside).toBeLessThan(0.02);
    // ずれるのは縁の画素だけ（実測 8.7%）。この球は投影直径が 51px しか
    // 無いので、縁の帯が占める割合そのものが大きい。分位で見ると、この
    // 付近は分布が急で添字 1 つで値が跳ねるため、割合で見る。
    expect(disagreeing / subjectPixels).toBeLessThan(0.12);
  });

  it('被写体が無ければ理由を添えて止まる', () => {
    expect(() =>
      calibrate({
        raw: new Float32Array(64), width: 8, height: 8,
        alpha: new Uint8ClampedArray(64), kind: 'depth', focalPx: FOCAL,
      }),
    ).toThrow(/被写体が見つかりません/);
  });
});

describe('境界画素の深度の引き込み', () => {
  it('軟化境界（0 < α < 128）の深度を内部の値で置き換える', () => {
    const w = 8;
    const h = 1;
    const depth = Uint16Array.from([9999, 9999, 100, 200, 300, 9999, 9999, 9999]);
    //                               背景  軟化   内部  内部  内部  軟化  背景  背景
    const alpha = Uint8ClampedArray.from([0, 60, 255, 255, 255, 60, 0, 0]);
    const out = pullBoundaryDepthInward(depth, alpha, w, h);

    expect(out[1]).toBe(100); // 最も近い内部画素（index 2）
    expect(out[5]).toBe(300); // 最も近い内部画素（index 4）
    expect(out[0]).toBe(9999); // 完全な背景はそのまま
    expect(out[2]).toBe(100); // 内部はそのまま
  });

  it('α が完全な 0 の画素には触れない', () => {
    const depth = Uint16Array.from([5, 5, 5, 5]);
    const alpha = Uint8ClampedArray.from([0, 0, 255, 0]);
    const out = pullBoundaryDepthInward(depth, alpha, 4, 1);
    expect([...out]).toEqual([5, 5, 5, 5]);
  });
});

describe('実寸と局所起伏（v2.2、実写での測定にもとづく）', () => {
  const S = 64;
  const FOCAL = 80;

  /** 中央に立体的な被写体。手前に膨らみ、細かい起伏を持つ。 */
  function scene(): { raw: Float32Array; alpha: Uint8ClampedArray } {
    const raw = new Float32Array(S * S);
    const alpha = new Uint8ClampedArray(S * S);
    for (let y = 0; y < S; y++) {
      for (let x = 0; x < S; x++) {
        const i = y * S + x;
        const dx = (x - S / 2) / (S * 0.25);
        const dy = (y - S / 2) / (S * 0.4);
        const rr = dx * dx + dy * dy;
        if (rr > 1) {
          raw[i] = 2.0; // 背景は遠い
          continue;
        }
        alpha[i] = 255;
        // 実寸 1m 付近、膨らみ 0.1m、細かい起伏 0.005m
        raw[i] = 1.0 - 0.1 * Math.sqrt(1 - rr) + 0.005 * Math.sin(x * 0.9) * Math.sin(y * 0.8);
      }
    }
    return { raw, alpha };
  }

  it('比率を指定しなければ実寸をそのまま使う', () => {
    const { raw, alpha } = scene();
    const r = calibrate({ raw, width: S, height: S, alpha, kind: 'depth', focalPx: FOCAL });
    expect(r.metric).toBe(true);
    // 被写体の実際の奥行き（約 0.1m）に近い範囲になる。局所強調のぶん少し広がる。
    expect(r.farZ - r.nearZ).toBeGreaterThan(0.05);
    expect(r.farZ - r.nearZ).toBeLessThan(0.4);
  });

  it('比率を指定すれば従来どおり引き伸ばす', () => {
    const { raw, alpha } = scene();
    const r = calibrate({
      raw, width: S, height: S, alpha, kind: 'depth', focalPx: FOCAL, depthToWidthRatio: 0.65,
    });
    expect(r.metric).toBe(false);
  });

  it('被写体の外の外れ値に範囲を引っ張られない', () => {
    // 髪やマットの染み出しを模して、数画素だけ極端に遠くする
    const { raw, alpha } = scene();
    const clean = calibrate({ raw, width: S, height: S, alpha, kind: 'depth', focalPx: FOCAL });
    for (let i = 0; i < 40; i++) {
      const j = (S / 2) * S + 10 + i;
      if (alpha[j] === 255) raw[j] = 5.0; // 極端な外れ値
    }
    const dirty = calibrate({ raw, width: S, height: S, alpha, kind: 'depth', focalPx: FOCAL });
    const grow = (dirty.farZ - dirty.nearZ) / (clean.farZ - clean.nearZ);
    expect(grow, `外れ値で奥行きが ${grow.toFixed(1)} 倍に広がりました`).toBeLessThan(1.5);
  });

  it('局所強調が細かい起伏だけを持ち上げる', () => {
    const { raw, alpha } = scene();
    const flat = calibrate({
      raw, width: S, height: S, alpha, kind: 'depth', focalPx: FOCAL, reliefBoost: 1,
    });
    const boosted = calibrate({
      raw, width: S, height: S, alpha, kind: 'depth', focalPx: FOCAL, reliefBoost: 3,
    });

    // 細かい起伏（平滑化からのずれ）の大きさを比べる
    const detail = (r: { depth: Uint16Array; nearZ: number; farZ: number }): number => {
      const span = r.farZ - r.nearZ;
      let sum = 0;
      let n = 0;
      for (let y = 2; y < S - 2; y++) {
        for (let x = 2; x < S - 2; x++) {
          const i = y * S + x;
          if (alpha[i] !== 255) continue;
          const c = (r.depth[i] as number) / 65535;
          const avg =
            ((r.depth[i - 2] as number) + (r.depth[i + 2] as number) +
             (r.depth[i - 2 * S] as number) + (r.depth[i + 2 * S] as number)) / 4 / 65535;
          sum += Math.abs(c - avg) * span;
          n++;
        }
      }
      return n > 0 ? sum / n : 0;
    };
    expect(detail(boosted)).toBeGreaterThan(detail(flat) * 1.5);
  });

  it('奥行きが幅に対して人間離れしていたら引き戻す', () => {
    // 奥行きだけ極端に大きい被写体を作る
    const raw = new Float32Array(S * S);
    const alpha = new Uint8ClampedArray(S * S);
    for (let y = 20; y < 44; y++) {
      for (let x = 28; x < 36; x++) {
        const i = y * S + x;
        alpha[i] = 255;
        raw[i] = 1.0 + (y - 20) * 0.3; // 高さ方向に深度が激しく変わる
      }
    }
    const r = calibrate({ raw, width: S, height: S, alpha, kind: 'depth', focalPx: FOCAL });
    expect(r.depthToWidth).toBeLessThanOrEqual(0.9);
  });

  it('妥当性は高さではなく幅で見る（構図に左右されないため）', () => {
    // 同じ体を、全身とバストアップで切り出す。奥行き ÷ 高さ は構図で
    // 大きく変わるが、奥行き ÷ 幅 は変わらないはず。
    function subject(y0: number, y1: number) {
      const raw = new Float32Array(S * S);
      const alpha = new Uint8ClampedArray(S * S);
      const cx = S / 2;
      const halfW = 14;
      for (let y = y0; y < y1; y++) {
        for (let x = cx - halfW; x < cx + halfW; x++) {
          const i = y * S + x;
          alpha[i] = 255;
          // 円柱の表面。奥行きは幅で決まり、高さには依らない。
          const t = (x - cx) / halfW;
          raw[i] = 1.0 - 0.08 * Math.sqrt(Math.max(0, 1 - t * t));
        }
      }
      return { raw, alpha };
    }
    const full = subject(10, 86);
    const bust = subject(10, 40);
    const a = calibrate({ ...full, width: S, height: S, kind: 'depth', focalPx: FOCAL });
    const b = calibrate({ ...bust, width: S, height: S, kind: 'depth', focalPx: FOCAL });

    // 高さ比は構図で大きく動く
    expect(Math.abs(a.depthToHeight - b.depthToHeight)).toBeGreaterThan(0.1);
    // 幅比はほとんど動かない
    expect(Math.abs(a.depthToWidth - b.depthToWidth)).toBeLessThan(0.1);
  });

  it('局所強調は奥行きの幅を広げない', () => {
    // 「大域の形は平滑化成分が持つので比率は変わらない」と書いていたが、
    // 実測では 奥行き ÷ 身長 が 0.539 → 0.721（+34%）に広がっていた。
    // 体が奥へ伸びると、少し回しただけで串のように崩れる。
    const raw = new Float32Array(S * S);
    const alpha = new Uint8ClampedArray(S * S);
    const cx = S / 2;
    for (let y = 12; y < 84; y++) {
      for (let x = cx - 16; x < cx + 16; x++) {
        const i = y * S + x;
        alpha[i] = 255;
        const t = (x - cx) / 16;
        // 大域のふくらみ＋細かい起伏
        raw[i] =
          1.0 - 0.06 * Math.sqrt(Math.max(0, 1 - t * t)) + 0.004 * Math.sin(x * 0.9) * Math.cos(y * 0.9);
      }
    }
    const base = { raw, width: S, height: S, alpha, kind: 'depth' as const, focalPx: FOCAL };
    const flat = calibrate({ ...base, reliefBoost: 1 });
    const boosted = calibrate({ ...base, reliefBoost: 3 });
    // 奥行きの幅は変わらない（誤差 10% 以内）
    expect(boosted.depthToWidth).toBeCloseTo(flat.depthToWidth, 1);
  });
});

describe('局所強調は帯域を絞る（v2.3、実写で判明）', () => {
  const W = 96;

  /** 起伏だけ／ノイズだけの場面を作り分ける。 */
  function scene(reliefAmp: number, noiseAmp: number) {
    const z = new Float32Array(W * W);
    const alpha = new Uint8ClampedArray(W * W);
    let seed = 12345;
    const rnd = (): number => {
      seed = (seed * 1103515245 + 12345) & 0x7fffffff;
      return seed / 0x7fffffff - 0.5;
    };
    for (let y = 0; y < W; y++) {
      for (let x = 0; x < W; x++) {
        const i = y * W + x;
        alpha[i] = 255;
        // 波長 24px の起伏＝顔の凹凸に相当する「持ち上げたい帯」
        z[i] =
          1 +
          reliefAmp * Math.sin((x / 24) * Math.PI * 2) * Math.cos((y / 24) * Math.PI * 2) +
          noiseAmp * rnd();
      }
    }
    return { z, alpha };
  }

  /** 内側だけの標準偏差。縁は箱平均が切られるので外す。 */
  function energy(z: ArrayLike<number>): number {
    let s = 0;
    let ss = 0;
    let n = 0;
    for (let y = 30; y < W - 30; y++) {
      for (let x = 30; x < W - 30; x++) {
        const v = z[y * W + x] as number;
        s += v;
        ss += v * v;
        n++;
      }
    }
    return Math.sqrt(ss / n - (s / n) ** 2);
  }

  const RADIUS = 24;
  const BOOST = 3;

  it('起伏の帯はおよそ boost 倍される', () => {
    const { z, alpha } = scene(0.02, 0);
    const gain = energy(enhanceRelief(z, alpha, W, W, RADIUS, BOOST)) / energy(z);
    expect(gain).toBeGreaterThan(BOOST * 0.7);
  });

  it('画素ごとのノイズはほとんど増えない', () => {
    // ここが素朴なアンシャープマスク（z − 平滑化）との違い。ノイズまで
    // boost 倍すると、スカートの判定（深度の段差）が誤爆して顔じゅうに
    // 帯が林立する。実写では全スプラットの 63% がスカートになった。
    const { z, alpha } = scene(0, 0.02);
    const gain = energy(enhanceRelief(z, alpha, W, W, RADIUS, BOOST)) / energy(z);
    expect(gain).toBeLessThan(1.6);
  });

  it('起伏のほうがノイズより桁違いに強く持ち上がる', () => {
    const r = scene(0.02, 0);
    const n = scene(0, 0.02);
    const gr = energy(enhanceRelief(r.z, r.alpha, W, W, RADIUS, BOOST)) / energy(r.z);
    const gn = energy(enhanceRelief(n.z, n.alpha, W, W, RADIUS, BOOST)) / energy(n.z);
    expect(gr / gn).toBeGreaterThan(2);
  });

  it('boost が 1 なら何もしない', () => {
    const { z, alpha } = scene(0.02, 0.004);
    expect(enhanceRelief(z, alpha, W, W, RADIUS, 1)).toBe(z);
  });
});

describe('人体としてあり得ない帯を潰す（v2.5）', () => {
  const W = 64;
  const H = 80;
  const FOCAL = 100;

  /**
   * 上が細い胴（幅 12px）、下が広い脚（幅 48px）。胴だけが深い。
   * 全体で見ると脚が幅を決めるので、比は人体としてあり得る値に収まる。
   */
  function seated(torsoAmp: number, legAmp: number) {
    const z = new Float32Array(W * H);
    const alpha = new Uint8ClampedArray(W * H);
    const put = (x: number, y: number, v: number): void => {
      alpha[y * W + x] = 255;
      z[y * W + x] = v;
    };
    for (let y = 10; y < 40; y++) {
      for (let x = 26; x < 38; x++) put(x, y, 1 + torsoAmp * Math.cos(((x - 26) / 11) * Math.PI));
    }
    for (let y = 40; y < 70; y++) {
      for (let x = 8; x < 56; x++) put(x, y, 1 + legAmp * Math.cos(((x - 8) / 47) * Math.PI));
    }
    return { z, alpha };
  }

  /** 行 y0..y1 の 奥行き ÷ 幅（実寸）。 */
  function ratio(z: Float32Array, alpha: Uint8ClampedArray, y0: number, y1: number): number {
    const zs: number[] = [];
    let xlo = W;
    let xhi = -1;
    for (let y = y0; y < y1; y++) {
      for (let x = 0; x < W; x++) {
        const i = y * W + x;
        if ((alpha[i] as number) < 128) continue;
        zs.push(z[i] as number);
        if (x < xlo) xlo = x;
        if (x > xhi) xhi = x;
      }
    }
    if (xhi < xlo) return 0;
    zs.sort((a, b) => a - b);
    const q = (f: number): number => zs[Math.floor(f * (zs.length - 1))] as number;
    return (q(0.98) - q(0.02)) / (((xhi - xlo + 1) / FOCAL) * 1.0);
  }

  it('深すぎる帯だけを上限まで潰す', () => {
    const { z, alpha } = seated(0.2, 0.05);
    expect(ratio(z, alpha, 10, 40), '前提: 胴は深すぎる').toBeGreaterThan(2);

    const out = flattenImplausibleBands(z, alpha, W, H, FOCAL, 0.7);
    const torso = ratio(out, alpha, 10, 40);
    expect(torso, `胴 ${torso.toFixed(2)}`).toBeLessThan(0.9);
  });

  it('もともと浅い帯には触らない', () => {
    const { z, alpha } = seated(0.2, 0.05);
    const before = ratio(z, alpha, 40, 70);
    const after = ratio(flattenImplausibleBands(z, alpha, W, H, FOCAL, 0.7), alpha, 40, 70);
    expect(after, `脚 ${before.toFixed(2)} → ${after.toFixed(2)}`).toBeCloseTo(before, 2);
  });

  it('胴と脚の境目でも、境目に接する帯が素通りしない', () => {
    // 帯の幅を「帯の外接幅」で取ると、境目の帯は脚の幅で胴の奥行きを割って
    // しまい、比が小さく出て素通りする。分母を行ごとの幅にした理由。
    const { z, alpha } = seated(0.2, 0.05);
    const out = flattenImplausibleBands(z, alpha, W, H, FOCAL, 0.7);
    const lower = ratio(out, alpha, 30, 40); // 胴のうち脚に接する側
    expect(lower, `胴の下端 ${lower.toFixed(2)}`).toBeLessThan(0.9);
  });

  it('どの帯も上限内なら、同じ配列をそのまま返す', () => {
    const { z, alpha } = seated(0.02, 0.02);
    expect(flattenImplausibleBands(z, alpha, W, H, FOCAL, 0.7)).toBe(z);
  });

  it('被写体が無ければ何もしない', () => {
    const z = new Float32Array(W * H).fill(1);
    const alpha = new Uint8ClampedArray(W * H);
    expect(flattenImplausibleBands(z, alpha, W, H, FOCAL, 0.7)).toBe(z);
  });

  it('帯どうしの前後関係は保たれる（体が分断されない）', () => {
    // 胴を手前、脚を奥に置く。潰した後も胴が手前のままであること。
    const { z, alpha } = seated(0.2, 0.05);
    for (let y = 10; y < 40; y++) for (let x = 26; x < 38; x++) z[y * W + x]! += 0.3;

    const med = (a: Float32Array, y0: number, y1: number): number => {
      const v: number[] = [];
      for (let y = y0; y < y1; y++) {
        for (let x = 0; x < W; x++) if ((alpha[y * W + x] as number) >= 128) v.push(a[y * W + x] as number);
      }
      v.sort((p, q) => p - q);
      return v[v.length >> 1] as number;
    };
    const out = flattenImplausibleBands(z, alpha, W, H, FOCAL, 0.7);
    expect(med(out, 10, 40) - med(out, 40, 70)).toBeGreaterThan(0.25);
  });
});

describe('輪郭の棘を均す（v2.6、境界が突起するとの報告から）', () => {
  const W = 80;
  const H = 80;

  /** 中央の円盤。滑らかなドーム状の深度に、縁だけ棘を立てる。 */
  function disc(spike: number) {
    const z = new Float32Array(W * H);
    const alpha = new Uint8ClampedArray(W * H);
    const cx = 40;
    const cy = 40;
    const r = 28;
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const dx = x - cx;
        const dy = y - cy;
        const t = Math.sqrt(dx * dx + dy * dy) / r;
        if (t > 1) continue;
        const i = y * W + x;
        alpha[i] = 255;
        z[i] = 1 - 0.1 * Math.sqrt(Math.max(0, 1 - t * t)); // ドーム
        // 縁から 3px 以内の画素に、交互の符号で棘を立てる
        if (t > 1 - 3 / r) z[i] += (x % 2 === 0 ? spike : -spike);
      }
    }
    return { z, alpha };
  }

  /** 縁の帯での「局所平均からのずれ」の最大値。棘の高さを測る。 */
  function rimSpike(z: Float32Array, alpha: Uint8ClampedArray, band = 4): number {
    let worst = 0;
    for (let y = 1; y < H - 1; y++) {
      for (let x = 1; x < W - 1; x++) {
        const i = y * W + x;
        if ((alpha[i] as number) < 128) continue;
        // 縁からの距離を粗く見る: 4近傍のどれかが被写体外なら縁
        let d = 0;
        while (d < band) {
          const l = x - d - 1;
          const r = x + d + 1;
          if (l < 0 || r >= W) break;
          if ((alpha[y * W + l] as number) < 128 || (alpha[y * W + r] as number) < 128) break;
          d++;
        }
        if (d >= band) continue;
        let sum = 0;
        let n = 0;
        for (let dy = -2; dy <= 2; dy++) {
          for (let dx = -2; dx <= 2; dx++) {
            const j = (y + dy) * W + (x + dx);
            if (j < 0 || j >= W * H || (alpha[j] as number) < 128) continue;
            sum += z[j] as number;
            n++;
          }
        }
        if (n < 4) continue;
        worst = Math.max(worst, Math.abs((z[i] as number) - sum / n));
      }
    }
    return worst;
  }

  it('縁の棘を落とす', () => {
    const { z, alpha } = disc(0.05);
    const before = rimSpike(z, alpha);
    const after = rimSpike(calmSilhouetteRim(z, alpha, W, H, 6, 4), alpha);
    expect(before, `棘 ${before.toFixed(4)}`).toBeGreaterThan(0.03);
    expect(after, `棘 ${before.toFixed(4)} → ${after.toFixed(4)}`).toBeLessThan(before * 0.5);
  });

  it('内部の形は変えない', () => {
    const { z, alpha } = disc(0.05);
    const out = calmSilhouetteRim(z, alpha, W, H, 6, 4);
    // 円盤の中心付近（縁から 10px 以上）は 1 画素も動かないこと
    for (let y = 22; y < 58; y++) {
      for (let x = 22; x < 58; x++) {
        const i = y * W + x;
        const dx = x - 40;
        const dy = y - 40;
        if (Math.sqrt(dx * dx + dy * dy) > 18) continue;
        expect(out[i]).toBe(z[i]);
      }
    }
  });

  it('帯 0 なら何もしない', () => {
    const { z, alpha } = disc(0.05);
    expect(calmSilhouetteRim(z, alpha, W, H, 0, 4)).toBe(z);
  });

  it('強調は輪郭で 1 倍まで落ちる', () => {
    // 縁の画素は増幅されず、内部の画素は増幅されること。
    const { z, alpha } = disc(0);
    const plain = enhanceRelief(z, alpha, W, H, 8, 3);
    const tapered = enhanceRelief(z, alpha, W, H, 8, 3, 6);
    const edge = 40 * W + 14; // 中心から 26px（縁から 2px）
    const core = 40 * W + 40;
    expect(Math.abs((tapered[edge] as number) - (z[edge] as number)))
      .toBeLessThan(Math.abs((plain[edge] as number) - (z[edge] as number)) * 0.7);
    expect(tapered[core]).toBeCloseTo(plain[core] as number, 6);
  });
});
