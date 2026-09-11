/**
 * 重複除去の効き目を実素材で測る（docs/12 §12.8、§12.16.6）。
 *
 *   npx vite-node scripts/dedup_probe.ts -- tests/multiview-probe
 *
 * 測るのは2つ。
 *
 *   点の数    … バジェット（60 万）に収まるか
 *   二重率    … **同じ面が2つ置かれている点の割合**
 *
 * 二重率は、合成後の点のうち「**体の 1% 以内**に、向きの揃った別の点がある」
 * ものの割合である。1% は位置合わせの残差（128 画素グリッドで 1 画素 ≒ 0.8%）
 * ほどで、同じ面が2つ置かれていればこの範囲に相手がいる。
 * **目で見るより先に、これを数字で見る。**
 * 実素材の色つきを見て「顔が二重だから回っていない」と誤診したことがある
 * （docs/12 §12.16.2）。
 *
 * 本番の経路ではない。格子の目をいくつにするかを決めるための道具である。
 */
import { writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { registerViews } from '../src/pipeline/align/registerViews';
import { estimateNormals } from '../src/pipeline/3-calibrate';
import { adaptiveSample, solveSamplingParams } from '../src/pipeline/7-sample';
import { buildSplats, DEFAULT_BUILD_PARAMS, type SplatBuild } from '../src/pipeline/6-splats';
import { mergeBuilds, type MergeSource } from '../src/pipeline/align/mergeViews';
import { decodeOct } from '../src/codec/pack';
import { loadProbeViews, type ProbeView } from './probeData';

const STRIDE32 = 6;

/** view 1枚ぶんのスプラットを作る（align_probe と同じ手順）。 */
function buildOne(p: ProbeView): SplatBuild {
  const v = p.view;
  const g = v.width;
  const span = p.farZ - p.nearZ;
  const depth01 = new Float32Array(g * g);
  for (let i = 0; i < depth01.length; i++) {
    const z = v.depth[i] as number;
    depth01[i] = z > 0 ? Math.max(0, Math.min(1, (z - p.nearZ) / span)) : 0;
  }
  const metric = Float32Array.from(v.depth as ArrayLike<number>);
  const normals = estimateNormals(metric, g, g, v.camera.focalPx, span * 0.05);
  const alpha = v.alpha as Uint8ClampedArray;
  const params = solveSamplingParams(depth01, p.color, alpha, g, g, 0.3);
  const cells = adaptiveSample(depth01, p.color, alpha, g, g, params);
  return buildSplats(
    { cells, normals, width: g, height: g, focalPx: v.camera.focalPx, nearZ: p.nearZ, farZ: p.farZ },
    p.color,
    alpha,
    null,
    null,
    DEFAULT_BUILD_PARAMS,
  );
}

/**
 * 二重率。**別の view から来た点**が、同じ面としてすぐ近くにいる割合。
 *
 * 「すぐ近く」は体のいちばん長い辺の `near` 倍（既定 1%）。位置合わせの残差
 * （128 画素のグリッドで 1 画素 ≒ 0.8%）ほどで、同じ面が2つ置かれていれば
 * この範囲に相手がいる。
 *
 * **同じ view の点を数えてはいけない。** 点の間隔は体の 0.1% ほどしかないので、
 * 同じ view の隣を数えると落とす前も後も 100% になる。最初それで測って、
 * 意味のない数字を出した（docs/12 §12.16.6）。
 */
function doubleRate(b: SplatBuild, src: Uint16Array, near = 0.01, sample = 3000): number {
  const f32 = new Float32Array(b.data.buffer, b.data.byteOffset, b.count * STRIDE32);
  const u32 = new Uint32Array(b.data.buffer, b.data.byteOffset, b.count * STRIDE32);
  const lim = near; // 正規化済みなので、体のいちばん長い辺が 1
  if (!(lim > 0) || b.count === 0) return 0;

  const step = Math.max(1, Math.floor(b.count / sample));
  let checked = 0;
  let doubled = 0;
  for (let i = 0; i < b.count; i += step) {
    const x = f32[i * STRIDE32] as number;
    const y = f32[i * STRIDE32 + 1] as number;
    const z = f32[i * STRIDE32 + 2] as number;
    const n = decodeOct(u32[i * STRIDE32 + 3] as number);
    const si = src[i] as number;
    checked++;
    for (let j = 0; j < b.count; j++) {
      if ((src[j] as number) === si) continue; // 同じ view は数えない
      const dx = (f32[j * STRIDE32] as number) - x;
      if (dx > lim || dx < -lim) continue;
      const dy = (f32[j * STRIDE32 + 1] as number) - y;
      if (dy > lim || dy < -lim) continue;
      const dz = (f32[j * STRIDE32 + 2] as number) - z;
      if (dx * dx + dy * dy + dz * dz > lim * lim) continue;
      const m = decodeOct(u32[j * STRIDE32 + 3] as number);
      if (n[0] * m[0] + n[1] * m[1] + n[2] * m[2] < 0.7) continue;
      doubled++;
      break;
    }
  }
  return checked > 0 ? doubled / checked : 0;
}

function main(): void {
  const dir = process.argv[2] ?? 'tests/multiview-probe';
  const loaded = loadProbeViews(dir);
  if (loaded.length < 2) {
    console.error(`${dir} に view が足りません（${loaded.length} 枚）`);
    process.exit(1);
  }
  const views = loaded.map((l) => l.view);
  const reg = registerViews(views);
  console.log(
    `位置合わせ: ${reg.views.map((v) => `${v.slot} ${((v.pose.yaw * 180) / Math.PI).toFixed(1)}°`).join(' / ')}` +
      `  最悪M1=${reg.worstInsideRatio.toFixed(3)}`,
  );

  const sources: MergeSource[] = loaded.map((p, i) => {
    const build = buildOne(p);
    const r = reg.views[i];
    if (!r) throw new Error('位置合わせの結果がありません');
    console.log(`  ${p.view.slot.padEnd(5)} ${build.count.toLocaleString('ja-JP').padStart(9)} 個`);
    return { build, pose: r.pose, frame: r.frame };
  });

  const rows: Record<string, unknown>[] = [];
  console.log('');
  console.log('格子の目   点の数      落とした   二重率   ms');
  const sweep: readonly (readonly [number, boolean])[] = [
    [0, true],
    [0.01, true],
    [0.015, true],
    [0.02, true],
    [0.03, true],
    [0.05, true],
    // 向きを一切見ない（本番では使えない。残った二重像の出どころを測るため）
    [0.02, false],
    [0.03, false],
  ];
  for (const [cellRatio, splitSides] of sweep) {
    const t0 = Date.now();
    const m = mergeBuilds(
      sources,
      cellRatio === 0
        ? { dedupe: false, keepSourceIds: true }
        : { cellRatio, splitSides, keepSourceIds: true },
    );
    const ms = Date.now() - t0;
    const dr = doubleRate(m, m.mergeStats.sourceOf as Uint16Array);
    const label =
      cellRatio === 0
        ? '落とさない'
        : `${(cellRatio * 100).toFixed(1)}%${splitSides ? '' : ' 向き無視'}`;
    console.log(
      `${label.padEnd(10, ' ')} ${m.count.toLocaleString('ja-JP').padStart(9)}  ` +
        `${((m.mergeStats.dropped / Math.max(1, m.mergeStats.before)) * 100).toFixed(1).padStart(7)}%  ` +
        `${(dr * 100).toFixed(1).padStart(6)}%  ${String(ms).padStart(5)}`,
    );
    rows.push({
      cellRatio,
      splitSides,
      count: m.count,
      before: m.mergeStats.before,
      droppedRatio: m.mergeStats.dropped / Math.max(1, m.mergeStats.before),
      doubleRate: dr,
      ms,
    });
  }

  writeFileSync(join(dir, 'dedup_report.json'), JSON.stringify(rows, null, 2));
  console.log(`\n${join(dir, 'dedup_report.json')} に書きました。`);
}

main();
