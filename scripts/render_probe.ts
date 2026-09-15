/**
 * 合成した立体を**絵にして見る**（docs/12 §12.16.7）。
 *
 *   npx vite-node scripts/render_probe.ts -- tests/multiview-probe
 *
 * 数字だけで進めた結果、実機で「1枚のときより正面の見た目が悪い」と
 * 言われた。二重率も点数も改善していたのに、である。**測っていた量が、
 * 見え方を表していなかった。** 絵を出す道具をここに置く。
 *
 * 出すのは PLY。本番の書き出しと同じ `splatsToPly` を通すので、
 * 見ているものと利用者が保存するものがずれない。描画は Python 側
 * （`scripts/render_ply.py`）で行う。
 */
import { writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { registerViews } from '../src/pipeline/align/registerViews';
import { estimateNormals } from '../src/pipeline/3-calibrate';
import { adaptiveSample, solveSamplingParams } from '../src/pipeline/7-sample';
import {
  buildSplats,
  DEFAULT_BUILD_PARAMS,
  type BuildParams,
  type SplatBuild,
} from '../src/pipeline/6-splats';
import { mergeBuilds, type MergeSource } from '../src/pipeline/align/mergeViews';
import { splatsToPly } from '../src/ui/export';
import { loadProbeViews, type ProbeView } from './probeData';

function buildOne(p: ProbeView, params: BuildParams): SplatBuild {
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
  const sp = solveSamplingParams(depth01, p.color, alpha, g, g, 0.3);
  const cells = adaptiveSample(depth01, p.color, alpha, g, g, sp);
  return buildSplats(
    { cells, normals, width: g, height: g, focalPx: v.camera.focalPx, nearZ: p.nearZ, farZ: p.farZ },
    p.color,
    alpha,
    null, // 厚みマップ＝背面シェル。既定 false なので本番も作らない
    null,
    params,
  );
}

function main(): void {
  const dir = process.argv[2] ?? 'tests/multiview-probe';
  const loaded = loadProbeViews(dir);
  if (loaded.length < 2) {
    console.error(`${dir} に view が足りません`);
    process.exit(1);
  }
  const reg = registerViews(loaded.map((l) => l.view));

  const withSkirt = DEFAULT_BUILD_PARAMS;
  const noSkirt: BuildParams = { ...DEFAULT_BUILD_PARAMS, skirt: false };

  const sources = (params: BuildParams, occlude = false): MergeSource[] =>
    loaded.map((p, i) => {
      const r = reg.views[i];
      if (!r) throw new Error('位置合わせの結果がありません');
      const base = { build: buildOne(p, params), pose: r.pose, frame: r.frame };
      if (!occlude) return base;
      return {
        ...base,
        occluder: {
          width: p.view.width,
          height: p.view.height,
          camera: p.view.camera,
          depth: p.view.depth as Float32Array,
        },
      };
    });

  const withS = sources(withSkirt);
  const noS = sources(noSkirt);

  const variants: readonly (readonly [string, SplatBuild])[] = [
    // 比べる相手。利用者が「こちらのほうが良い」と言っている1枚モード。
    ['single_front', withS[0]?.build as SplatBuild],
    ['multi_skirt_nodedup', mergeBuilds(withS, { dedupe: false })],
    ['multi_skirt_dedup', mergeBuilds(withS)],
    ['multi_noskirt_nodedup', mergeBuilds(noS, { dedupe: false })],
    ['multi_noskirt_dedup', mergeBuilds(noS)],
    ['multi_free', mergeBuilds(sources(withSkirt, true))],
  ];

  // **view ごとに色を塗り分ける。** どの写真の点がどこに置かれているかを見る。
  // 自然色で見ると「顔が二重だから回っていない」と誤診したことがある（§12.16.2）。
  const TINT: readonly (readonly [number, number, number])[] = [
    [230, 60, 60],
    [60, 200, 90],
    [70, 120, 240],
  ];
  const tinted = withS.map((src, i) => {
    const t = TINT[i] as readonly [number, number, number];
    const data = new Uint8Array(src.build.data);
    const u32 = new Uint32Array(data.buffer);
    for (let k = 0; k < src.build.count; k++) {
      const o = k * 6 + 5;
      const a = ((u32[o] as number) >>> 24) & 0xff;
      u32[o] = ((a << 24) | (t[2] << 16) | (t[1] << 8) | t[0]) >>> 0;
    }
    return { ...src, build: { ...src.build, data } };
  });

  const all: readonly (readonly [string, SplatBuild])[] = [
    ...variants,
    ['bysource_nodedup', mergeBuilds(tinted, { dedupe: false })],
    ['bysource_dedup', mergeBuilds(tinted)],
  ];

  for (const [name, b] of all) {
    const ply = splatsToPly(b.data, b.count);
    writeFileSync(join(dir, `${name}.ply`), ply);
    console.log(`${name.padEnd(24)} ${b.count.toLocaleString('ja-JP').padStart(9)} 点  → ${name}.ply`);
  }
}

main();
