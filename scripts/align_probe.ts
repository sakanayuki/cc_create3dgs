/**
 * 実素材で位置合わせを測る（docs/12 §12.15 PoC-3A）。
 *
 * `scripts/multiview_probe.py` が書き出した α と深度を読んで、
 * `src/pipeline/align/registerViews.ts` を回し、M1・M2 を出す。
 *
 *   python3 scripts/multiview_probe.py --front ... --out tests/multiview-probe
 *   npx vite-node scripts/align_probe.ts -- tests/multiview-probe
 *
 * 本番の経路ではない。合格の判定に使う数字を、実写で出すためだけの道具である。
 */
import { readFileSync, existsSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { costAtPoses, registerViews, type AlignView } from '../src/pipeline/align/registerViews';
import { calibrate, estimateNormals } from '../src/pipeline/3-calibrate';
import { adaptiveSample, solveSamplingParams } from '../src/pipeline/7-sample';
import { buildSplats, DEFAULT_BUILD_PARAMS, type SplatBuild } from '../src/pipeline/6-splats';
import { mergeBuilds, type MergeSource } from '../src/pipeline/align/mergeViews';
import { splatsToSplat } from '../src/ui/export';
import { torsoAndHead } from '../src/pipeline/align/bodyParts';
import { REFERENCE_POSE, rotationMatrix, toReference, type ViewPose, type ViewSlot } from '../src/pipeline/align/rigid';

interface Meta {
  readonly slot: ViewSlot;
  readonly source: string;
  readonly width: number;
  readonly height: number;
  readonly focalPx: number;
  readonly cx: number;
  readonly cy: number;
  readonly face?: { readonly yawDeg: number; readonly score: number } | null;
}

interface Loaded { readonly view: AlignView; readonly nearZ: number; readonly farZ: number; }

function load(dir: string, slot: ViewSlot): Loaded | null {
  const metaPath = join(dir, `${slot}.json`);
  if (!existsSync(metaPath)) return null;
  const meta = JSON.parse(readFileSync(metaPath, 'utf8')) as Meta;
  const alphaBuf = readFileSync(join(dir, `${slot}.alpha.u8`));
  const rawBuf = readFileSync(join(dir, `${slot}.raw.f32`));
  const alpha = new Uint8ClampedArray(
    new Uint8Array(alphaBuf.buffer, alphaBuf.byteOffset, alphaBuf.byteLength),
  );
  const raw = new Float32Array(rawBuf.buffer, rawBuf.byteOffset, rawBuf.byteLength / 4);

  // ③ の較正を本番と同じ関数で通す（docs/03 §3.5）。
  //
  // これを飛ばすと、DA3 の実寸そのままでは被写体までの距離が被写体の高さより
  // 近いことになり、透視が実際より強く出る。calibrate() は「奥行き ÷ 幅」を
  // 妥当な帯へ収めるので、深度と焦点距離が噛み合う（docs/12 §12.15.4）。
  //
  // 顔の箱は渡さない。渡さないと局所強調が体にも 3 倍でかかるので、
  // reliefBoost は 1 にして強調そのものを止める。位置合わせが見るのは
  // 大づかみの形で、顔の細かい起伏は要らない。
  const cal = calibrate({
    raw,
    width: meta.width,
    height: meta.height,
    alpha,
    kind: 'depth',
    focalPx: meta.focalPx,
    reliefBoost: 1,
  });

  // 0..65535 の正規化深度を、カメラからの z に戻す。被写体の外は 0。
  const depth = new Float32Array(raw.length);
  const span = cal.farZ - cal.nearZ;
  for (let i = 0; i < depth.length; i++) {
    depth[i] = (alpha[i] as number) >= 128 ? cal.nearZ + ((cal.depth[i] as number) / 65535) * span : 0;
  }
  console.log(
    `  ${slot}: 較正 nearZ=${cal.nearZ.toFixed(3)} farZ=${cal.farZ.toFixed(3)} ` +
      `奥行き÷幅=${cal.depthToWidth.toFixed(3)} 奥行き÷高さ=${cal.depthToHeight.toFixed(3)} 実寸=${cal.metric}`,
  );

  return {
    view: {
      slot,
      width: meta.width,
      height: meta.height,
      camera: { focalPx: meta.focalPx, cx: meta.cx, cy: meta.cy },
      alpha,
      depth,
      // 顔から測ったヨー。**符号だけ**使う（docs/12 §12.15.6）。
      ...(meta.face ? { headYawDeg: meta.face.yawDeg } : {}),
    },
    nearZ: cal.nearZ,
    farZ: cal.farZ,
  };
}

const deg = (rad: number): number => (rad * 180) / Math.PI;

/**
 * view 1枚ぶんのスプラットを作る。
 *
 * 本番（`generate()`）の ④〜⑦ と同じ関数を同じ順で呼ぶ。違うのは
 * ⑥⑦ を作業グリッドのまま回すこと（元写真の再サンプルはこの道具では省く）と、
 * 背面シェル・スカートの色の差し替えを行わないこと。
 * **合成が実物の形で動くかを見るための道具**であって、品質の再現ではない。
 */
function buildOne(v: AlignView, color: Uint8ClampedArray, near: number, far: number): SplatBuild {
  const g = v.width;
  const span = far - near;
  const depth01 = new Float32Array(g * g);
  for (let i = 0; i < depth01.length; i++) {
    const z = v.depth[i] as number;
    depth01[i] = z > 0 ? Math.max(0, Math.min(1, (z - near) / span)) : 0;
  }
  const metric = Float32Array.from(v.depth as ArrayLike<number>);
  const normals = estimateNormals(metric, g, g, v.camera.focalPx, span * 0.05);
  const alpha = v.alpha as Uint8ClampedArray;
  const params = solveSamplingParams(depth01, color, alpha, g, g, 0.3);
  const cells = adaptiveSample(depth01, color, alpha, g, g, params);
  return buildSplats(
    { cells, normals, width: g, height: g, focalPx: v.camera.focalPx, nearZ: near, farZ: far },
    color,
    alpha,
    null,
    null,
    DEFAULT_BUILD_PARAMS,
  );
}

function main(): void {
  const dir = process.argv[2] ?? 'tests/multiview-probe';
  const slots: ViewSlot[] = ['front', 'right', 'left'];
  const loaded = slots.map((s) => load(dir, s)).filter((v): v is Loaded => v !== null);
  const views = loaded.map((l) => l.view);
  const ranges = loaded.map((l) => ({ nearZ: l.nearZ, farZ: l.farZ }));
  if (views.length < 2) {
    console.error(`${dir} に view が足りません（${views.length} 枚）。先に multiview_probe.py を回してください。`);
    process.exit(1);
  }
  console.log(`${views.length} 枚を読みました: ${views.map((v) => v.slot).join(', ')}`);

  const report: Record<string, unknown> = {};
  for (const useParts of [false, true]) {
    const input: AlignView[] = views.map((v) =>
      useParts ? { ...v, usable: torsoAndHead(v.alpha, v.width, v.height) } : v,
    );
    const label = useParts ? '胴と頭だけ（docs/12 §12.7）' : '被写体まるごと';
    const t0 = Date.now();
    const res = registerViews(input, { samplesPerView: 2500, gridLongSide: 128 });
    const ms = Date.now() - t0;

    console.log(`\n── ${label} ── ${ms} ms`);
    console.log(`  目的関数 = ${res.cost.toFixed(5)} / M1（収まり率）の最小 = ${res.worstInsideRatio.toFixed(3)}`);
    if (res.anySlotFlipped) {
      const names = res.views.filter((v) => v.slotFlipped).map((v) => v.slot);
      console.log(`  **枠を読み替えました**: ${names.join(', ')}（docs/12 R20）`);
    }
    for (const v of res.views) {
      console.log(
        `  ${v.slot.padEnd(5)} yaw=${deg(v.pose.yaw).toFixed(1)}° ` +
          `pitch=${deg(v.pose.pitch).toFixed(1)}° roll=${deg(v.pose.roll).toFixed(1)}° ` +
          `尺度=${v.pose.scale.toFixed(3)} ` +
          `t=(${v.pose.tx.toFixed(3)},${v.pose.ty.toFixed(3)},${v.pose.tz.toFixed(3)}) ` +
          `M1=${v.insideRatio.toFixed(3)} IoU=${v.iou.toFixed(3)} はみ出し=${v.containmentPx.toFixed(2)}px`,
      );
    }
    report[useParts ? 'torsoAndHead' : 'whole'] = {
      ms,
      cost: res.cost,
      worstInsideRatio: res.worstInsideRatio,
      views: res.views.map((v) => ({
        slot: v.slot,
        yawDeg: deg(v.pose.yaw),
        scale: v.pose.scale,
        insideRatio: v.insideRatio, iou: v.iou,
        containmentPx: v.containmentPx,
      })),
    };
  }

  // 片方ずつ振る。左右をまとめて振ると、非対称な解を見落とす。
  console.log('\n── 片方ずつ振ってみる（もう片方は 90° に固定） ──');
  {
    const usable2 = views.map((v) => ({ ...v, usable: torsoAndHead(v.alpha, v.width, v.height) }));
    for (const moving of ['right', 'left'] as const) {
      const row: string[] = [];
      for (let d = 30; d <= 120; d += 10) {
        const poses: ViewPose[] = views.map((v) => {
          if (v.slot === 'front') return REFERENCE_POSE;
          const sign = v.slot === 'right' ? 1 : -1;
          const angle = v.slot === moving ? d : 90;
          return { ...REFERENCE_POSE, yaw: (sign * angle * Math.PI) / 180, scale: 1.05 };
        });
        const c = costAtPoses(usable2, poses, { samplesPerView: 2500, gridLongSide: 128 });
        row.push(`${d}:${c.containment.toFixed(4)}`);
      }
      console.log(`  ${moving} を振る（収まり）: ${row.join('  ')}`);
    }
  }

  // 手で置いた ±90° と、見つけた姿勢のコストを比べる。
  // 「見つけた姿勢のほうが安いのに、見た目は間違っている」なら、目的関数がまだ悪い。
  console.log('\n── 手で置いた姿勢との比べ ──');
  const usable = views.map((v) => ({ ...v, usable: torsoAndHead(v.alpha, v.width, v.height) }));
  for (const d of [45, 60, 75, 90, 105]) {
    const poses: ViewPose[] = views.map((v) =>
      v.slot === 'front'
        ? REFERENCE_POSE
        : { ...REFERENCE_POSE, yaw: ((v.slot === 'right' ? d : -d) * Math.PI) / 180, scale: 1.05 },
    );
    const c = costAtPoses(usable, poses, { samplesPerView: 2500, gridLongSide: 128 });
    console.log(
      `  ±${String(d).padStart(3)}°  目的関数=${c.total.toFixed(5)}  収まり=${c.containment.toFixed(5)}  縦の広がり=${c.extent.toFixed(5)}`,
    );
  }

  // --- 実素材から本当にスプラットを作って合成する（Ⓒ の検算）
  console.log('\n── 合成（Ⓒ） ──');
  const reg = registerViews(views, { samplesPerView: 2500, gridLongSide: 128 });
  const sources: MergeSource[] = views.map((v, i) => {
    const meta = JSON.parse(readFileSync(join(dir, `${v.slot}.json`), 'utf8')) as Meta;
    const colorBuf = readFileSync(join(dir, `${v.slot}.color.rgba8`));
    const color = new Uint8ClampedArray(
      new Uint8Array(colorBuf.buffer, colorBuf.byteOffset, colorBuf.byteLength),
    );
    const near = ranges[i]?.nearZ ?? 0;
    const far = ranges[i]?.farZ ?? 1;
    const build = buildOne(v, color, near, far);
    console.log(
      `  ${v.slot.padEnd(5)} ${build.count.toLocaleString('ja-JP').padStart(9)} 個  ` +
        `焦点 ${meta.focalPx.toFixed(0)}px`,
    );
    const r = reg.views[i];
    if (!r) throw new Error('位置合わせの結果がありません');
    console.log(
      `        yaw=${deg(r.pose.yaw).toFixed(1)}° 尺度=${r.pose.scale.toFixed(3)} ` +
        `frame.source=(${r.frame.source.map((x) => x.toFixed(3)).join(',')}) ` +
        `target=(${r.frame.target.map((x) => x.toFixed(3)).join(',')})`,
    );
    return { build, pose: r.pose, frame: r.frame };
  });
  // **回転が効いているかを、幅と奥行きの入れ替わりで見る。**
  // 横向きの view は「幅＝被写体の厚み、奥行き＝表面の起伏」で入っている。
  // 90° 回せばこの2つが入れ替わるはずで、入れ替わらなければ回っていない。
  // 見た目だけで判断しようとして一度誤診したので、数字で見る。
  for (const src of sources) {
    const R = rotationMatrix(src.pose);
    const f = new Float32Array(src.build.data.buffer, src.build.data.byteOffset, src.build.count * 6);
    const inv = 1 / src.build.normalization.scale;
    const c = src.build.normalization.center;
    const t = new Float64Array(3);
    let ax = Infinity, bx = -Infinity, az = Infinity, bz = -Infinity;
    let cx0 = Infinity, cx1 = -Infinity, cz0 = Infinity, cz1 = -Infinity;
    for (let i = 0; i < src.build.count; i += 37) {
      const X = (f[i * 6] as number) * inv + c[0];
      const Y = (f[i * 6 + 1] as number) * inv + c[1];
      const Z = (f[i * 6 + 2] as number) * inv + c[2];
      if (X < cx0) cx0 = X; if (X > cx1) cx1 = X;
      if (Z < cz0) cz0 = Z; if (Z > cz1) cz1 = Z;
      toReference(R, src.pose, src.frame, X, Y, Z, t);
      const x = t[0] as number, z = t[2] as number;
      if (x < ax) ax = x; if (x > bx) bx = x;
      if (z < az) az = z; if (z > bz) bz = z;
    }
    console.log(
      `        運ぶ前 幅=${(cx1 - cx0).toFixed(3)} 奥行き=${(cz1 - cz0).toFixed(3)}` +
        ` → 運んだ後 幅=${(bx - ax).toFixed(3)} 奥行き=${(bz - az).toFixed(3)}`,
    );
  }

  const merged = mergeBuilds(sources);
  const sum = sources.reduce((n, s) => n + s.build.count, 0);
  console.log(`  合計 ${sum.toLocaleString('ja-JP')} 個 → 合成後 ${merged.count.toLocaleString('ja-JP')} 個`);
  console.log(`  外接の中心 (${merged.normalization.center.map((x) => x.toFixed(3)).join(', ')}) / 尺度 ${merged.normalization.scale.toFixed(3)}`);
  console.log(`  被写体の縦の広がり ${merged.metricHeight.toFixed(3)}（深度モデルの単位）`);

  // view ごとに色を塗り分けたものも出す。どの view がどこに置かれたかを見るため。
  const tint: [number, number, number][] = [
    [255, 70, 70],
    [70, 255, 70],
    [90, 140, 255],
  ];
  const tinted = mergeBuilds(
    sources.map((src, i) => {
      const copy = new Uint8Array(src.build.data);
      const u32 = new Uint32Array(copy.buffer);
      const t = tint[i % tint.length] as [number, number, number];
      for (let k = 0; k < src.build.count; k++) {
        const o = k * 6 + 5;
        const a = ((u32[o] as number) >>> 24) & 0xff;
        u32[o] = (a << 24) | (t[2] << 16) | (t[1] << 8) | t[0];
      }
      return { ...src, build: { ...src.build, data: copy } };
    }),
  );
  writeFileSync(
    join(dir, 'merged_bysource.splat'),
    splatsToSplat(tinted.data, tinted.count, { ...tinted.normalization, metricFix: 1 }),
  );

  const splat = splatsToSplat(merged.data, merged.count, {
    ...merged.normalization,
    metricFix: 1,
  });
  const splatPath = join(dir, 'merged.splat');
  writeFileSync(splatPath, splat);
  console.log(`  書き出し: ${splatPath}（${(splat.byteLength / 1e6).toFixed(1)} MB）`);

  const out = join(dir, 'align_report.json');
  writeFileSync(out, JSON.stringify(report, null, 2));
  console.log(`\n書き出しました: ${out}`);
}

main();
