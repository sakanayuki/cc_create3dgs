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
import { torsoAndHead } from '../src/pipeline/align/bodyParts';
import { REFERENCE_POSE, type ViewPose, type ViewSlot } from '../src/pipeline/align/rigid';

interface Meta {
  readonly slot: ViewSlot;
  readonly source: string;
  readonly width: number;
  readonly height: number;
  readonly focalPx: number;
  readonly cx: number;
  readonly cy: number;
}

function load(dir: string, slot: ViewSlot): AlignView | null {
  const metaPath = join(dir, `${slot}.json`);
  if (!existsSync(metaPath)) return null;
  const meta = JSON.parse(readFileSync(metaPath, 'utf8')) as Meta;
  const alphaBuf = readFileSync(join(dir, `${slot}.alpha.u8`));
  const depthBuf = readFileSync(join(dir, `${slot}.depth.f32`));
  const alpha = new Uint8Array(alphaBuf.buffer, alphaBuf.byteOffset, alphaBuf.byteLength);
  const depth = new Float32Array(depthBuf.buffer, depthBuf.byteOffset, depthBuf.byteLength / 4);
  return {
    slot,
    width: meta.width,
    height: meta.height,
    camera: { focalPx: meta.focalPx, cx: meta.cx, cy: meta.cy },
    alpha,
    depth,
  };
}

const deg = (rad: number): number => (rad * 180) / Math.PI;

function main(): void {
  const dir = process.argv[2] ?? 'tests/multiview-probe';
  const slots: ViewSlot[] = ['front', 'right', 'left'];
  const views = slots.map((s) => load(dir, s)).filter((v): v is AlignView => v !== null);
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

  const out = join(dir, 'align_report.json');
  writeFileSync(out, JSON.stringify(report, null, 2));
  console.log(`\n書き出しました: ${out}`);
}

main();
