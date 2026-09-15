/**
 * 合格ラインを実測から決める（docs/12 §12.16.5）。
 *
 *   npx vite-node scripts/gate_probe.ts -- tests/multiview-probe
 *
 * `registerViews` が解いた姿勢だけでなく、**わざと間違えた姿勢**でも
 * M1（収まり率）と平均はみ出しを測る。合格ラインは片側だけ見ても決まらない。
 * 「正しく合ったときいくつか」と「間違えたときどこまで落ちるか」の**両方**が
 * 要る。0.95 は前者だけを見て置いてしまい、実素材の正解を落とした。
 *
 * 本番の経路ではない。線を引く場所を数字で決めるための道具である。
 */
import { writeFileSync } from 'node:fs';
import { join } from 'node:path';
import {
  measureAtPoses,
  registerViews,
  type AlignView,
} from '../src/pipeline/align/registerViews';
import { loadProbeViews } from './probeData';
import { REFERENCE_POSE, type ViewPose } from '../src/pipeline/align/rigid';

const rad = (d: number): number => (d * Math.PI) / 180;
const f3 = (x: number): string => x.toFixed(3);

interface Row {
  readonly 場合: string;
  readonly 説明: string;
  readonly 最悪M1: number;
  readonly 最悪はみ出しpx: number;
  readonly 内訳: string;
}

function row(label: string, note: string, views: readonly AlignView[], poses: readonly ViewPose[]): Row {
  const m = measureAtPoses(views, poses);
  return {
    場合: label,
    説明: note,
    最悪M1: Math.min(...m.map((x) => x.insideRatio)),
    最悪はみ出しpx: Math.max(...m.map((x) => x.containmentPx)),
    内訳: m.map((x) => `${x.slot} ${f3(x.insideRatio)}/${f3(x.containmentPx)}`).join('  '),
  };
}

function main(): void {
  const dir = process.argv[2] ?? 'tests/multiview-probe';
  const views = loadProbeViews(dir).map((p) => p.view);
  if (views.length < 2) {
    console.error(`${dir} に view が足りません（${views.length} 枚）`);
    process.exit(1);
  }

  const solved = registerViews(views);
  const solvedPoses = solved.views.map((v) => v.pose);
  console.log(
    `解いた姿勢: ${solved.views.map((v) => `${v.slot} ${((v.pose.yaw * 180) / Math.PI).toFixed(1)}°`).join(' / ')}`,
  );

  const rows: Row[] = [row('正解', 'registerViews が解いた姿勢', views, solvedPoses)];

  // 間違えた姿勢。yaw をずらすのは「横向きの写真を別の角度だと思い込んだ」に当たる。
  // 利用者が実際にやる間違い（少し斜めの写真を横向きの枠に入れる、など）は
  // このあたりに落ちる。
  for (const off of [10, 20, 30, 45, 60, 90]) {
    const poses = solvedPoses.map((p, i) =>
      i === solved.referenceIndex ? p : { ...p, yaw: p.yaw + rad(off) },
    );
    rows.push(row(`yaw +${off}°`, '横向きの角度を取り違えた', views, poses));
  }
  // 枠を入れ違えた（R20 が直す前の状態）。符号を反転させる。
  rows.push(
    row(
      'yaw 符号反転',
      '右と左の枠を入れ違えた（R20 が直す前）',
      views,
      solvedPoses.map((p, i) => (i === solved.referenceIndex ? p : { ...p, yaw: -p.yaw })),
    ),
  );
  // 合わせを一切しない（枠の値そのまま、尺度も 1）。
  rows.push(
    row(
      '合わせ無し',
      '枠の角度をそのまま信じ、尺度も位置も合わせない',
      views,
      views.map((v, i) =>
        i === solved.referenceIndex
          ? REFERENCE_POSE
          : { ...REFERENCE_POSE, yaw: rad(v.slot === 'right' ? 90 : v.slot === 'left' ? -90 : 0) },
      ),
    ),
  );
  // 尺度が大きく外れた（被写体との距離が違う写真を混ぜた）。
  for (const k of [1.15, 1.3]) {
    rows.push(
      row(
        `尺度 ×${k}`,
        '被写体との距離が違う写真を混ぜた',
        views,
        solvedPoses.map((p, i) => (i === solved.referenceIndex ? p : { ...p, scale: p.scale * k })),
      ),
    );
  }

  console.log('');
  console.log('場合                説明                                    最悪M1  最悪はみ出し  内訳(M1/px)');
  for (const r of rows) {
    console.log(
      `${r.場合.padEnd(16, ' ')} ${r.説明.padEnd(38, ' ')} ${f3(r.最悪M1)}   ${r.最悪はみ出しpx.toFixed(2).padStart(7)}   ${r.内訳}`,
    );
  }

  // --- 2枚のとき
  //
  // **3枚あって初めて角度が縛られる。** 2枚だと拘束が足りず、正しい組でも
  // M1 が下がる。合格ラインを枚数で変える必要があるかを、ここで確かめる。
  const pairs: readonly (readonly [string, readonly AlignView[]])[] = [
    ['正面+右', [views[0] as AlignView, views[1] as AlignView]],
    ['正面+左', [views[0] as AlignView, views[2] as AlignView]],
  ];
  const pairRows: Row[] = [];
  for (const [name, vs] of pairs) {
    if (vs.length < 2 || vs.some((v) => v === undefined)) continue;
    const sol = registerViews(vs);
    const sp = sol.views.map((v) => v.pose);
    pairRows.push(row(`${name} 正解`, 'registerViews が解いた姿勢', vs, sp));
    for (const off of [20, 45, 90]) {
      pairRows.push(
        row(
          `${name} yaw +${off}°`,
          '角度を取り違えた',
          vs,
          sp.map((p, i) => (i === sol.referenceIndex ? p : { ...p, yaw: p.yaw + rad(off) })),
        ),
      );
    }
  }

  console.log('');
  console.log('--- 2枚のとき ---');
  console.log('場合                説明                                    最悪M1  最悪はみ出し  内訳(M1/px)');
  for (const r of pairRows) {
    console.log(
      `${r.場合.padEnd(16, ' ')} ${r.説明.padEnd(38, ' ')} ${f3(r.最悪M1)}   ${r.最悪はみ出しpx.toFixed(2).padStart(7)}   ${r.内訳}`,
    );
  }

  writeFileSync(
    join(dir, 'gate_report.json'),
    JSON.stringify({ 三枚: rows, 二枚: pairRows }, null, 2),
  );
  console.log(`\n${join(dir, 'gate_report.json')} に書きました。`);
}

main();
