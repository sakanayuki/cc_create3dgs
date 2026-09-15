/**
 * 書き出した `.splat` が「人体としてあり得る形か」を測る（docs/13 §13.3）。
 *
 *   npx tsx scripts/shape_probe.ts out.splat [他.splat ...]
 *
 * **被写体だけを含むファイルに使う。** `tests/` にある参照実装の出力は背景込み
 * なので、行を横に切ると壁まで一続きの区間になってしまい、数が意味を持たない。
 * 目安として出しているのは、背景を除いた参照実装の出力を同じ物差しで測った値
 * （飛び出し99% 0.051 / 凹み90% 0.131）。合格ラインは docs/11 §11.5。
 */
import { readFileSync, existsSync } from 'node:fs';
import { basename } from 'node:path';
import {
  crossSectionBulge,
  crossSectionDent,
  fromSplatFile,
  frontDepthMap,
} from '../tests/helpers/bodyShape';

const REFERENCE = [
  'tests/SHARPtest26.splat',
  'tests/SHARPtest31.splat',
  'tests/SHARPtest38.splat',
].filter((p) => existsSync(p));

const files = process.argv.slice(2);
const targets = files.length > 0 ? files : REFERENCE;
if (targets.length === 0) {
  console.error('測るファイルがありません。.splat を引数で渡してください。');
  process.exit(1);
}

if (files.length === 0) {
  console.warn(
    '注意: tests/ の参照実装の出力は**背景込み**なので、この物差しは当てになりません。\n' +
      '      被写体だけを含むファイル（私たちの出力）を引数で渡してください。\n',
  );
}

console.log('横断面の形（体幅に対する比。小さいほど人体らしい）');
console.log(
  '%s %s %s %s %s',
  '名前'.padEnd(22),
  '飛び出し99%'.padStart(12),
  '飛び出し90%'.padStart(12),
  '凹み90%'.padStart(10),
  '凹み最大'.padStart(10),
);
// 目安は参照実装（SHARP）の被写体だけの出力を、同じ物差しで測った値。
console.log('  目安（参照実装の実測）: 飛び出し99% 0.051 / 90% 0.028 / 凹み90% 0.131 / 凹み最大 0.431');

for (const file of targets) {
  const bytes = new Uint8Array(readFileSync(file));
  const { data, count } = fromSplatFile(bytes);
  if (count === 0) {
    console.log('%s  読めません（32 バイト区切りではない？）', basename(file).padEnd(22));
    continue;
  }
  const map = frontDepthMap(data, count);
  const bulge = crossSectionBulge(map);
  const dent = crossSectionDent(map);
  console.log(
    '%s %s %s %s %s  (%d 枚)',
    basename(file).padEnd(22),
    bulge.p99.toFixed(4).padStart(12),
    bulge.p90.toFixed(4).padStart(12),
    dent.p90.toFixed(4).padStart(10),
    dent.max.toFixed(4).padStart(10),
    count,
  );
}
