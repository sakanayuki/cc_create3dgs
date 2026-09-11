/**
 * オフラインのハーネスが共通で使う、実素材の読み込み（docs/12 §12.15）。
 *
 *   python3 scripts/multiview_probe.py --front ... --out tests/multiview-probe
 *
 * が書き出した α・深度・色を読み、**本番と同じ ③ の較正を通して**
 * `AlignView` にする。
 *
 * ## 3つの道具で同じものを読む
 *
 * `align_probe` `gate_probe` `dedup_probe` が別々に読み込みを書いていた。
 * 較正の通し方が1つでもずれると、出てくる数字が比べられなくなる。実際、
 * 較正を通す前の M1（0.985）と通したあとの M1（0.949）を取り違えて、
 * 合格ラインを間違った値で引いた（docs/12 §12.16.5）。**読み込みは1か所に置く。**
 *
 * 本番の経路ではない。実写で数字を出すためだけの道具である。
 */
import { readFileSync, existsSync } from 'node:fs';
import { join } from 'node:path';
import { calibrate } from '../src/pipeline/3-calibrate';
import type { AlignView } from '../src/pipeline/align/registerViews';
import type { ViewSlot } from '../src/pipeline/align/rigid';

export interface ProbeMeta {
  readonly slot: ViewSlot;
  readonly source?: string;
  readonly width: number;
  readonly height: number;
  readonly focalPx: number;
  readonly cx: number;
  readonly cy: number;
  readonly face?: { readonly yawDeg: number; readonly score?: number } | null;
}

export interface ProbeView {
  readonly view: AlignView;
  readonly meta: ProbeMeta;
  /** 較正で決まった実距離の範囲。スプラットを組むときに要る。 */
  readonly nearZ: number;
  readonly farZ: number;
  /** 元写真の色（作業グリッド、RGBA）。 */
  readonly color: Uint8ClampedArray;
}

export const PROBE_SLOTS: readonly ViewSlot[] = ['front', 'right', 'left'];

export function loadProbeView(dir: string, slot: ViewSlot): ProbeView | null {
  const metaPath = join(dir, `${slot}.json`);
  if (!existsSync(metaPath)) return null;
  const meta = JSON.parse(readFileSync(metaPath, 'utf8')) as ProbeMeta;

  const alphaBuf = readFileSync(join(dir, `${slot}.alpha.u8`));
  const rawBuf = readFileSync(join(dir, `${slot}.raw.f32`));
  const colorBuf = readFileSync(join(dir, `${slot}.color.rgba8`));
  const alpha = new Uint8ClampedArray(
    new Uint8Array(alphaBuf.buffer, alphaBuf.byteOffset, alphaBuf.byteLength),
  );
  const raw = new Float32Array(rawBuf.buffer, rawBuf.byteOffset, rawBuf.byteLength / 4);
  const color = new Uint8ClampedArray(
    new Uint8Array(colorBuf.buffer, colorBuf.byteOffset, colorBuf.byteLength),
  );

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
    meta,
    nearZ: cal.nearZ,
    farZ: cal.farZ,
    color,
  };
}

/** そろっている view を順に読む。足りなければそのぶん短い配列になる。 */
export function loadProbeViews(dir: string): readonly ProbeView[] {
  return PROBE_SLOTS.map((s) => loadProbeView(dir, s)).filter((v): v is ProbeView => v !== null);
}
