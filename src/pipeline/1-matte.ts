/**
 * ① 被写体抽出の後処理（docs/03 §3.3）。
 *
 * マット推定モデル（人物 = MODNet、物体 = ISNet）の生の出力は、そのままでは使えない。
 *   ・境界が色の輪郭とずれる（入力を縮小して推論するため）
 *   ・背景の一部を拾った小さな島が残る
 *   ・被写体の内部に穴が空く（黒い服、影）
 *   ・上の2つを二値処理で潰すと輪郭が階段状になる
 *
 * 順に「ガイデッドフィルタ → 小連結成分除去 → 内部の穴埋め → 外側2pxの軟化」で直す。
 * 全て CPU で、1024² で合計 60ms 程度。GPU に載せる価値が出るほど重くない。
 */

/** α をこの値以上なら被写体とみなす。二値処理の閾値。 */
export const SUBJECT_ALPHA = 128;

/**
 * 埋めてよい内部の穴の上限（被写体の画素数に対する割合）。
 *
 * 実測で、埋めたい穴（マットの抜け）は被写体の 0.01%、埋めてはいけない
 * 隙間（両脚の間）は 3.41% だった。3 桁の隔たりの中央あたりに置く。
 */
const DEFAULT_MAX_HOLE_RATIO = 0.005;

/**
 * 割合に関わらず埋めてよい穴の大きさ（画素）。
 *
 * 被写体が極端に小さいとき（検査用の合成画像や縮小プレビュー）、割合だけだと
 * 数画素の抜けも「大きな隙間」に見えてしまう。この大きさの穴が実際の隙間で
 * あることは無いので、下限として置く。
 */
const MIN_FILL_PIXELS = 16;

// --- 箱フィルタ（積分画像） -------------------------------------------------

/**
 * 半径 r の箱フィルタ。積分画像を使うので半径によらず O(n)。
 *
 * 端は「はみ出した分を切り詰めて、実際に足した画素数で割る」。0 で埋めると
 * 縁が暗くなり、そのぶん境界が内側にずれてしまう。
 */
export function boxFilter(
  src: ArrayLike<number>,
  width: number,
  height: number,
  radius: number,
): Float32Array {
  const sat = new Float64Array((width + 1) * (height + 1));
  for (let y = 0; y < height; y++) {
    let rowSum = 0;
    for (let x = 0; x < width; x++) {
      rowSum += src[y * width + x] as number;
      sat[(y + 1) * (width + 1) + x + 1] = (sat[y * (width + 1) + x + 1] as number) + rowSum;
    }
  }

  const out = new Float32Array(width * height);
  for (let y = 0; y < height; y++) {
    const y0 = Math.max(0, y - radius);
    const y1 = Math.min(height - 1, y + radius);
    for (let x = 0; x < width; x++) {
      const x0 = Math.max(0, x - radius);
      const x1 = Math.min(width - 1, x + radius);
      const s =
        (sat[(y1 + 1) * (width + 1) + x1 + 1] as number) -
        (sat[y0 * (width + 1) + x1 + 1] as number) -
        (sat[(y1 + 1) * (width + 1) + x0] as number) +
        (sat[y0 * (width + 1) + x0] as number);
      out[y * width + x] = s / ((y1 - y0 + 1) * (x1 - x0 + 1));
    }
  }
  return out;
}

// --- ガイデッドフィルタ -----------------------------------------------------

export interface GuidedFilterParams {
  /** 窓の半径（画素）。docs/03 §3.3 は 4。 */
  readonly radius: number;
  /**
   * 正則化。小さいほど色の境界に強く貼りつくが、平坦部のノイズを拾う。
   * 色を 0〜1 に正規化した上での分散なので 1e-4 前後（＝色差 1% 相当）が目安。
   */
  readonly eps: number;
}

/**
 * カラーガイド版ガイデッドフィルタ（He et al., 2013）。
 *
 * α を写真そのものをガイドにして整える。マット推定モデルは入力を縮小して
 * 推論するので、返ってくる α の境界は色の輪郭から数画素ずれている。
 * このフィルタは「出力は窓ごとにガイドの線形関数」という仮定を置くので、
 * 色が変わる位置で α も変わるようになり、境界が写真の輪郭に吸い付く。
 *
 * グレースケールでなくカラーで解くのは、髪と背景のように**明度が近く色相が違う**
 * 境界を拾うため。3×3 の対称行列を画素ごとに解く必要があるが、その価値がある。
 *
 * 効き方には非対称があるので、期待しすぎないこと（測って確かめた）。
 * α が背景側へはみ出している側は強く削れる（ガイドが暗い所で α も 0 に落ちる）が、
 * α が内側に足りない側は窓の中で「暗いガイドなのに α が高い」という説明を
 * 迫られ、切片が持ち上がって数画素にじむ。3px ずらした α では、片側は完全に
 * 輪郭へ吸い付いたが、反対側は 8px ほど尾を引いた。
 * つまりこれは**ぼけを締める道具であって、位置ずれを直す道具ではない**。
 * 大きな位置ずれは推論解像度を上げて防ぐ（docs/03 §3.2 の 1024² 化）。
 *
 * @param alpha  0〜255 の α。長さ width×height。
 * @param rgb    ガイド画像。RGBA8 が width×height×4 で並んだもの（A は使わない）。
 */
export function guidedFilterColor(
  alpha: ArrayLike<number>,
  rgb: ArrayLike<number>,
  width: number,
  height: number,
  params: GuidedFilterParams,
): Uint8ClampedArray {
  const n = width * height;
  const { radius: r, eps } = params;

  const ir = new Float32Array(n);
  const ig = new Float32Array(n);
  const ib = new Float32Array(n);
  const p = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    ir[i] = (rgb[i * 4] as number) / 255;
    ig[i] = (rgb[i * 4 + 1] as number) / 255;
    ib[i] = (rgb[i * 4 + 2] as number) / 255;
    p[i] = (alpha[i] as number) / 255;
  }

  const meanR = boxFilter(ir, width, height, r);
  const meanG = boxFilter(ig, width, height, r);
  const meanB = boxFilter(ib, width, height, r);
  const meanP = boxFilter(p, width, height, r);

  const prod = (a: Float32Array, b: Float32Array): Float32Array => {
    const t = new Float32Array(n);
    for (let i = 0; i < n; i++) t[i] = (a[i] as number) * (b[i] as number);
    return boxFilter(t, width, height, r);
  };

  // 共分散 cov(I, p) と、ガイドの分散共分散行列（対称なので上三角の6要素）
  const covRP = prod(ir, p);
  const covGP = prod(ig, p);
  const covBP = prod(ib, p);
  const varRR = prod(ir, ir);
  const varRG = prod(ir, ig);
  const varRB = prod(ir, ib);
  const varGG = prod(ig, ig);
  const varGB = prod(ig, ib);
  const varBB = prod(ib, ib);

  const aR = new Float32Array(n);
  const aG = new Float32Array(n);
  const aB = new Float32Array(n);
  const b = new Float32Array(n);

  for (let i = 0; i < n; i++) {
    const mr = meanR[i] as number;
    const mg = meanG[i] as number;
    const mb = meanB[i] as number;
    const mp = meanP[i] as number;

    const crp = (covRP[i] as number) - mr * mp;
    const cgp = (covGP[i] as number) - mg * mp;
    const cbp = (covBP[i] as number) - mb * mp;

    // Σ + eps·E
    const s11 = (varRR[i] as number) - mr * mr + eps;
    const s12 = (varRG[i] as number) - mr * mg;
    const s13 = (varRB[i] as number) - mr * mb;
    const s22 = (varGG[i] as number) - mg * mg + eps;
    const s23 = (varGB[i] as number) - mg * mb;
    const s33 = (varBB[i] as number) - mb * mb + eps;

    // 3×3 対称行列の逆行列を余因子で解く
    const c11 = s22 * s33 - s23 * s23;
    const c12 = s13 * s23 - s12 * s33;
    const c13 = s12 * s23 - s13 * s22;
    const det = s11 * c11 + s12 * c12 + s13 * c13;

    if (Math.abs(det) < 1e-12) {
      // 窓の中の色がほぼ一様だと退化する。そこでは平均で置き換える（＝ただの箱フィルタ）。
      aR[i] = 0;
      aG[i] = 0;
      aB[i] = 0;
      b[i] = mp;
      continue;
    }

    const c22 = s11 * s33 - s13 * s13;
    const c23 = s13 * s12 - s11 * s23;
    const c33 = s11 * s22 - s12 * s12;
    const inv = 1 / det;

    const ar = (c11 * crp + c12 * cgp + c13 * cbp) * inv;
    const ag = (c12 * crp + c22 * cgp + c23 * cbp) * inv;
    const ab = (c13 * crp + c23 * cgp + c33 * cbp) * inv;
    aR[i] = ar;
    aG[i] = ag;
    aB[i] = ab;
    b[i] = mp - ar * mr - ag * mg - ab * mb;
  }

  const mAR = boxFilter(aR, width, height, r);
  const mAG = boxFilter(aG, width, height, r);
  const mAB = boxFilter(aB, width, height, r);
  const mB = boxFilter(b, width, height, r);

  const out = new Uint8ClampedArray(n);
  for (let i = 0; i < n; i++) {
    const v =
      (mAR[i] as number) * (ir[i] as number) +
      (mAG[i] as number) * (ig[i] as number) +
      (mAB[i] as number) * (ib[i] as number) +
      (mB[i] as number);
    out[i] = Math.round(v * 255);
  }
  return out;
}

// --- 連結成分 ---------------------------------------------------------------

export interface ComponentStats {
  /** 各画素のラベル。0 は背景。 */
  readonly labels: Int32Array;
  /** ラベルごとの画素数。添字 0 は使わない。 */
  readonly sizes: number[];
}

/**
 * 8近傍の連結成分ラベリング。明示的なスタックで反復的に塗る
 * （再帰にすると 1024² の被写体でスタックが溢れる）。
 */
export function connectedComponents(
  alpha: ArrayLike<number>,
  width: number,
  height: number,
  threshold = SUBJECT_ALPHA,
): ComponentStats {
  const labels = new Int32Array(width * height);
  const sizes: number[] = [0];
  const stack: number[] = [];
  let next = 1;

  for (let start = 0; start < width * height; start++) {
    if (labels[start] !== 0 || (alpha[start] as number) < threshold) continue;
    const label = next++;
    let count = 0;
    stack.push(start);
    labels[start] = label;

    while (stack.length > 0) {
      const i = stack.pop() as number;
      count++;
      const x = i % width;
      const y = (i / width) | 0;
      for (let dy = -1; dy <= 1; dy++) {
        const ny = y + dy;
        if (ny < 0 || ny >= height) continue;
        for (let dx = -1; dx <= 1; dx++) {
          const nx = x + dx;
          if (nx < 0 || nx >= width) continue;
          const j = ny * width + nx;
          if (labels[j] !== 0 || (alpha[j] as number) < threshold) continue;
          labels[j] = label;
          stack.push(j);
        }
      }
    }
    sizes.push(count);
  }
  return { labels, sizes };
}

/**
 * 最大成分に対して `minRatio` 未満の島を消す（docs/03 §3.3 は 2%）。
 *
 * 背景に紛れ込んだ小片をそのまま3D化すると、被写体から離れた空中に
 * 板が浮く。これは「立体になっていない」と一目で分かる壊れ方をする。
 */
export function removeSmallComponents(
  alpha: Uint8ClampedArray,
  width: number,
  height: number,
  minRatio = 0.02,
): Uint8ClampedArray {
  const { labels, sizes } = connectedComponents(alpha, width, height);
  if (sizes.length <= 1) return alpha;

  let largest = 0;
  for (let l = 1; l < sizes.length; l++) largest = Math.max(largest, sizes[l] as number);
  const minSize = largest * minRatio;

  const out = new Uint8ClampedArray(alpha);
  for (let i = 0; i < labels.length; i++) {
    const l = labels[i] as number;
    if (l !== 0 && (sizes[l] as number) < minSize) out[i] = 0;
  }
  return out;
}

/**
 * 被写体内部の穴を埋める。**ただし小さいものだけ**（v2.6.4、docs/09 §V17）。
 *
 * 背景側から塗りつぶして、どこからも到達できない背景画素＝内部の穴とみなす。
 * 黒い服や濃い影でマット推定が抜けることがあり、そのまま3D化すると
 * 被写体を貫通する窓が空く。
 *
 * **大きさの上限が要る。** 直立した人物では、両脚の間の隙間が股と足先で
 * 閉じるため位相的に「内部の穴」になり、外周からの塗りつぶしが届かない。
 * 実写（立ち姿）で MODNet も u2netp も脚の間を 6〜20px 正しく開けていたのに、
 * ここで塞いでいた。塞いだ帯には床の深度が入るので、脚の間に板が張る。
 *
 * 埋めるべき穴と埋めてはいけない隙間は、大きさが桁で違う。同じ写真での実測:
 *
 * | | 大きさ | 被写体に対する割合 | 外接 |
 * |---|---|---|---|
 * | 脚の間（埋めてはいけない） | 4,418 px | **3.41%** | 22×347 |
 * | 本物の穴（埋めたい） | 13 / 13 / 1 px | **0.01%** | 2×8 など |
 *
 * 3 桁離れているので、割合の閾値ひとつで分けられる。
 *
 * 埋める値は 255。半端な値で埋めると、そこだけ半透明の膜が張って見える。
 *
 * @param maxHoleRatio 被写体の画素数に対するこの割合を超える穴は埋めない。
 *                     0 以下ならすべて埋める（v2.6.3 までの挙動）。
 */
export function fillInteriorHoles(
  alpha: Uint8ClampedArray,
  width: number,
  height: number,
  threshold = SUBJECT_ALPHA,
  maxHoleRatio = DEFAULT_MAX_HOLE_RATIO,
): Uint8ClampedArray {
  const n = width * height;
  const reached = new Uint8Array(n);
  const stack: number[] = [];

  const push = (i: number): void => {
    if (reached[i] === 1 || (alpha[i] as number) >= threshold) return;
    reached[i] = 1;
    stack.push(i);
  };

  for (let x = 0; x < width; x++) {
    push(x);
    push((height - 1) * width + x);
  }
  for (let y = 0; y < height; y++) {
    push(y * width);
    push(y * width + width - 1);
  }

  while (stack.length > 0) {
    const i = stack.pop() as number;
    const x = i % width;
    const y = (i / width) | 0;
    // 穴埋めは4近傍で塗る。8近傍だと、斜めに1画素だけ触れている輪郭の
    // 隙間から外へ漏れて、埋めたい穴が埋まらない。
    if (x > 0) push(i - 1);
    if (x < width - 1) push(i + 1);
    if (y > 0) push(i - width);
    if (y < height - 1) push(i + width);
  }

  const out = new Uint8ClampedArray(alpha);
  if (maxHoleRatio <= 0) {
    for (let i = 0; i < n; i++) {
      if (reached[i] === 0 && (alpha[i] as number) < threshold) out[i] = 255;
    }
    return out;
  }

  // 穴を連結成分に分け、大きすぎるものは残す。
  let subject = 0;
  for (let i = 0; i < n; i++) if ((alpha[i] as number) >= threshold) subject++;
  const limit = Math.max(subject * maxHoleRatio, MIN_FILL_PIXELS);

  const seen = new Uint8Array(n);
  const group: number[] = [];
  for (let start = 0; start < n; start++) {
    if (seen[start] === 1 || reached[start] === 1) continue;
    if ((alpha[start] as number) >= threshold) continue;
    // 幅優先で 1 つの穴を集める
    group.length = 0;
    seen[start] = 1;
    group.push(start);
    for (let head = 0; head < group.length; head++) {
      const i = group[head] as number;
      const x = i % width;
      const y = (i / width) | 0;
      const visit = (j: number): void => {
        if (seen[j] === 1 || reached[j] === 1) return;
        if ((alpha[j] as number) >= threshold) return;
        seen[j] = 1;
        group.push(j);
      };
      if (x > 0) visit(i - 1);
      if (x < width - 1) visit(i + 1);
      if (y > 0) visit(i - width);
      if (y < height - 1) visit(i + width);
    }
    if (group.length > limit) continue; // 脚の間のような大きな隙間は残す
    for (const i of group) out[i] = 255;
  }
  return out;
}

/**
 * 輪郭の階段を均す（docs/03 §3.3「外側2pxの軟化」）。
 *
 * 小成分除去と穴埋めは二値処理なので、触った場所の輪郭が階段状になる。
 * そこで**輪郭の近傍だけ**を弱くぼかす。画面全体をぼかしてはいけない。
 * 髪の毛のような、モデルが出した本物の半透明部分まで潰れてしまう。
 *
 * 帯の内側では元の α を残す割合を上げる。被写体の中身は触らない。
 */
export function softenEdge(
  alpha: Uint8ClampedArray,
  width: number,
  height: number,
  band = 2,
): Uint8ClampedArray {
  const blurred = boxFilter(alpha, width, height, 1);
  const out = new Uint8ClampedArray(alpha);

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const i = y * width + x;
      // 帯の中かどうかは「近傍に自分と反対側の画素があるか」で判定する。
      // 距離変換を回すほどの精度は要らない。
      let hasFg = false;
      let hasBg = false;
      for (let dy = -band; dy <= band && !(hasFg && hasBg); dy++) {
        const ny = y + dy;
        if (ny < 0 || ny >= height) continue;
        for (let dx = -band; dx <= band; dx++) {
          const nx = x + dx;
          if (nx < 0 || nx >= width) continue;
          if ((alpha[ny * width + nx] as number) >= SUBJECT_ALPHA) hasFg = true;
          else hasBg = true;
          if (hasFg && hasBg) break;
        }
      }
      if (hasFg && hasBg) out[i] = Math.round(blurred[i] as number);
    }
  }
  return out;
}

export interface MatteRefineParams {
  readonly guided: GuidedFilterParams;
  /** 最大成分に対するこの割合未満の島を消す。 */
  readonly minComponentRatio: number;
  /** 軟化する帯の幅（画素）。 */
  readonly softenBand: number;
}

export const DEFAULT_MATTE_PARAMS: MatteRefineParams = {
  guided: { radius: 4, eps: 1e-4 },
  minComponentRatio: 0.02,
  softenBand: 2,
};

/**
 * 「領域の門」でマットを絞る（docs/03 §3.3.1、v2.5）。
 *
 * `fine` の輪郭の細かさを保ったまま、`gate` が被写体と認めない場所を落とす。
 * 門は少しだけ膨らませる。輪郭を出すモデルのほうがわずかに外側までシルエットを
 * 取るので、そのぶんを削らないため。
 *
 * 実写では、床に座る人物で MODNet が敷物を α=254 で被写体に含めていた
 * （モデル全体の 28%）。u2netp を門にすると敷物が消え、輪郭の細かさは
 * MODNet のまま残る。
 *
 * @param dilate 門を広げる画素数。0 なら門そのまま。
 */
export function applyMatteGate(
  fine: ArrayLike<number>,
  gate: ArrayLike<number>,
  width: number,
  height: number,
  dilate = 4,
  threshold = 128,
): Uint8ClampedArray {
  const open = new Uint8Array(width * height);
  for (let i = 0; i < open.length; i++) open[i] = (gate[i] as number) >= threshold ? 1 : 0;

  const grown = dilate > 0 ? dilateMask(open, width, height, dilate) : open;
  const out = new Uint8ClampedArray(width * height);
  for (let i = 0; i < out.length; i++) out[i] = grown[i] ? (fine[i] as number) : 0;
  return out;
}

/** 二値マスクを r 画素ぶん膨らませる（分離可能な最大値フィルタ）。 */
function dilateMask(mask: Uint8Array, width: number, height: number, r: number): Uint8Array {
  const tmp = new Uint8Array(width * height);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      let v = 0;
      for (let k = -r; k <= r && !v; k++) {
        const nx = x + k;
        if (nx >= 0 && nx < width && mask[y * width + nx]) v = 1;
      }
      tmp[y * width + x] = v;
    }
  }
  const out = new Uint8Array(width * height);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      let v = 0;
      for (let k = -r; k <= r && !v; k++) {
        const ny = y + k;
        if (ny >= 0 && ny < height && tmp[ny * width + x]) v = 1;
      }
      out[y * width + x] = v;
    }
  }
  return out;
}

/**
 * docs/03 §3.3 の後処理を順に適用する。
 *
 * 順序に意味がある。ガイデッドフィルタを先にかけるのは、境界を写真の輪郭に
 * 合わせてから二値処理をしたいため。逆にすると、ずれた位置で穴を埋めたり
 * 島を消したりして、その誤りがフィルタ後も残る。
 * 軟化を最後に置くのは、二値処理が作った階段を均すのが目的だから。
 */
export function refineMatte(
  alpha: ArrayLike<number>,
  rgb: ArrayLike<number>,
  width: number,
  height: number,
  params: MatteRefineParams = DEFAULT_MATTE_PARAMS,
): Uint8ClampedArray {
  let a = guidedFilterColor(alpha, rgb, width, height, params.guided);
  a = removeSmallComponents(a, width, height, params.minComponentRatio);
  a = fillInteriorHoles(a, width, height);
  a = softenEdge(a, width, height, params.softenBand);
  return a;
}
