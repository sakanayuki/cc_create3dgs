// サーフェルのラスタライズ（docs/06 §6.3 ⑥）。
//
// ソート済みインデックスの順にインスタンス描画し、各スプラットを画面空間の
// 楕円としてクアッドに載せて α ブレンドする。ソートは遠→近なので "over" で正しい。

@group(0) @binding(1) var<uniform>       cam:    Camera;
@group(0) @binding(2) var<storage, read> sorted: array<u32>;

struct VSOut {
  @builtin(position) pos: vec4<f32>,
  // クアッド内の局所座標。ガウシアンの減衰を測る（±2σ を ±1 に正規化）
  @location(0) local: vec2<f32>,
  @location(1) color: vec4<f32>,
};

// サーフェルの厚み ÷ 面内の半径。SplatRenderer.ts の SURFEL_FLATNESS と
// 同じ値でなければならない（exportSplats.test.ts が確かめる）。
const SURFEL_FLATNESS: f32 = 0.35;

// 法線から接平面の正規直交基底を作る（Duff らの分岐なし ONB）。
fn onb(n: vec3<f32>) -> mat2x3<f32> {
  let s = select(-1.0, 1.0, n.z >= 0.0);
  let a = -1.0 / (s + n.z);
  let b = n.x * n.y * a;
  let t1 = vec3<f32>(1.0 + s * n.x * n.x * a, s * b, -s * n.x);
  let t2 = vec3<f32>(b, s + n.y * n.y * a, -n.y);
  return mat2x3<f32>(t1, t2);
}

// クアッドの4隅（triangle-strip）
const CORNERS = array<vec2<f32>, 4>(
  vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0),
  vec2<f32>(-1.0,  1.0), vec2<f32>(1.0,  1.0),
);

@vertex
fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VSOut {
  var out: VSOut;
  let s = loadSplat(sorted[ii]);

  let clip = cam.viewProj * vec4<f32>(s.pos, 1.0);
  if (clip.w <= 0.0) {
    out.pos = vec4<f32>(0.0, 0.0, 2.0, 1.0);  // クリップ外へ飛ばす
    out.local = vec2<f32>(0.0);
    out.color = vec4<f32>(0.0);
    return out;
  }
  let center = clip.xy / clip.w;

  // 接平面の2軸と**厚み**を世界空間で作り、画面空間の楕円に射影する。
  //
  // 厚みを 0 にすると、寝たサーフェルが画面上で線に潰れて斜めの縞になる
  // （docs/09 §V15）。SURFEL_FLATNESS の注記に実測値がある。
  let n = decodeOct(s.nrm);
  let basis = onb(n);
  let sc = unpack2xf16(s.scale);
  let e1w = basis[0] * sc.x;
  let e2w = basis[1] * sc.y;
  let e3w = n * (SURFEL_FLATNESS * max(sc.x, sc.y));

  let p1 = cam.viewProj * vec4<f32>(s.pos + e1w, 1.0);
  let p2 = cam.viewProj * vec4<f32>(s.pos + e2w, 1.0);
  let p3 = cam.viewProj * vec4<f32>(s.pos + e3w, 1.0);
  let b1 = p1.xy / p1.w - center;
  let b2 = p2.xy / p2.w - center;
  let b3 = p3.xy / p3.w - center;

  // 画面上の 2×2 共分散 = 3 軸それぞれの射影の外積の和。
  // 3 軸は画面上では直交しないので、軸をそのままクアッドに使えない。
  let cxx = b1.x * b1.x + b2.x * b2.x + b3.x * b3.x;
  let cxy = b1.x * b1.y + b2.x * b2.y + b3.x * b3.y;
  let cyy = b1.y * b1.y + b2.y * b2.y + b3.y * b3.y;

  // 2×2 対称行列の固有分解（閉形式）。固有ベクトルは 2 通りの作り方が
  // あり、片方は退化する（cxy = 0 のとき）ので、長いほうを採る。
  let tr = cxx + cyy;
  let det = cxx * cyy - cxy * cxy;
  let disc = sqrt(max(tr * tr * 0.25 - det, 0.0));
  let lam1 = tr * 0.5 + disc;
  let lam2 = max(tr * 0.5 - disc, 0.0);
  var v1 = vec2<f32>(lam1 - cyy, cxy);
  if (dot(v1, v1) < 1e-20) { v1 = vec2<f32>(cxy, lam1 - cxx); }
  if (dot(v1, v1) < 1e-20) { v1 = vec2<f32>(1.0, 0.0); }
  v1 = normalize(v1);
  var a1 = v1 * sqrt(lam1);
  var a2 = vec2<f32>(-v1.y, v1.x) * sqrt(lam2);

  // Mip-Splatting 流の 2D 低域フィルタ（docs/06 §6.10）。
  // 画面上 1px を下回るサーフェルは点滅するので、最低でも filter2d 相当まで広げる。
  let px = 2.0 / cam.viewport;  // NDC における 1px
  let minAxis = px * cam.filter2d;
  let l1 = max(length(a1), 1e-8);
  let l2 = max(length(a2), 1e-8);
  a1 = a1 * max(1.0, minAxis.x / l1);
  a2 = a2 * max(1.0, minAxis.y / l2);

  // ±2σ までを描く。それ以上は寄与が 2% を切る。
  let k = 2.0;
  let corner = CORNERS[vi];
  let offset = (a1 * corner.x + a2 * corner.y) * k;

  out.pos = vec4<f32>(center + offset, clip.z / clip.w, 1.0);
  out.local = corner * k;
  out.color = unpackColor(s.color);
  return out;
}

@fragment
fn fs(in: VSOut) -> @location(0) vec4<f32> {
  let r2 = dot(in.local, in.local);
  if (r2 > 4.0) { discard; }
  let g = exp(-0.5 * r2);
  let a = in.color.a * g;
  if (a < 0.004) { discard; }
  return vec4<f32>(in.color.rgb, a);
}
