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

  // 接平面の2軸を世界空間で作り、画面空間の楕円軸に射影する
  let n = decodeOct(s.nrm);
  let basis = onb(n);
  let sc = unpack2xf16(s.scale);
  let e1w = basis[0] * sc.x;
  let e2w = basis[1] * sc.y;

  let p1 = cam.viewProj * vec4<f32>(s.pos + e1w, 1.0);
  let p2 = cam.viewProj * vec4<f32>(s.pos + e2w, 1.0);
  var a1 = p1.xy / p1.w - center;
  var a2 = p2.xy / p2.w - center;

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
