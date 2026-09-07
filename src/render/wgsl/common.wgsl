// 描画で共有する型とユーティリティ。
//
// スプラットは 24 バイトに詰める（docs/04 §4.8.1 の「展開しない」方針）。
//   pos   : vec3<f32>          12 B  被写体は単位立方体に正規化済み
//   nrm   : u32 (oct 16x2)      4 B  八面体写像。背面カリングとリム処理に使う
//   scale : u32 (f16 x2)        4 B  接平面内の2軸。サーフェルなので厚みは持たない
//   color : u32 (rgba8unorm)    4 B  SH 次数0（＝視点非依存）＋不透明度
struct Splat {
  pos: vec3<f32>,
  nrm: u32,
  scale: u32,
  color: u32,
};

struct Camera {
  viewProj: mat4x4<f32>,
  view: mat4x4<f32>,
  eye: vec3<f32>,
  focalPx: f32,
  viewport: vec2<f32>,
  nearZ: f32,
  farZ: f32,
  // 背面カリングの閾値。0 ちょうどで切るとシルエットが欠けるので少し裏まで残す。
  cullCos: f32,
  // Mip-Splatting 流の 2D 低域フィルタ。画素サイズ相当の分散を加える（docs/06 §6.10）。
  filter2d: f32,
  splatCount: u32,
  lodStride: u32,
};

// 八面体写像の復号。単位球面を 2D に写す、歪みの最も少ない写像。
fn decodeOct(packed: u32) -> vec3<f32> {
  let e = vec2<f32>(
    f32(packed & 0xffffu) / 65535.0 * 2.0 - 1.0,
    f32(packed >> 16u) / 65535.0 * 2.0 - 1.0,
  );
  var n = vec3<f32>(e.x, e.y, 1.0 - abs(e.x) - abs(e.y));
  let t = max(-n.z, 0.0);
  n = vec3<f32>(
    n.x + select(t, -t, n.x >= 0.0),
    n.y + select(t, -t, n.y >= 0.0),
    n.z,
  );
  return normalize(n);
}

fn unpack2xf16(v: u32) -> vec2<f32> {
  return unpack2x16float(v);
}

fn unpackColor(v: u32) -> vec4<f32> {
  return unpack4x8unorm(v);
}
