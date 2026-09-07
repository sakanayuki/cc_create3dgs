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

// スプラット配列は「生の u32 列」として受け、手で切り出す。
//
// array<Splat> にしてはいけない。vec3<f32> を含む構造体は整列規則で
// 16 バイト境界に切り上げられ、配列のストライドが 24 ではなく 32 バイトになる。
// CPU 側（SPLAT_BYTES = 24）とずれるので、3個に1個だけ偶然正しい位置に当たり、
// 残りは隣のスプラットの断片を読む。球は球らしく見えたまま画面全体に薄い靄と
// 筋が乗る、という気付きにくい壊れ方をする。ピクセルを読み戻して初めて分かった。
const SPLAT_WORDS: u32 = 6u;  // 24 バイト = u32 6 個

@group(0) @binding(0) var<storage, read> splatWords: array<u32>;

fn loadSplat(i: u32) -> Splat {
  let o = i * SPLAT_WORDS;
  var s: Splat;
  s.pos = vec3<f32>(
    bitcast<f32>(splatWords[o]),
    bitcast<f32>(splatWords[o + 1u]),
    bitcast<f32>(splatWords[o + 2u]),
  );
  s.nrm = splatWords[o + 3u];
  s.scale = splatWords[o + 4u];
  s.color = splatWords[o + 5u];
  return s;
}

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
