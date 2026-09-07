#version 300 es
// サーフェルのラスタライズ（WebGL2 版）。
//
// WGSL 版（src/render/wgsl/splat.wgsl）と同じ見え方になるよう、投影・楕円軸の計算・
// 2D 低域フィルタは同じ式を使う。違いは次の2点だけ。
//   ・ストレージバッファが無いので、スプラットを RGBA32UI テクスチャに置く
//   ・ソートは CPU で行い、その結果をインスタンス属性 aIndex として渡す
precision highp float;
precision highp int;
precision highp usampler2D;

// 1スプラット = 2テクセル
//   texel0: [posX, posY, posZ, normal(oct 16x2)]   ※位置は uintBitsToFloat で戻す
//   texel1: [scale(f16 x2), color(rgba8), 0, 0]
uniform usampler2D uSplats;
uniform int uTexWidth;

uniform mat4 uViewProj;
uniform vec2 uViewport;
uniform float uFilter2d;

in uint aIndex;   // インスタンスごと: ソート済みのスプラット番号
in vec2 aCorner;  // 頂点ごと: クアッドの隅 (±1)

out vec2 vLocal;
out vec4 vColor;

// 八面体写像の復号。WGSL 版の decodeOct と同じ。
vec3 decodeOct(uint packed) {
  vec2 e = vec2(
    float(packed & 0xffffu) / 65535.0 * 2.0 - 1.0,
    float(packed >> 16u) / 65535.0 * 2.0 - 1.0
  );
  vec3 n = vec3(e.x, e.y, 1.0 - abs(e.x) - abs(e.y));
  float t = max(-n.z, 0.0);
  n.x += n.x >= 0.0 ? -t : t;
  n.y += n.y >= 0.0 ? -t : t;
  return normalize(n);
}

// 法線から接平面の正規直交基底を作る（Duff らの分岐なし ONB）。WGSL 版と同じ。
void onb(vec3 n, out vec3 t1, out vec3 t2) {
  float s = n.z >= 0.0 ? 1.0 : -1.0;
  float a = -1.0 / (s + n.z);
  float b = n.x * n.y * a;
  t1 = vec3(1.0 + s * n.x * n.x * a, s * b, -s * n.x);
  t2 = vec3(b, s + n.y * n.y * a, -n.y);
}

void main() {
  int t = int(aIndex) * 2;
  ivec2 uv0 = ivec2(t % uTexWidth, t / uTexWidth);
  ivec2 uv1 = ivec2((t + 1) % uTexWidth, (t + 1) / uTexWidth);
  uvec4 a = texelFetch(uSplats, uv0, 0);
  uvec4 b = texelFetch(uSplats, uv1, 0);

  vec3 pos = vec3(uintBitsToFloat(a.x), uintBitsToFloat(a.y), uintBitsToFloat(a.z));
  vec3 nrm = decodeOct(a.w);
  vec2 scale = unpackHalf2x16(b.x);
  vec4 color = unpackUnorm4x8(b.y);

  vec4 clip = uViewProj * vec4(pos, 1.0);
  if (clip.w <= 0.0) {
    gl_Position = vec4(0.0, 0.0, 2.0, 1.0);  // クリップ外へ飛ばす
    vLocal = vec2(0.0);
    vColor = vec4(0.0);
    return;
  }
  vec2 center = clip.xy / clip.w;

  vec3 e1, e2;
  onb(nrm, e1, e2);
  vec4 p1 = uViewProj * vec4(pos + e1 * scale.x, 1.0);
  vec4 p2 = uViewProj * vec4(pos + e2 * scale.y, 1.0);
  vec2 a1 = p1.xy / p1.w - center;
  vec2 a2 = p2.xy / p2.w - center;

  // Mip-Splatting 流の 2D 低域フィルタ。1px を下回るサーフェルの点滅を抑える。
  vec2 px = 2.0 / uViewport;
  vec2 minAxis = px * uFilter2d;
  a1 *= max(1.0, minAxis.x / max(length(a1), 1e-8));
  a2 *= max(1.0, minAxis.y / max(length(a2), 1e-8));

  const float k = 2.0;  // ±2σ まで描く。それ以上は寄与が 2% を切る
  vec2 offset = (a1 * aCorner.x + a2 * aCorner.y) * k;

  gl_Position = vec4(center + offset, clip.z / clip.w, 1.0);
  vLocal = aCorner * k;
  vColor = color;
}
