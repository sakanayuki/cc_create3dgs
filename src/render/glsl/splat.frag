#version 300 es
precision highp float;

in vec2 vLocal;
in vec4 vColor;
out vec4 fragColor;

void main() {
  float r2 = dot(vLocal, vLocal);
  if (r2 > 4.0) discard;
  float g = exp(-0.5 * r2);
  float a = vColor.a * g;
  if (a < 0.004) discard;
  fragColor = vec4(vColor.rgb, a);
}
