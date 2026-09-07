// 深度ソート（docs/06 §6.5）。
//
// 一般の 3DGS は基数ソートを使うが、本設計は被写体が単位立方体に正規化されていて
// 深度レンジが既知なので、より単純な「深度バケットの計数ソート」で足りる。
//   ・1パスで済む（基数ソートは 4 パス）
//   ・大域プレフィックス和が 8192 要素の 1 ワークグループに収まる
//   ・バケット幅は被写体 1m で約 0.12mm。サーフェル 1 個の大きさより細かいので、
//     バケット内の順序が不定でも α ブレンドの見た目に影響しない
//
// 3 つのエントリポイントを順に dispatch する: histogram → scan → scatter

const BUCKETS: u32 = 8192u;
const SCAN_THREADS: u32 = 256u;
const PER_THREAD: u32 = BUCKETS / SCAN_THREADS;  // 32

@group(0) @binding(1) var<uniform>             cam:       Camera;
@group(0) @binding(2) var<storage, read_write> histogram: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> offsets:   array<u32>;
@group(0) @binding(4) var<storage, read_write> sorted:    array<u32>;
// visible は描画の indirect 引数そのものを兼ねる:
//   [0]=vertexCount(4), [1]=instanceCount(可視数), [2]=firstVertex, [3]=firstInstance
// 可視数のカウンタがそのままインスタンス数になるので、CPU への読み戻しが要らない。
@group(0) @binding(5) var<storage, read_write> visible:   array<atomic<u32>>;

// 深度をバケット番号に変換する。遠いものほど小さい番号（＝先に描く）。
fn bucketOf(depth: f32) -> u32 {
  let t = clamp((depth - cam.nearZ) / max(cam.farZ - cam.nearZ, 1e-6), 0.0, 1.0);
  // 遠い→手前 の順に並べたいので反転する
  return min(u32((1.0 - t) * f32(BUCKETS - 1u)), BUCKETS - 1u);
}

// 描画対象かどうか。背面カリングと視錐台カリング（docs/06 §6.4）。
fn isVisible(i: u32, depth: ptr<function, f32>) -> bool {
  let s = loadSplat(i);
  let toEye = cam.eye - s.pos;
  let d = length(toEye);
  if (d < 1e-6) { return false; }

  // サーフェルなので法線で表裏が決まる。通常の 3DGS には無い最適化。
  let n = decodeOct(s.nrm);
  if (dot(n, toEye / d) < cam.cullCos) { return false; }

  let clip = cam.viewProj * vec4<f32>(s.pos, 1.0);
  if (clip.w <= 0.0) { return false; }
  let ndc = clip.xyz / clip.w;
  // 少し広めに取る。縁のサーフェルは中心が画面外でも寄与するため。
  if (any(abs(ndc.xy) > vec2<f32>(1.3))) { return false; }

  let vz = (cam.view * vec4<f32>(s.pos, 1.0)).z;
  *depth = -vz;
  return true;
}

@compute @workgroup_size(256)
fn clearHistogram(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i < BUCKETS) {
    atomicStore(&histogram[i], 0u);
    offsets[i] = 0u;
  }
  if (i == 0u) {
    atomicStore(&visible[0], 4u);  // vertexCount: クアッドは triangle-strip の4頂点
    atomicStore(&visible[1], 0u);  // instanceCount: これから数える
    atomicStore(&visible[2], 0u);
    atomicStore(&visible[3], 0u);
  }
}

@compute @workgroup_size(256)
fn histogramPass(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x * cam.lodStride;
  if (i >= cam.splatCount) { return; }
  var depth: f32 = 0.0;
  if (!isVisible(i, &depth)) { return; }
  atomicAdd(&histogram[bucketOf(depth)], 1u);
  atomicAdd(&visible[1], 1u);
}

// 8192 バケットの排他的プレフィックス和を 1 ワークグループで行う。
// 各スレッドが 32 個を直列に合計 → 256 個の部分和を Hillis-Steele で走査 → 書き戻し。
var<workgroup> partial: array<u32, SCAN_THREADS>;

@compute @workgroup_size(SCAN_THREADS)
fn scanPass(@builtin(local_invocation_id) lid: vec3<u32>) {
  let t = lid.x;
  let base = t * PER_THREAD;

  var sum: u32 = 0u;
  for (var k: u32 = 0u; k < PER_THREAD; k = k + 1u) {
    sum = sum + atomicLoad(&histogram[base + k]);
  }
  partial[t] = sum;
  workgroupBarrier();

  // 包括的走査 → 排他的に直す
  for (var stride: u32 = 1u; stride < SCAN_THREADS; stride = stride * 2u) {
    var v: u32 = 0u;
    if (t >= stride) { v = partial[t - stride]; }
    workgroupBarrier();
    if (t >= stride) { partial[t] = partial[t] + v; }
    workgroupBarrier();
  }

  var running: u32 = 0u;
  if (t > 0u) { running = partial[t - 1u]; }

  for (var k: u32 = 0u; k < PER_THREAD; k = k + 1u) {
    offsets[base + k] = running;
    running = running + atomicLoad(&histogram[base + k]);
  }
}

// scatter では offsets を書き込みカーソルとして使う。atomicAdd で位置を取る。
@group(0) @binding(6) var<storage, read_write> cursor: array<atomic<u32>>;

@compute @workgroup_size(256)
fn initCursor(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i < BUCKETS) { atomicStore(&cursor[i], offsets[i]); }
}

@compute @workgroup_size(256)
fn scatterPass(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x * cam.lodStride;
  if (i >= cam.splatCount) { return; }
  var depth: f32 = 0.0;
  if (!isVisible(i, &depth)) { return; }
  let slot = atomicAdd(&cursor[bucketOf(depth)], 1u);
  sorted[slot] = i;
}
