/**
 * `.pgs`（Photo Gaussian Splat）コンテナの読み書き（docs/05 §5.2）。
 *
 * ZIP も外部ライブラリも使わない素朴な連結形式。パーサは50行程度で書ける。
 *
 *   0    8  マジック "PGSPLAT\0"
 *   8    2  メジャーバージョン (u16 LE)
 *   10   2  マイナーバージョン (u16 LE)
 *   12   4  マニフェスト長 M (u32 LE)
 *   16   M  マニフェスト (UTF-8 JSON)
 *   16+M    チャンク列: [4 ID][4 長さ][ペイロード]
 *
 * ペイロードは **PNG / WebP のバイト列そのもの**。読み込み側は
 * `createImageBitmap(new Blob([payload]))` に渡すだけでよく、デコードは
 * ブラウザのネイティブ実装が担う。自前のデコーダを一切書かない。
 *
 * この層は画像の中身を知らない純粋なコンテナなので、Node でも単体テストできる。
 */

export const PGS_MAGIC = 'PGSPLAT\0';
export const PGS_VERSION = { major: 1, minor: 0 } as const;

/** チャンク ID。未知の ID は読み飛ばす（前方互換）。 */
export type ChunkId =
  | 'DPTH' // 深度の上位 8bit
  | 'DPTL' // 深度の下位 4bit
  | 'COLR' // 前面色 RGB
  | 'ALFA' // 不透明度 兼 占有マスク
  | 'BCOL' // 背面色 RGB（半解像度）
  | 'INPT' // 遮蔽部の補完テクスチャ RGBA（半解像度）
  | 'THIK' // 厚みマップ（手で編集した場合のみ）
  | 'THMB'; // サムネイル

/** 必須チャンク。欠けていたら読み込みを拒否する。 */
export const REQUIRED_CHUNKS: readonly ChunkId[] = ['DPTH', 'DPTL', 'COLR', 'ALFA', 'BCOL'];

export interface PgsFile {
  readonly version: { major: number; minor: number };
  readonly manifest: Record<string, unknown>;
  readonly chunks: ReadonlyMap<string, Uint8Array>;
  /** 未知のチャンク ID。読み飛ばしたことを呼び出し側に伝える。 */
  readonly unknownChunks: readonly string[];
}

const enc = new TextEncoder();
const dec = new TextDecoder();

function writeAscii4(view: Uint8Array, offset: number, id: string): void {
  for (let i = 0; i < 4; i++) view[offset + i] = id.charCodeAt(i) & 0xff;
}

function readAscii(bytes: Uint8Array, offset: number, length: number): string {
  let s = '';
  for (let i = 0; i < length; i++) s += String.fromCharCode(bytes[offset + i] ?? 0);
  return s;
}

/** マニフェストとチャンク群を1つのバイト列にまとめる。 */
export function packPgs(
  manifest: Record<string, unknown>,
  chunks: ReadonlyMap<string, Uint8Array>,
): Uint8Array {
  const manifestBytes = enc.encode(JSON.stringify(manifest));
  let total = 16 + manifestBytes.length;
  for (const payload of chunks.values()) total += 8 + payload.length;

  const out = new Uint8Array(total);
  const dv = new DataView(out.buffer);

  writeAscii4(out, 0, PGS_MAGIC.slice(0, 4));
  writeAscii4(out, 4, PGS_MAGIC.slice(4, 8));
  dv.setUint16(8, PGS_VERSION.major, true);
  dv.setUint16(10, PGS_VERSION.minor, true);
  dv.setUint32(12, manifestBytes.length, true);
  out.set(manifestBytes, 16);

  let off = 16 + manifestBytes.length;
  for (const [id, payload] of chunks) {
    if (id.length !== 4) throw new Error(`チャンク ID は4文字である必要があります: ${id}`);
    writeAscii4(out, off, id);
    dv.setUint32(off + 4, payload.length, true);
    out.set(payload, off + 8);
    off += 8 + payload.length;
  }
  return out;
}

/** バイト列を解釈する。壊れていれば理由を添えて例外を投げる。 */
export function unpackPgs(bytes: Uint8Array): PgsFile {
  if (bytes.length < 16) throw new Error('ファイルが短すぎます（ヘッダに満たない）');
  if (readAscii(bytes, 0, 8) !== PGS_MAGIC) throw new Error('.pgs ファイルではありません（マジックが不一致）');

  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const major = dv.getUint16(8, true);
  const minor = dv.getUint16(10, true);
  if (major > PGS_VERSION.major) {
    throw new Error(`このバージョンでは読めません（ファイル v${major}.${minor} / 対応 v${PGS_VERSION.major}.x）`);
  }

  const manifestLen = dv.getUint32(12, true);
  if (16 + manifestLen > bytes.length) throw new Error('マニフェストの長さが壊れています');

  let manifest: Record<string, unknown>;
  try {
    manifest = JSON.parse(dec.decode(bytes.subarray(16, 16 + manifestLen))) as Record<string, unknown>;
  } catch (e) {
    throw new Error(`マニフェストを解釈できません: ${String(e)}`);
  }

  const chunks = new Map<string, Uint8Array>();
  const unknownChunks: string[] = [];
  const known = new Set<string>(['DPTH', 'DPTL', 'COLR', 'ALFA', 'BCOL', 'INPT', 'THIK', 'THMB']);

  let off = 16 + manifestLen;
  while (off + 8 <= bytes.length) {
    const id = readAscii(bytes, off, 4);
    const len = dv.getUint32(off + 4, true);
    if (off + 8 + len > bytes.length) {
      throw new Error(`チャンク ${id} の長さが壊れています（宣言 ${len} バイト、残り ${bytes.length - off - 8} バイト）`);
    }
    // subarray なのでコピーしない。呼び出し側は Blob に渡すだけ。
    chunks.set(id, bytes.subarray(off + 8, off + 8 + len));
    if (!known.has(id)) unknownChunks.push(id);
    off += 8 + len;
  }
  if (off !== bytes.length) throw new Error('末尾に解釈できないバイトが残っています');

  return { version: { major, minor }, manifest, chunks, unknownChunks };
}

/** 必須チャンクが揃っているか確かめる。 */
export function missingRequiredChunks(file: PgsFile): ChunkId[] {
  return REQUIRED_CHUNKS.filter((id) => !file.chunks.has(id));
}
