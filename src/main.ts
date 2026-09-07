/**
 * 本体アプリのエントリ。
 *
 * 生成パイプラインは PoC-1 の結果待ち（docs/08 §8.11）なので、いまは端末の能力を
 * 判定して、PoC-1 への導線を出す役目だけを持つ。
 */
import { detectCapability, describeCapability } from './runtime/capability';

async function main(): Promise<void> {
  const cap = await detectCapability();
  const card = document.querySelector('.card.accent');
  if (!card) return;

  const note = document.createElement('p');
  note.style.marginTop = '10px';
  note.style.fontSize = '12.5px';
  note.className = 'mono';

  if (cap.webgpu.supported) {
    note.textContent =
      `この端末: WebGPU 利用可 / 作業グリッド ${cap.workingGrid}² ` +
      `${cap.webgpu.vendor ? `/ ${cap.webgpu.vendor}` : ''}`;
  } else {
    note.textContent = `この端末: WebGPU 利用不可（${cap.webgpu.reason ?? '理由不明'}）`;
  }
  card.appendChild(note);

  // 詳細はコンソールに出す。実機でのデバッグ用。
  console.info('[PhotoSplat] 端末の能力\n' + describeCapability(cap).join('\n'));
}

void main();
