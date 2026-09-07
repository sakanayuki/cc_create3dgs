/**
 * 本体アプリのエントリ。
 *
 * することは2つだけ。
 *   1. マルチスレッド化の下ごしらえ（決定 D20）。Service Worker が
 *      ページを制御し始めるまで待ってから読み込み直す必要がある。
 *   2. 画面の組み立て（src/ui/app.ts）
 *
 * 1 を先に済ませないと、SharedArrayBuffer が無いまま ORT が初期化されて
 * 単スレッドに固定される。
 */
import { ensureCrossOriginIsolation } from './runtime/crossOriginIsolation';
import { mountApp } from './ui/app';

async function main(): Promise<void> {
  // isolation が成立していなければ Service Worker を入れて読み込み直す。
  // 成立済み・不要・失敗のいずれでもここは戻ってくる。
  await ensureCrossOriginIsolation();
  mountApp();
}

void main();
