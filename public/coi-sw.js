/**
 * COOP/COEP を付与する Service Worker。
 *
 * GitHub Pages はカスタム HTTP ヘッダを設定できないため、そのままでは
 * `crossOriginIsolated` にならず `SharedArrayBuffer` が使えない。
 * すると ONNX Runtime の WASM バックエンドが単スレッドに制限される。
 *
 * この Service Worker はレスポンスに COOP/COEP を足して isolation を成立させる。
 * 一般に coi-serviceworker と呼ばれる手法だが、外部依存を持ち込まずに
 * 中身を把握しておきたいので自前で持つ（40行程度）。
 *
 * 制約: COEP: require-corp のもとでは、クロスオリジンのサブリソースに
 * CORP か CORS が必要になる。本アプリのモデルは同一オリジン配信（決定 D7）なので
 * 影響しない。PoC が HuggingFace から直接取得する経路も、HF が
 * `access-control-allow-origin` を返すので CORS モードの fetch は通る（実測確認済み）。
 */

self.addEventListener('install', () => self.skipWaiting());

self.addEventListener('activate', (event) => event.waitUntil(self.clients.claim()));

self.addEventListener('message', (event) => {
  if (event.data?.type === 'deregister') {
    self.registration
      .unregister()
      .then(() => self.clients.matchAll())
      .then((clients) => clients.forEach((c) => c.navigate(c.url)));
  }
});

self.addEventListener('fetch', (event) => {
  const request = event.request;

  // Range リクエストや only-if-cached は触らない（触ると壊れる）
  if (request.cache === 'only-if-cached' && request.mode !== 'same-origin') return;

  event.respondWith(
    fetch(request)
      .then((response) => {
        // opaque レスポンスはヘッダを触れない。そのまま返す。
        if (response.status === 0) return response;

        const headers = new Headers(response.headers);
        headers.set('Cross-Origin-Embedder-Policy', 'require-corp');
        headers.set('Cross-Origin-Opener-Policy', 'same-origin');
        // クロスオリジンの取得物が COEP に弾かれないようにする
        headers.set('Cross-Origin-Resource-Policy', 'cross-origin');

        return new Response(response.body, {
          status: response.status,
          statusText: response.statusText,
          headers,
        });
      })
      .catch((e) => {
        // ネットワーク失敗を 502 のレスポンスに変換すると、呼び出し側には
        // 「サーバが 502 を返した」ように見えて原因を誤らせる。
        // 本来のネットワークエラーとして伝わるよう、そのまま投げ直す。
        console.error('[coi-sw] fetch に失敗しました', request.url, e);
        throw e;
      }),
  );
});
