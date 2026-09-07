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
 * CORP か CORS が必要になる。
 *
 *   ・モデルは同一オリジン配信（決定 D7）なので影響しない
 *   ・PoC が HuggingFace から直接取得する経路は `fetch()` が CORS モードなので、
 *     このワーカーが中身を読めて CORP を付けられる（実測確認済み）
 *   ・**Google Fonts の `<link rel="stylesheet">` は no-cors** で飛ぶため、そのままだと
 *     不透明レスポンスになりヘッダを付けられず、COEP に弾かれてフォントが落ちる。
 *     Google Fonts は CORS を返すので、no-cors のクロスオリジン要求は
 *     **CORS モードで取り直して**中身を読めるようにする（下の refetchAsCors）。
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

/**
 * クロスオリジンの no-cors 要求を CORS モードで取り直す。
 *
 * no-cors のままだと不透明レスポンスになり、ヘッダを足せず COEP に弾かれる。
 * 相手が CORS を返すなら（Google Fonts はそう）、CORS で取り直せば中身が読めて
 * CORP を付けられる。取り直しに失敗したら元の要求に戻す。
 */
function refetchAsCors(request) {
  return fetch(
    new Request(request.url, {
      method: request.method,
      headers: request.headers,
      mode: 'cors',
      credentials: 'omit',
      referrer: request.referrer,
      redirect: 'follow',
    }),
  ).catch(() => fetch(request));
}

function withCoiHeaders(response) {
  // 不透明レスポンスはヘッダを触れない。そのまま返す（COEP に弾かれるが、
  // ここで壊すよりは呼び出し側に本来の挙動を見せるほうがよい）。
  if (response.status === 0 || response.type === 'opaque') return response;

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
}

self.addEventListener('fetch', (event) => {
  const request = event.request;

  // Range リクエストや only-if-cached は触らない（触ると壊れる）
  if (request.cache === 'only-if-cached' && request.mode !== 'same-origin') return;

  const sameOrigin = new URL(request.url).origin === self.location.origin;
  const needsCorsRefetch = !sameOrigin && request.mode === 'no-cors';

  event.respondWith(
    (needsCorsRefetch ? refetchAsCors(request) : fetch(request))
      .then(withCoiHeaders)
      .catch((e) => {
        // ネットワーク失敗を 502 のレスポンスに変換すると、呼び出し側には
        // 「サーバが 502 を返した」ように見えて原因を誤らせる。
        // 本来のネットワークエラーとして伝わるよう、そのまま投げ直す。
        console.error('[coi-sw] fetch に失敗しました', request.url, e);
        throw e;
      }),
  );
});
