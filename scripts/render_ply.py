"""合成した立体を絵にする（docs/12 §12.16.7）。

    python3 scripts/render_ply.py tests/multiview-probe/multi_skirt_dedup.ply out.png --yaw 0

数字だけで進めて「1枚のときより見た目が悪い」と言われたので置いた道具。
本番の描画そのものではない（正射影・簡易スプラット）が、**どの点が画面の
どこに何色で出るか**は同じなので、破綻の有無は分かる。
"""
import sys, argparse
import numpy as np

def read_ply(path):
    raw = open(path, 'rb').read()
    i = raw.index(b'end_header\n') + len(b'end_header\n')
    head = raw[:i].decode('ascii', 'replace')
    n = int([l for l in head.splitlines() if l.startswith('element vertex')][0].split()[-1])
    props = [l.split()[-1] for l in head.splitlines() if l.startswith('property float')]
    d = np.frombuffer(raw, dtype='<f4', offset=i, count=n * len(props)).reshape(n, len(props))
    return {p: d[:, k] for k, p in enumerate(props)}, n

def render(path, out, yaw_deg=0.0, size=700, cull=True):
    f, n = read_ply(path)
    p = np.stack([f['x'], f['y'], f['z']], 1).astype(np.float64)
    nrm = np.stack([f['nx'], f['ny'], f['nz']], 1).astype(np.float64)
    a = 1 / (1 + np.exp(-f['opacity'].astype(np.float64)))
    SH = 0.28209479177387814
    rgb = np.clip(0.5 + SH * np.stack([f['f_dc_0'], f['f_dc_1'], f['f_dc_2']], 1), 0, 1)
    s = np.exp(f['scale_0'].astype(np.float64))

    c = p.mean(0)
    p = p - c
    t = np.radians(yaw_deg)
    R = np.array([[np.cos(t), 0, np.sin(t)], [0, 1, 0], [-np.sin(t), 0, np.cos(t)]])
    p = p @ R.T
    nr = nrm @ R.T

    # カメラは -z 側から見る（本体と同じ、y は下が正）
    if cull:
        keep = nr[:, 2] < 0.0          # こちらを向いている面だけ
        p, nr, a, rgb, s = p[keep], nr[keep], a[keep], rgb[keep], s[keep]

    span = max(np.ptp(p[:, 0]), np.ptp(p[:, 1])) * 1.1
    k = size / span
    u = (p[:, 0] * k + size / 2).astype(np.int32)
    v = (p[:, 1] * k + size / 2).astype(np.int32)
    r = np.maximum(1, (s * k * 1.5)).astype(np.int32)
    r = np.minimum(r, 6)

    order = np.argsort(-p[:, 2])       # 奥から手前へ（画家のアルゴリズム）
    img = np.ones((size, size, 3))
    for i in order:
        x, y, rad = u[i], v[i], r[i]
        if x < -rad or y < -rad or x >= size + rad or y >= size + rad:
            continue
        x0, x1 = max(0, x - rad), min(size, x + rad + 1)
        y0, y1 = max(0, y - rad), min(size, y + rad + 1)
        if x0 >= x1 or y0 >= y1:
            continue
        al = a[i]
        img[y0:y1, x0:x1] = img[y0:y1, x0:x1] * (1 - al) + rgb[i] * al

    from PIL import Image
    Image.fromarray((img * 255).astype(np.uint8)).save(out)
    print(f"{path} yaw={yaw_deg:+.0f}° → {out}  ({len(p)} 点を描画)")

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('ply'); ap.add_argument('out')
    ap.add_argument('--yaw', type=float, default=0.0)
    ap.add_argument('--size', type=int, default=700)
    ap.add_argument('--no-cull', action='store_true')
    g = ap.parse_args()
    render(g.ply, g.out, g.yaw, g.size, not g.no_cull)
