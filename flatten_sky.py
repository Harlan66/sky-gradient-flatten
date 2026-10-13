"""sky-gradient-flatten: even out an uneven night-sky background in a sky + foreground photo.

Only the sky is changed. Foreground pixels (mask alpha = 1) are written back bit-for-bit.
The input file is never overwritten.

  python flatten_sky.py IN.tif OUT.tif --mask FOREGROUND_MASK.tif [--ref REF.tif]

Model (additive light pollution, in linear light):
  1. sRGB TIFF -> linear light. Sky weight = 1 - foreground alpha. If IN is a crop of the
     mask's canvas, the crop offset is found automatically by matching IN against --ref.
  2. Large-scale background B: stars removed by a median at 1/4 scale, only compact bright
     outliers rejected, then a locally-linear masked normalized convolution (--sigma).
  3. Target field Tf = T + keep_glow * smooth horizon glow (--glow-sigma).
  4. I' = I - (B - Tf), then a small-scale residual pass (--fine-sigma) and an
     edge-preserving median pass (--seam) for stacking seams / steps.
  5. out = w*I' + (1-w)*I, back to sRGB, ICC profile kept.
"""
import argparse, json, hashlib, os, sys, time
import numpy as np, tifffile, cv2

def srgb2lin(a): return np.where(a <= 0.04045, a / 12.92, ((a + 0.055) / 1.055) ** 2.4).astype(np.float32)
def lin2srgb(a):
    a = np.clip(a, 0, 1); return np.where(a <= 0.0031308, a * 12.92, 1.055 * a ** (1 / 2.4) - 0.055).astype(np.float32)
def sha(f):
    h = hashlib.sha256()
    with open(f, 'rb') as fh:
        for c in iter(lambda: fh.read(1 << 24), b''): h.update(c)
    return h.hexdigest()

def edges(x, f):
    g = cv2.resize(x.astype(np.float32).mean(2), None, fx=1 / f, fy=1 / f, interpolation=cv2.INTER_AREA)
    return cv2.Laplacian(cv2.GaussianBlur(np.log1p(g), (0, 0), 1), cv2.CV_32F)

def find_offset(img, ref):
    """img is a translated crop of ref (e.g. a Lightroom crop). Match the lower half by structure, then refine at full resolution."""
    h = img.shape[0] // 2
    A, B = edges(img, 4)[h // 4:], edges(ref, 4)
    r = cv2.matchTemplate(B, A, cv2.TM_CCOEFF_NORMED); _, s0, _, (x0, y0) = cv2.minMaxLoc(r)
    dy0, dx0 = y0 * 4 - (h // 4) * 4, x0 * 4
    a = img[h:h + 1024, 1024:-1024].astype(np.float32).mean(2)
    best = (-2, dy0, dx0)
    for dy in range(dy0 - 6, dy0 + 7):
        for dx in range(dx0 - 6, dx0 + 7):
            if dy < 0 or dx < 0: continue
            b = ref[h + dy:h + dy + 1024, 1024 + dx:1024 + dx + a.shape[1]].astype(np.float32).mean(2)
            if b.shape != a.shape: continue
            c = np.corrcoef(cv2.Laplacian(np.log1p(a), cv2.CV_32F).ravel()[::7], cv2.Laplacian(np.log1p(b), cv2.CV_32F).ravel()[::7])[0, 1]
            if c > best[0]: best = (c, dy, dx)
    return best, s0

def star_free(lin, down):
    q = cv2.resize(lin, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    q = cv2.medianBlur(cv2.resize(q, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA), 5)  # median at 1/4 scale removes stars
    if down > 4: q = cv2.resize(q, None, fx=4 / down, fy=4 / down, interpolation=cv2.INTER_AREA)
    return q

def compact_ok(q, v, sig):
    """Reject only compact bright outliers (bright stars, planes). Wide smooth bands/steps stay in the samples so the model fits them."""
    L = q.mean(2); vf = v.astype(np.float32)
    n = cv2.GaussianBlur(L * vf, (0, 0), sig); d = cv2.GaussianBlur(vf, (0, 0), sig)
    r = L - n / np.maximum(d, 1e-6)
    mad = 1.4826 * np.median(np.abs(r[v] - np.median(r[v])))
    return v & (r < 4 * mad), float(mad)

def masked_blur(I, v, s):
    num = cv2.GaussianBlur(I * v[..., None], (0, 0), s, borderType=cv2.BORDER_REFLECT)
    den = cv2.GaussianBlur(v.astype(np.float32), (0, 0), s, borderType=cv2.BORDER_REFLECT)
    return num, den

def local_linear(I, v, s):
    """First-order (locally linear) masked normalized convolution. Unlike zero-order, it is not pulled down by darker inner sky at the mask edge."""
    h, w_ = v.shape
    yy, xx = np.mgrid[0:h, 0:w_].astype(np.float32); yy /= s; xx /= s
    G = lambda a: cv2.GaussianBlur(a, (0, 0), s, borderType=cv2.BORDER_REFLECT)
    X, Y = xx, yy
    m = {}
    for key, f in {'1': v, 'x': v * X, 'y': v * Y, 'xx': v * X * X, 'xy': v * X * Y, 'yy': v * Y * Y}.items(): m[key] = G(f)
    Sx = m['x'] - X * m['1']; Sy = m['y'] - Y * m['1']
    Sxx = m['xx'] - 2 * X * m['x'] + X * X * m['1']; Syy = m['yy'] - 2 * Y * m['y'] + Y * Y * m['1']
    Sxy = m['xy'] - X * m['y'] - Y * m['x'] + X * Y * m['1']
    lam = 0.02 * m['1'] + 1e-6  # ridge: slopes shrink to 0 where samples are sparse
    M = np.stack([np.stack([m['1'], Sx, Sy], -1), np.stack([Sx, Sxx + lam, Sxy], -1), np.stack([Sy, Sxy, Syy + lam], -1)], -2)
    M[..., 0, 0] += 1e-6
    out = np.empty_like(I)
    for c in range(I.shape[2]):
        vi = v * I[..., c]; a0 = G(vi); ax = G(vi * X) - X * a0; ay = G(vi * Y) - Y * a0
        out[..., c] = np.linalg.solve(M, np.stack([a0, ax, ay], -1)[..., None])[..., 0, 0]
    return out, m['1']

def main():
    ap = argparse.ArgumentParser(description='Even out uneven night-sky background; foreground is left untouched.')
    ap.add_argument('inp'); ap.add_argument('out')
    ap.add_argument('--mask', help='foreground alpha mask (white = foreground/buildings, black = sky); same canvas as --ref')
    ap.add_argument('--ref', help='image on the mask canvas; used to find the crop offset of the input when sizes differ')
    ap.add_argument('--offset', type=int, nargs=2, metavar=('DY', 'DX'), help='crop offset of the input inside the mask canvas, set manually')
    ap.add_argument('--sigma', type=float, default=250, help='background model scale in px; smaller fixes smaller blotches but may remove real sky structure (e.g. Milky Way)')
    ap.add_argument('--strength', type=float, default=1.0, help='0 = no change, 1 = full correction')
    ap.add_argument('--fine-sigma', type=float, default=80, help='scale of the small-scale residual pass in px, 0 = off')
    ap.add_argument('--fine', type=float, default=1.0, help='strength of the small-scale residual pass')
    ap.add_argument('--seam', type=float, default=100, help='edge-preserving median window in px for stacking seams/steps, 0 = off')
    ap.add_argument('--keep-glow', type=float, default=0.5, help='how much smooth horizon glow to keep, 0 = completely flat')
    ap.add_argument('--glow-sigma', type=float, default=400, help='smoothing scale of the kept glow in px')
    ap.add_argument('--target-pct', type=float, default=50, help='target sky level = this percentile of the background')
    ap.add_argument('--down', type=int, default=8, help='downsample factor for the background model')
    ap.add_argument('--invert-mask', action='store_true', help='mask is white = sky instead of white = foreground')
    ap.add_argument('--force', action='store_true', help='overwrite OUT if it exists')
    a = ap.parse_args()
    if os.path.abspath(a.inp) == os.path.abspath(a.out): sys.exit('refusing to overwrite the input file')
    if os.path.exists(a.out) and not a.force: sys.exit(f'{a.out} exists; use --force to overwrite')
    t0 = time.time(); h_in = sha(a.inp)
    with tifffile.TiffFile(a.inp) as tf:
        p = tf.pages[0]; src = p.asarray(); tags = p.tags
        icc = tags[34675].value if 34675 in tags else None
        res = (tags[282].value, tags[283].value) if 282 in tags else None
    if src.ndim != 3 or src.shape[2] < 3 or src.dtype not in (np.uint8, np.uint16): sys.exit(f'need an 8/16-bit RGB TIFF, got {src.dtype} {src.shape}')
    src = src[..., :3]; MAXV = 65535 if src.dtype == np.uint16 else 255
    H, W = src.shape[:2]
    log = dict(input=a.inp, input_sha256=h_in, shape=[H, W], sigma=a.sigma, fine_sigma=a.fine_sigma, fine=a.fine, keep_glow=a.keep_glow, seam=a.seam, glow_sigma=a.glow_sigma, strength=a.strength, target_pct=a.target_pct)

    # ---- sky weight
    if a.mask:
        m = tifffile.imread(a.mask).astype(np.float32); m = m / (65535 if m.max() > 255 else 255)
        if m.ndim == 3: m = m[..., 0]
        if a.invert_mask: m = 1 - m
        if a.offset: dy, dx = a.offset
        elif m.shape == (H, W): dy = dx = 0
        else:
            if not a.ref: sys.exit('mask size differs from input: pass --ref (image on the mask canvas) or --offset DY DX')
            (c, dy, dx), s0 = find_offset(src, tifffile.imread(a.ref)); log['offset_score'] = [float(s0), float(c)]
            if c < 0.8: sys.exit(f'crop offset match is unreliable (corr={c:.3f}); pass --offset DY DX')
        log['offset'] = [int(dy), int(dx)]
        ground = m[dy:dy + H, dx:dx + W]; assert ground.shape == (H, W), ground.shape
    else:
        sys.exit('--mask is required so the foreground is never modified')
    w = 1 - ground

    # ---- background model
    lin = srgb2lin(src.astype(np.float32) / MAXV)
    q = star_free(lin, a.down)
    wq = cv2.resize(w, (q.shape[1], q.shape[0]), interpolation=cv2.INTER_AREA)
    v = wq > 0.999
    v = cv2.erode(v.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)  # keep away from lit foreground edges
    s = a.sigma / a.down
    v, mad = compact_ok(q, v, 2); log['clip_mad'] = mad
    B, den = local_linear(q, v.astype(np.float32), s)
    # far from sky samples (deep under the foreground) fall back to zero-order so slopes cannot run away
    num2, den2 = masked_blur(q, v.astype(np.float32), s * 3)
    B2 = num2 / np.maximum(den2, 1e-6)[..., None]
    t = np.clip((den - 0.05) / 0.25, 0, 1)[..., None]
    B = t * B + (1 - t) * B2
    Bf = cv2.resize(B, (W, H), interpolation=cv2.INTER_CUBIC)
    Bf = cv2.GaussianBlur(Bf, (0, 0), a.down)

    # ---- target colour
    Ls = Bf[::4, ::4].mean(2)[w[::4, ::4] > 0.999]
    lo, hi = np.percentile(Ls, [a.target_pct - 5, a.target_pct + 5])
    sel = (Bf[::4, ::4].mean(2) >= lo) & (Bf[::4, ::4].mean(2) <= hi) & (w[::4, ::4] > 0.999)
    T = Bf[::4, ::4][sel].mean(0)
    log['target_linear_rgb'] = T.tolist()
    log['bg_linear_rgb_p5_p50_p95'] = [np.percentile(Bf[::4, ::4][w[::4, ::4] > .999][:, c], [5, 50, 95]).tolist() for c in range(3)]

    # target field: constant T plus part of the smooth horizon glow; blotches and bands are not kept
    Tf = np.broadcast_to(T, Bf.shape).astype(np.float32)
    if a.keep_glow > 0:
        Bq, dq = masked_blur(cv2.resize(Bf, (W // 16, H // 16), interpolation=cv2.INTER_AREA),
                             (cv2.resize(w, (W // 16, H // 16), interpolation=cv2.INTER_AREA) > .5).astype(np.float32), a.glow_sigma / 16)
        Bq = Bq / np.maximum(dq, 1e-6)[..., None]
        Bg = cv2.GaussianBlur(cv2.resize(Bq, (W, H), interpolation=cv2.INTER_CUBIC), (0, 0), 8)
        Tf = T + a.keep_glow * np.maximum(Bg - T, 0)
    cor = lin - a.strength * (Bf - Tf)
    # pass 2: small-scale zero-order residual (overshoot near the skyline, mid-scale blotches)
    if a.fine_sigma > 0:
        q2 = star_free(cor, 4)
        w2 = cv2.resize(w, (q2.shape[1], q2.shape[0]), interpolation=cv2.INTER_AREA)
        v2 = cv2.erode((w2 > 0.999).astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        Tq = cv2.resize(Tf, (q2.shape[1], q2.shape[0]), interpolation=cv2.INTER_AREA)
        v2, _ = compact_ok(q2, v2, 3)
        n2, d2 = masked_blur(q2 - Tq, v2.astype(np.float32), a.fine_sigma / 4)
        R = n2 / np.maximum(d2, 1e-6)[..., None] * np.clip(d2 / 0.1, 0, 1)[..., None]
        R = cv2.GaussianBlur(cv2.resize(R, (W, H), interpolation=cv2.INTER_CUBIC), (0, 0), 4)
        cor = cor - a.strength * a.fine * R
        log['fine_residual_linear_p1_p99'] = np.percentile(R[::4, ::4][w[::4, ::4] > .999].mean(1), [1, 99]).tolist()
    # pass 3: edge-preserving median residual (sharp stacking seams that smooth models cannot remove)
    if a.seam > 0:
        from scipy.ndimage import median_filter, distance_transform_edt
        q3 = star_free(cor, 4)
        Tq = cv2.resize(Tf, (q3.shape[1], q3.shape[0]), interpolation=cv2.INTER_AREA)
        w3 = cv2.resize(w, (q3.shape[1], q3.shape[0]), interpolation=cv2.INTER_AREA) > 0.999
        r3 = q3 - Tq
        ok, _ = compact_ok(q3, w3, 3)  # keep bright star residue out
        idx = distance_transform_edt(~ok, return_distances=False, return_indices=True)
        r3 = r3[idx[0], idx[1]]  # fill foreground/stars with nearest sky so the median never sees buildings
        k = max(3, int(a.seam / 4) | 1)
        R3 = np.stack([median_filter(r3[..., c], size=k, mode='reflect') for c in range(3)], -1)
        R3 = cv2.GaussianBlur(R3, (0, 0), 1.0)
        R3 = cv2.GaussianBlur(cv2.resize(R3, (W, H), interpolation=cv2.INTER_LINEAR), (0, 0), 2)
        cor = cor - a.strength * R3
        log['seam_residual_linear_p1_p99'] = np.percentile(R3[::4, ::4][w[::4, ::4] > .999].mean(1), [1, 99]).tolist()
    log['neg_clipped_frac_sky'] = float(((cor < 0).any(2) & (w > .999)).sum() / max((w > .999).sum(), 1))
    out = w[..., None] * cor + (1 - w[..., None]) * lin
    o16 = np.round(lin2srgb(out) * MAXV).astype(src.dtype)
    o16[ground >= 1] = src[ground >= 1]  # foreground pixels bit-for-bit unchanged
    kw = dict(photometric='rgb', planarconfig='contig', software='flatten_sky.py', metadata=None)
    if res: kw['resolution'] = (res[0], res[1])
    if icc is not None: kw['iccprofile'] = icc
    tifffile.imwrite(a.out, o16, **kw)

    # ---- QA
    base = os.path.splitext(a.out)[0]
    chk = tifffile.imread(a.out); log['readback_equal'] = bool(np.array_equal(chk, o16))
    log['ground_unchanged'] = bool(np.array_equal(chk[ground >= 1], src[ground >= 1]))
    log['input_unchanged'] = sha(a.inp) == h_in
    prev = lambda x: cv2.cvtColor(np.round(cv2.resize(x.astype(np.float32), (W // 4, H // 4), interpolation=cv2.INTER_AREA) * (255 / MAXV)).astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imwrite(base + '_compare.jpg', np.vstack([prev(src), prev(o16)]), [cv2.IMWRITE_JPEG_QUALITY, 88])
    bgv = lin2srgb(Bf / max(Bf[w > .999].max(), 1e-6))
    cv2.imwrite(base + '_bgmodel.jpg', cv2.cvtColor((cv2.resize(bgv, (W // 4, H // 4)) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
    # uniformity: spread of per-block sky background level, before/after
    def blocks(img):
        L = srgb2lin(img.astype(np.float32) / MAXV).mean(2); vals = []
        for y in range(0, H - 200, 200):
            for x in range(0, W - 200, 200):
                if w[y:y + 200, x:x + 200].min() > .999: vals.append(np.median(L[y:y + 200, x:x + 200]))
        v_ = np.array(vals)
        if len(v_) < 4: return dict(n=len(v_))
        return dict(n=len(v_), cv=float(v_.std() / v_.mean()), p95_p5=float(np.percentile(v_, 95) / np.percentile(v_, 5)))
    log['sky_uniformity_before'] = blocks(src); log['sky_uniformity_after'] = blocks(o16)
    log['output'] = a.out; log['output_sha256'] = sha(a.out); log['seconds'] = round(time.time() - t0, 1)
    json.dump(log, open(base + '_log.json', 'w'), ensure_ascii=False, indent=1)
    print(json.dumps({k: log[k] for k in ['offset', 'sky_uniformity_before', 'sky_uniformity_after', 'neg_clipped_frac_sky', 'ground_unchanged', 'readback_equal', 'input_unchanged', 'seconds'] if k in log}, ensure_ascii=False))

if __name__ == '__main__': main()
