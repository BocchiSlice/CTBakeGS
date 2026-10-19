#!/usr/bin/env python3
"""
Where do the splat models differ from the reference?  Error heatmaps and error by image region.

  pip install numpy pillow
  python error_maps.py OUTPUT_FOLDER [--variant 2M] [--views wide_000,close_000] [--vmax 0.25]

OUTPUT_FOLDER is the study output (case/look/... folders or case__look.zip files).
For every scene and every view, the reference images/ref/<view>.png is compared with images/gs_<variant>/<view>.png.

Writes, in OUTPUT_FOLDER/error_maps/:
  heat_<case>__<look>__<view>.png   absolute error (mean over RGB, 0..1) with a fixed colour scale (0..vmax),
                                    for the views given by --views (figure panels)
  colorbar.png                      the colour scale
  error_by_region.csv               per case, look, view kind and region: share of pixels, share of the
                                    total squared error, mean absolute error
  error_by_region_summary.csv       the same, median over cases

Regions (from the reference image):
  silhouette  foreground pixels within --band px of the foreground/background border (both sides)
  edges       other foreground pixels whose luminance gradient is in the top 10% of the view
  interior    remaining foreground pixels
  background  remaining pixels
"""
import argparse, csv, io, os, re, zipfile
from collections import defaultdict

import numpy as np
from PIL import Image

# magma-like colour ramp (dependency-free), 0..1 -> RGB
_STOPS = np.array([
    [0.00, 0.001, 0.000, 0.014], [0.13, 0.078, 0.042, 0.206], [0.25, 0.232, 0.059, 0.437],
    [0.38, 0.390, 0.100, 0.502], [0.50, 0.550, 0.161, 0.506], [0.63, 0.716, 0.215, 0.475],
    [0.75, 0.868, 0.288, 0.409], [0.88, 0.967, 0.439, 0.360], [1.00, 0.987, 0.991, 0.750]])


def cmap(x):
    x = np.clip(x, 0, 1)
    return np.stack([np.interp(x, _STOPS[:, 0], _STOPS[:, c]) for c in (1, 2, 3)], -1)


class Scene:
    def __init__(self, path):
        self.zip = zipfile.ZipFile(path) if path.endswith('.zip') else None
        self.path = path
        self.names = set(self.zip.namelist() if self.zip else [
            os.path.relpath(os.path.join(d, f), path).replace(os.sep, '/') for d, _, fs in os.walk(path) for f in fs])

    def read(self, n):
        if self.zip:
            return self.zip.read(n)
        with open(os.path.join(self.path, n), 'rb') as f:
            return f.read()

    def img(self, n):
        return np.asarray(Image.open(io.BytesIO(self.read(n))).convert('RGB'), dtype=np.float32) / 255


def find_scenes(root):
    out = []
    for d, _, fs in os.walk(root):
        rel = os.path.relpath(d, root)
        if 'error_maps' in d or ('figures_' in rel and 'figures_' not in os.path.basename(os.path.abspath(root))):
            continue   # high-resolution figure images are analysed separately: python error_maps.py OUT/figures_2400
        if 'metrics_per_view.csv' in fs:
            out.append(d)
        out += [os.path.join(d, f) for f in fs if f.endswith('.zip') and '__' in f]
    return sorted(out)


def dilate(m, r):
    o = m.copy()
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            o |= np.roll(np.roll(m, dy, 0), dx, 1)
    return o


def regions(ref, band):
    fg = ref.max(-1) > 2 / 255
    border = dilate(fg, 1) & dilate(~fg, 1)
    sil = dilate(border, band)
    lum = ref @ np.array([0.299, 0.587, 0.114], np.float32)
    gy, gx = np.gradient(lum)
    g = np.hypot(gx, gy)
    inner = fg & ~sil
    thr = np.quantile(g[inner], 0.9) if inner.any() else np.inf
    edges = inner & (g >= thr)
    interior = inner & ~edges
    bg = ~(sil | edges | interior)
    return {'silhouette': sil, 'edges': edges, 'interior': interior, 'background': bg}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('root')
    ap.add_argument('--variant', default='2M')
    ap.add_argument('--views', default='wide_000,close_000')
    ap.add_argument('--vmax', type=float, default=0.25)
    ap.add_argument('--band', type=int, default=2)
    a = ap.parse_args()
    od = os.path.join(a.root, 'error_maps')
    os.makedirs(od, exist_ok=True)
    figviews = set(v for v in a.views.split(',') if v)
    pat = re.compile(r'^images/gs_' + re.escape(a.variant) + r'/((wide|close)_\d+)\.png$')
    acc = defaultdict(lambda: defaultdict(float))  # (case, look, kind) -> sums
    for sp in find_scenes(a.root):
        s = Scene(sp)
        if 'metrics_per_view.csv' in s.names:
            r0 = next(csv.DictReader(io.StringIO(s.read('metrics_per_view.csv').decode())))
            case, look = r0['case'], r0['look']
        else:
            case, look = os.path.basename(sp).replace('.zip', '').split('__')[:2]
        n = 0
        for name in sorted(s.names):
            m = pat.match(name)
            if not m or f'images/ref/{m.group(1)}.png' not in s.names:
                continue
            view, kind = m.group(1), m.group(2)
            ref, gs = s.img(f'images/ref/{view}.png'), s.img(name)
            ae = np.abs(ref - gs).mean(-1)
            se = ((ref - gs) ** 2).mean(-1)
            k = (case, look, kind)
            acc[k]['n'] += 1
            tot = se.sum() or 1.0
            for rn, mask in regions(ref, a.band).items():
                acc[k][rn + '_px'] += mask.mean()
                acc[k][rn + '_se'] += se[mask].sum() / tot
                acc[k][rn + '_ae'] += ae[mask].mean() if mask.any() else 0.0
            if view in figviews:
                rgb = (cmap(ae / a.vmax) * 255 + 0.5).astype(np.uint8)
                Image.fromarray(rgb).save(os.path.join(od, f'heat_{case}__{look}__{view}.png'))
            n += 1
        print(f'{case} {look}: {n} views', flush=True)
    # colour bar
    bar = (cmap(np.linspace(0, 1, 512))[None].repeat(24, 0) * 255).astype(np.uint8)
    Image.fromarray(bar).save(os.path.join(od, 'colorbar.png'))
    R = ['silhouette', 'edges', 'interior', 'background']
    rows = []
    for (case, look, kind), d in sorted(acc.items()):
        for rn in R:
            rows.append([case, look, kind, rn, d[rn + '_px'] / d['n'], d[rn + '_se'] / d['n'], d[rn + '_ae'] / d['n']])
    hdr = ['case', 'look', 'view_kind', 'region', 'pixel_share', 'sq_error_share', 'mean_abs_error']
    with open(os.path.join(od, 'error_by_region.csv'), 'w', newline='') as f:
        w = csv.writer(f); w.writerow(hdr)
        w.writerows([r[:4] + [f'{x:.5f}' for x in r[4:]] for r in rows])
    grp = defaultdict(list)
    for r in rows:
        grp[(r[1], r[2], r[3])].append(r[4:])
    with open(os.path.join(od, 'error_by_region_summary.csv'), 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['look', 'view_kind', 'region', 'n_cases'] + hdr[4:])
        for k in sorted(grp):
            v = np.array(grp[k])
            w.writerow(list(k) + [len(v)] + [f'{x:.5f}' for x in np.median(v, 0)])
    print('wrote', od)


if __name__ == '__main__':
    main()
