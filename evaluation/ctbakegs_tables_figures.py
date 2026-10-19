#!/usr/bin/env python3
"""
Tables and Figure 2 of the CTBakeGS paper, from the output folder of the evaluation page.

  python ctbakegs_tables_figures.py OUT_FOLDER [--desktop PERF_FOLDER] [--laptop PERF_FOLDER] [--results RESULTS_FOLDER]

OUT_FOLDER   folder written by "Run study" (for example out6), with summary.csv (LPIPS is computed on the first run with study_lpips.py)
--ct         optional: ct.nii.gz of case s0138, for the CT slice in Figure 1 (needs nibabel)
--desktop    optional: "perf" folder written by "Measure speed only" on a desktop
--laptop     optional: "perf" folder written by "Measure speed only" on an older laptop
Writes CSV tables, figure2.png/.pdf and figureS1_ablation.png into RESULTS_FOLDER (default: OUT_FOLDER/paper_results).
Needs numpy, pandas, scipy, matplotlib and pillow.
"""
import argparse, os
ap = argparse.ArgumentParser()
ap.add_argument('out'); ap.add_argument('--laptop', default=''); ap.add_argument('--desktop', default=''); ap.add_argument('--results', default=''); ap.add_argument('--ct', default='', help='ct.nii.gz of case s0138 for the CT slice in Figure 1 (needs nibabel)')
args = ap.parse_args()
OUT = os.path.abspath(args.out); PERF_LAPTOP = args.laptop; PERF_DESKTOP = args.desktop; CT_FILE = args.ct; RESULTS = args.results or os.path.join(OUT, 'paper_results')
CASES = [('s0289', 1), ('s0138', 2), ('s1340', 3), ('s0970', 4), ('s1121', 5), ('s0931', 6)]   # identifier, case number
THICK = ('s0289', 's0289_5mm')
LOOKS = [('dvrBone', 'Bone'), ('dvrCta', 'Vessels'), ('dvrSoft', 'Skin and bone')]
os.makedirs(RESULTS, exist_ok=True)
import matplotlib
matplotlib.use('Agg')
import csv, json, glob
import numpy as np, pandas as pd
from scipy.stats import wilcoxon

if not os.path.exists(os.path.join(OUT, 'summary_all.csv')):   # first run: LPIPS for every image (study_lpips.py, next to this script)
    import subprocess, sys
    subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'study_lpips.py'), OUT], check=True)
SA = pd.read_csv(os.path.join(OUT, 'summary_all.csv'))      # per case / look / variant / view kind (PSNR, SSIM, LPIPS)
SU = pd.read_csv(os.path.join(OUT, 'summary.csv'))          # per case / look / variant (splats, conversion time, fps)
C = [c for c, _ in CASES]

def m(case, look, var, kind, col):
    r = SA[(SA.case == case) & (SA.look == look) & (SA.variant == var) & (SA.view_kind == kind)]
    return float(r[col].iloc[0]) if len(r) else np.nan

def msd(x, d=2):
    x = np.asarray(x, float); return f'{x.mean():.{d}f} ± {x.std(ddof=1):.{d}f}'

def p_exact(d):
    d = np.asarray(d, float)
    return 1.0 if np.allclose(d, 0) else wilcoxon(d, alternative='two-sided', mode='exact').pvalue
print(len(SA), 'rows;', sorted(SA.variant.unique()))

rows = []
for look, ln in LOOKS:
    for kind, kn in [('wide', 'wide'), ('close', 'close-up')]:
        rows.append({'Look, views': f'{ln}, {kn}',
                     'PSNR 1 M (dB)': msd([m(c, look, '1M', kind, 'psnr_mean') for c in C]),
                     'PSNR 2 M (dB)': msd([m(c, look, '2M', kind, 'psnr_mean') for c in C]),
                     'PSNR 3 M (dB)': msd([m(c, look, '3M', kind, 'psnr_mean') for c in C]),
                     'SSIM 2 M': msd([m(c, look, '2M', kind, 'ssim_mean') for c in C], 3),
                     'LPIPS 2 M': msd([m(c, look, '2M', kind, 'lpips_vgg_mean') for c in C], 3),
                     'PSNR real-time photorealistic (dB)': msd([m(c, look, 'realtimePhotoreal', kind, 'psnr_mean') for c in C])})
T1 = pd.DataFrame(rows); T1.to_csv(os.path.join(RESULTS, 'table1_quality.csv'), index=False); print('\n' + T1.to_string(index=False))

def mr(x, fmt='{:.1f}'):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    return f'{fmt.format(x.mean())} ({fmt.format(x.min())}–{fmt.format(x.max())})' + ('' if len(x) == len(C) * len(LOOKS) else f' [n={len(x)}]')
def perf(S, col, var, scale=1.0, fmt='{:.1f}'):
    v = S[(S.variant == var) & (S.case.isin(C))][col]
    return mr(pd.to_numeric(v, errors='coerce') * scale, fmt)
rows = [{'': 'Splats (million)', **{v: perf(SU, 'splats', v, 1e-6, '{:.2f}') for v in ['1M', '2M', '3M']}},
        {'': 'Conversion time, main computer (s)', **{v: perf(SU, 'bake_s', v) for v in ['1M', '2M', '3M']}},
        {'': 'Frame rate, main computer (fps)', **{v: perf(SU, 'fps_800', v, fmt='{:.0f}') for v in ['1M', '2M', '3M']}}]
if PERF_DESKTOP and os.path.exists(os.path.join(PERF_DESKTOP, 'summary.csv')):
    SD = pd.read_csv(os.path.join(PERF_DESKTOP, 'summary.csv')).drop_duplicates(['case', 'look', 'variant'], keep='last')
    rows.append({'': 'Conversion time, desktop (s)', **{v: perf(SD, 'bake_s', v) for v in ['1M', '2M', '3M']}})
    rows.append({'': 'Frame rate, desktop (fps)', **{v: perf(SD, 'fps_800', v, fmt='{:.0f}') for v in ['1M', '2M', '3M']}})
if PERF_LAPTOP and os.path.exists(os.path.join(PERF_LAPTOP, 'summary.csv')):
    SL = pd.read_csv(os.path.join(PERF_LAPTOP, 'summary.csv')).drop_duplicates(['case', 'look', 'variant'], keep='last')
    rows.append({'': 'Conversion time, older laptop (s)', **{v: perf(SL, 'bake_s', v, fmt='{:.0f}') for v in ['1M', '2M', '3M']}})
T2 = pd.DataFrame(rows); T2.to_csv(os.path.join(RESULTS, 'table2_performance.csv'), index=False); print('\n' + T2.to_string(index=False))

rows = []
for c, n in CASES:
    sj = json.load(open(os.path.join(OUT, c, 'dvrBone', 'scene.json')))
    d = sj['ct']['dims']
    t = SU[(SU.case == c) & (SU.variant == '2M')].bake_s.astype(float).mean()
    rows.append({'Case': n, 'Identifier': c, 'Matrix (voxels)': ' × '.join(map(str, d)), 'Voxels (million)': round(d[0] * d[1] * d[2] / 1e6, 1),
                 'Spacing (mm)': ' × '.join(f'{s:g}' for s in sj['ct']['spacingMm']), 'Conversion, one pass, 2 M (s)': round(t, 1)})
TS1 = pd.DataFrame(rows); TS1.to_csv(os.path.join(RESULTS, 'tableS1_cases.csv'), index=False); print('\n' + TS1.to_string(index=False))

VARS = [('1M', '1 M splats', '2M'), ('3M', '3 M splats', '2M'), ('noAA', 'No anti-aliasing compensation', '2M'),
        ('naiveOpacity', 'No opacity mapping', '2M'), ('uniformCells', 'Uniform grid (1 cell/voxel)', '2M'),
        ('uniformCells', 'Uniform grid vs octree, equal count', 'adaptiveMatched')]
rows = []
for look, ln in LOOKS:
    for kind, kn in [('wide', 'wide'), ('close', 'close-up')]:
        for v, vn, base in VARS:
            r = {'Look': ln, 'Views': kn, 'Variant': vn}
            for col, lab in [('psnr_mean', 'ΔPSNR (dB)'), ('ssim_mean', 'ΔSSIM'), ('lpips_vgg_mean', 'ΔLPIPS')]:
                d = [m(c, look, v, kind, col) - m(c, look, base, kind, col) for c in C]
                r[lab] = f'{np.median(d):+.3f}'; r[lab + ' p'] = f'{p_exact(d):.3f}'
            rows.append(r)
TS2 = pd.DataFrame(rows); TS2.to_csv(os.path.join(RESULTS, 'tableS2_comparisons.csv'), index=False); print('\n' + TS2.to_string(index=False))

thin, thick = THICK
rows = []
for look, ln in LOOKS:
    for kind, kn in [('wide', 'wide'), ('close', 'close-up')]:
        f = lambda case, v: f"{m(case, look, v, kind, 'psnr_mean'):.2f} / {m(case, look, v, kind, 'ssim_mean'):.3f} / {m(case, look, v, kind, 'lpips_vgg_mean'):.3f}"
        rows.append({'Look': ln, 'Views': kn, '1.5-mm, millimetre cells': f(thin, '2M'), '5-mm, millimetre cells': f(thick, '2M'), '5-mm, voxel-unit cells': f(thick, 'voxelCells')})
TS3 = pd.DataFrame(rows); TS3.to_csv(os.path.join(RESULTS, 'tableS3_thick.csv'), index=False); print('\n' + TS3.to_string(index=False))

from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
plt.rcParams['svg.fonttype'] = 'none'; plt.rcParams['pdf.fonttype'] = 42   # text stays text in SVG/PDF
plt.rcParams['font.family'] = 'sans-serif'; plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'Liberation Sans', 'DejaVu Sans']   # Arial as in most journals

FIG2 = [  # title, case, look, wide view, close-up view, inset corner ('tl', 'tr', 'bl', 'br')
    ('a  Bone', 's0289', 'dvrBone', 'wide_017', 'close_014', 'tr'),
    ('b  Contrast-enhanced vessels', 's0970', 'dvrCta', 'wide_016', 'close_015', 'tr'),
    ('c  Skin and bone', 's1340', 'dvrSoft', 'wide_016', 'close_015', 'tr'),
]
VMAX = 0.2
INSET = 0.30   # inset side as a fraction of the panel width
ASPECT = 1.3   # panel height / width (portrait: less black beside tall bodies)
STOPS = np.array([[0.00, 0.001, 0.000, 0.014], [0.13, 0.078, 0.042, 0.206], [0.25, 0.232, 0.059, 0.437], [0.38, 0.390, 0.100, 0.502],
                  [0.50, 0.550, 0.161, 0.506], [0.63, 0.716, 0.215, 0.475], [0.75, 0.868, 0.288, 0.409], [0.88, 0.967, 0.439, 0.360], [1.00, 0.987, 0.991, 0.750]])
def cmap(x):
    x = np.clip(x, 0, 1); return np.stack([np.interp(x, STOPS[:, 0], STOPS[:, k]) for k in (1, 2, 3)], -1)
def img(case, look, method, view):
    return np.asarray(Image.open(os.path.join(OUT, case, look, 'images', method, view + '.png')).convert('RGB'), np.float32) / 255
def heat(a, b): return cmap(np.abs(b - a).mean(-1) / VMAX)
def close_box(case, look, wide, close, W=800):
    V = {v['name']: v for v in json.load(open(os.path.join(OUT, case, look, 'cameras.json')))['views']}
    F = (W / 2) / np.tan(np.radians(V[wide]['fovYDeg'] / 2))
    def proj(v, P):
        M = np.array(v['c2w']); pc = M[:3, :3].T @ (P - M[:3, 3]); z = -pc[2]
        return np.array([W / 2 + F * pc[0] / z, W / 2 - F * pc[1] / z])
    c = V[close]; M = np.array(c['c2w']); t = np.array(c['target']); hs = np.linalg.norm(t - M[:3, 3]) * np.tan(np.radians(c['fovYDeg'] / 2))
    q = np.array([proj(V[wide], t + sx * hs * M[:3, 0] + sy * hs * M[:3, 1]) for sx in (-1, 1) for sy in (-1, 1)])
    return q.min(0), q.max(0)
def crop_box(a, b, bmin, bmax):
    msk = (a.max(-1) > 6 / 255) | (b.max(-1) > 6 / 255); ys, xs = np.where(msk)
    x0, x1 = min(xs.min(), bmin[0]), max(xs.max(), bmax[0]); y0, y1 = min(ys.min(), bmin[1]), max(ys.max(), bmax[1])
    hh = min(800, max(y1 - y0, (x1 - x0) * ASPECT) * 1.04); ww = hh / ASPECT
    cx, cy = min(max((x0 + x1) / 2, ww / 2), 800 - ww / 2), min(max((y0 + y1) / 2, hh / 2), 800 - hh / 2)
    return [int(round(v)) for v in (cx - ww / 2, cy - hh / 2, cx + ww / 2, cy + hh / 2)]

# layout in inches: the same gap between panels horizontally and vertically
FIG_W, LBL, TTL, GAP, CB = 6.69, 0.25, 0.27, 0.05, 0.42   # width, row-label strip, title strip, gap, colour-bar strip
pw = (FIG_W - LBL - 2 * GAP) / 3; ph = pw * ASPECT; FIG_H = TTL + 3 * ph + 2 * GAP + CB
fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=300); ax = np.empty((3, 3), dtype=object)
for i in range(3):
    for j in range(3):
        ax[i, j] = fig.add_axes([(LBL + j * (pw + GAP)) / FIG_W, (FIG_H - TTL - (i + 1) * ph - i * GAP) / FIG_H, pw / FIG_W, ph / FIG_H])
for j, (title, case, look, wide, close, corner) in enumerate(FIG2):
    a, b = img(case, look, 'ref', wide), img(case, look, 'gs_2M', wide)
    ca, cb = img(case, look, 'ref', close), img(case, look, 'gs_2M', close)
    bmin, bmax = close_box(case, look, wide, close); X0, Y0, X1, Y1 = crop_box(a, b, bmin, bmax); S = X1 - X0
    bx0, by0, bw, bh = (bmin[0] - X0) / S, (bmin[1] - Y0) / S, (bmax[0] - bmin[0]) / S, (bmax[1] - bmin[1]) / S   # in units of the panel width
    isz = INSET   # inset size (same in every column); shrinks only if it would cover the dashed box
    while True:
        ix = 0 if corner[1] == 'l' else 1 - isz; iy = ASPECT - isz if corner[0] == 'b' else 0
        ov = max(0, min(ix + isz, bx0 + bw) - max(ix, bx0)) * max(0, min(iy + isz, by0 + bh) - max(iy, by0))
        if ov <= 1e-6 or isz <= 0.28: break
        isz -= 0.01
    for i, (big, small) in enumerate([(a, ca), (b, cb), (heat(a, b), heat(ca, cb))]):
        A = ax[i, j]; A.imshow(big[Y0:Y1, X0:X1], extent=(0, 1, ASPECT, 0)); A.set_xticks([]); A.set_yticks([])
        for s in A.spines.values(): s.set_visible(False)
        A.add_patch(Rectangle((bx0, by0), bw, bh, fill=False, ec='#FFD400', lw=0.6, ls='--'))
        A.imshow(small, extent=(ix, ix + isz, iy + isz, iy)); A.add_patch(Rectangle((ix, iy), isz, isz, fill=False, ec='#FFD400', lw=0.9))
        A.set_xlim(0, 1); A.set_ylim(ASPECT, 0)
    ax[0, j].set_title(title, fontsize=10, fontweight='bold', loc='left', pad=5)
for i, n in enumerate(['Reference', '3DGS (2 M splats)', 'Absolute difference']):
    ax[i, 0].set_ylabel(n, fontsize=10, fontweight='bold', labelpad=3)
cbx = LBL + 2 * (pw + GAP); cax = fig.add_axes([cbx / FIG_W, (CB - 0.2) / FIG_H, pw / FIG_W, 0.09 / FIG_H]); cax.imshow(cmap(np.linspace(0, 1, 256))[None], aspect='auto', extent=(0, VMAX, 0, 1))
cax.set_yticks([]); cax.set_xticks([0, VMAX]); cax.set_xticklabels(['0', f'≥{VMAX:g}'], fontsize=8); cax.tick_params(length=2, pad=1)
fig.text((cbx - 0.08) / FIG_W, (CB - 0.155) / FIG_H, 'Mean absolute error (fraction of full scale)', fontsize=8, ha='right', va='center')
for ext in ('png', 'pdf', 'svg'): fig.savefig(os.path.join(RESULTS, f'figure2.{ext}'), bbox_inches='tight', pad_inches=0.01)   # PDF and SVG keep the text editable
fig.savefig(os.path.join(RESULTS, 'figure2.tif'), dpi=600, bbox_inches='tight', pad_inches=0.02, pil_kwargs={'compression': 'tiff_lzw'})   # raster for submission
plt.close()

# ---- Figure 1: the conversion pipeline (schematic) ----
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, FancyArrow
plt.rcParams['svg.fonttype'] = 'none'; plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['font.family'] = 'sans-serif'; plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'Liberation Sans', 'DejaVu Sans']
FIG1_CASE, FIG1_LOOK, FIG1_VIEW = 's0138', 'dvrCta', 'wide_016'   # 3DGS example in the last column
FIG1_CT = CT_FILE   # NIfTI of the same case for the CT slice ('' to leave the slot grey)

def ct_slice(path, frac=0.55, wl=40, ww=400):
    try:
        import nibabel as nib
        v = np.asarray(nib.load(path).dataobj, np.float32); s = v[:, :, int(v.shape[2] * frac)]
        return np.clip((np.rot90(s) - (wl - ww / 2)) / ww, 0, 1)
    except Exception as e:
        print('CT slice not drawn:', e); return None

steps = [('CT volume +\ntransfer function', 'Hounsfield units →\ncolour and opacity'),
         ('Light volume', 'sky visibility (24\ndirections) + key-light\nsoft shadows'),
         ('Adaptive octree of\nmillimetre cubes', 'split at opacity\nboundaries, down to\nsub-voxel size'),
         ('Bake per cell', '27 stratified samples:\ncolour and lighting of\nthe volume renderer'),
         ('Opacity mapping', r'$\tau_{\mathrm{splat}} = \tau_{\mathrm{cell}}\,h^2/(2\pi s^2)$' + '\n' + r'$s = 0.55\,h$'),
         ('Standard 3DGS', '.ply / .spz\nfor common 3DGS\nviewers')]
W1, H1 = 6.69, 2.00; n = len(steps); bw = 0.98; gap = (W1 - 0.16 - n * bw) / (n - 1)
fig = plt.figure(figsize=(W1, H1), dpi=300); A = fig.add_axes([0, 0, 1, 1]); A.set_xlim(0, W1); A.set_ylim(H1, 0); A.axis('off')
yB, hB, yI, S = 0.08, 0.5, 0.7, 0.86; yC = yI + S + 0.09
X = lambda i: 0.08 + i * (bw + gap); OX = lambda i: X(i) + bw / 2 - S / 2
for i, (a, b) in enumerate(steps):
    A.add_patch(FancyBboxPatch((X(i), yB), bw, hB, boxstyle='round,pad=0,rounding_size=0.06', fc='#C0392B' if i == n - 1 else '#2F3B4C', ec='none'))
    A.text(X(i) + bw / 2, yB + hB / 2, a, ha='center', va='center', color='white', fontsize=7.5, fontweight='bold', linespacing=1.15)
    A.text(X(i) + bw / 2, yC, b, ha='center', va='top', color='#333333', fontsize=6.5, linespacing=1.25)
    if i < n - 1: A.add_patch(FancyArrow(X(i) + bw + 0.015, yB + hB / 2, gap - 0.03, 0, width=0.06, head_width=0.14, head_length=0.05, length_includes_head=True, color='#8C8C8C', lw=0))
ct = ct_slice(FIG1_CT) if FIG1_CT else None   # 0: CT slice
if ct is not None: A.imshow(ct, cmap='gray', extent=(OX(0), OX(0) + S, yI + S, yI), vmin=0, vmax=1)
else: A.add_patch(Rectangle((OX(0), yI), S, S, fc='#D9D9D9', ec='none'))
g, c = 5, (S - 0.12) / 5; x0, y0 = OX(1) + 0.12, yI + 0.12   # 1: light volume
for r in range(g):
    for k in range(g): v = (250 - 26 * (r + k)) / 255; A.add_patch(Rectangle((x0 + k * c, y0 + r * c), c, c, fc=(v, v, v), ec='white', lw=0.5))
A.add_patch(Circle((OX(1) + 0.06, yI + 0.06), 0.1, fc='#F5B041', ec='#D68910', lw=0.5))
for dx, dy in [(0.3, 0.12), (0.22, 0.26), (0.12, 0.32)]: A.annotate('', xy=(OX(1) + 0.13 + dx, yI + 0.1 + dy), xytext=(OX(1) + 0.13, yI + 0.1), arrowprops=dict(arrowstyle='-|>', color='#D68910', lw=0.75))
x0, q = OX(2), S / 2; r_, e = q / 2, q / 4   # 2: adaptive cells
A.add_patch(Rectangle((x0, yI), S, S, fc='white', ec='#555555', lw=0.5))
for dx, dy in [(0, 0), (q, 0), (0, q)]: A.add_patch(Rectangle((x0 + dx, yI + dy), q, q, fc='white', ec='#555555', lw=0.5))
for dx, dy in [(q, q), (q + r_, q), (q, q + r_), (q + r_, q + r_)]: A.add_patch(Rectangle((x0 + dx, yI + dy), r_, r_, fc='#F2D7D5', ec='#555555', lw=0.4))
for dx, dy in [(q + r_, q + r_), (q + r_ + e, q + r_), (q + r_, q + r_ + e), (q + r_ + e, q + r_ + e)]: A.add_patch(Rectangle((x0 + dx, yI + dy), e, e, fc='#E6B0AA', ec='#555555', lw=0.3))
x0, y0, cc = OX(3) + 0.08, yI + 0.08, S - 0.16; d = cc / 3   # 3: one cell, stratified samples
A.add_patch(Rectangle((x0, y0), cc, cc, fc='#FDF2E9', ec='#555555', lw=0.6))
for k in (1, 2): A.plot([x0 + k * d] * 2, [y0, y0 + cc], ls='--', c='#BBBBBB', lw=0.4); A.plot([x0, x0 + cc], [y0 + k * d] * 2, ls='--', c='#BBBBBB', lw=0.4)
jit = [[0.3, 0.6], [0.7, 0.35], [0.45, 0.25], [0.6, 0.7], [0.25, 0.4], [0.55, 0.5], [0.4, 0.75], [0.75, 0.6], [0.35, 0.3]]
for r in range(3):
    for k in range(3): jx, jy = jit[r * 3 + k]; A.add_patch(Circle((x0 + (k + jx) * d, y0 + (r + jy) * d), 0.03, fc='#C0392B', ec='none'))
RED = (0.753, 0.224, 0.169); hh, TC = 0.2, 0.85; cyy = yI + 0.14; xa, xb = OX(4) + 0.17, OX(4) + S - 0.2   # 4a: one cell -> one soft Gaussian with the same total opacity
A.add_patch(Rectangle((xa - hh / 2, cyy - hh / 2), hh, hh, fc=RED + (TC,), ec='none'))
A.annotate('', xy=(xb - 0.17, cyy), xytext=(xa + hh / 2 + 0.03, cyy), arrowprops=dict(arrowstyle='-|>', color='#555555', lw=0.6, shrinkA=0, shrinkB=0))
sg = 0.55 * hh; R = 3 * sg; gx = np.linspace(-R, R, 201); GX, GY = np.meshgrid(gx, gx)
rgba = np.zeros(GX.shape + (4,)); rgba[..., :3] = RED; rgba[..., 3] = TC * hh ** 2 / (2 * np.pi * sg ** 2) * np.exp(-(GX ** 2 + GY ** 2) / (2 * sg ** 2))
A.imshow(rgba, extent=(xb - R, xb + R, cyy + R, cyy - R), interpolation='bilinear')
A.add_patch(Rectangle((xb - hh / 2, cyy - hh / 2), hh, hh, fill=False, ec='#555555', lw=0.4, ls=(0, (2, 1.5))))
x0, yb, hp = OX(4) + S / 2, yI + S - 0.03, S / 3.4; HT = 0.3   # 4b: profile through a row: overlapping Gaussians add up to the cell value
sp = 0.55 * hp; pk = hp ** 2 / (2 * np.pi * sp ** 2); xx = np.linspace(-1.6 * hp, 1.6 * hp, 300)
tot = sum(pk * np.exp(-((xx - i * hp) ** 2 + (j * hp) ** 2) / (2 * sp ** 2)) for i in range(-6, 7) for j in range(-6, 7))
A.add_patch(Rectangle((x0 - hp / 2, yb - HT), hp, HT, fc='none', ec='#2F3B4C', lw=0.5, ls=(0, (2, 1.5))))
for i in (-1, 0, 1): bump = pk * np.exp(-(xx - i * hp) ** 2 / (2 * sp ** 2)); A.fill_between(x0 + xx, yb, yb - HT * bump, fc=RED + (0.28,), ec='none'); A.plot(x0 + xx, yb - HT * bump, c='#C0392B', lw=0.5)
A.plot(x0 + xx, yb - HT * tot, c='#7B241C', lw=0.9); A.plot([x0 - 1.6 * hp, x0 + 1.6 * hp], [yb, yb], c='#555555', lw=0.5)
A.text(x0 + 1.6 * hp, yb - HT - 0.02, 'sum = $\\tau_{\\mathrm{cell}}$', ha='right', va='bottom', fontsize=5.5, color='#7B241C')
gimg = img(FIG1_CASE, FIG1_LOOK, 'gs_2M', FIG1_VIEW); msk = gimg.max(-1) > 6 / 255; ys, xs = np.where(msk)   # 5: converted 3DGS
cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2; s2 = max(np.ptp(xs), np.ptp(ys)) * 0.52
x_a, x_b = int(max(0, cx - s2)), int(min(800, cx + s2)); y_a, y_b = int(max(0, cy - s2)), int(min(800, cy + s2))
A.imshow(gimg[y_a:y_b, x_a:x_b], extent=(OX(5), OX(5) + S, yI + S, yI))
A.add_patch(Rectangle((OX(5), yI), S, S, fill=False, ec='black', lw=0.3))
A.set_xlim(0, W1); A.set_ylim(H1, 0)
for ext in ('png', 'pdf', 'svg'): fig.savefig(os.path.join(RESULTS, f'figure1.{ext}'), bbox_inches='tight', pad_inches=0.01)
fig.savefig(os.path.join(RESULTS, 'figure1.tif'), dpi=600, bbox_inches='tight', pad_inches=0.01, pil_kwargs={'compression': 'tiff_lzw'})
plt.close()

import matplotlib.pyplot as plt
sub = TS2[~TS2.Variant.str.startswith('Uniform grid vs')]
fig, axs = plt.subplots(1, 2, figsize=(6.69, 2.6), dpi=200, sharey=True)
for A, kn in zip(axs, ['wide', 'close-up']):
    d = sub[sub.Views == kn]; vars_ = list(dict.fromkeys(d.Variant)); x = np.arange(len(vars_)); w = 0.26
    for k, (ln, col) in enumerate(zip([l for _, l in LOOKS], ['#B8860B', '#C0392B', '#D98880'])):
        y = [float(d[(d.Look == ln) & (d.Variant == v)]['ΔPSNR (dB)'].iloc[0]) for v in vars_]
        A.bar(x + (k - 1) * w, y, w, label=ln, color=col)
    A.axhline(0, color='k', lw=0.6); A.set_xticks(x); A.set_xticklabels(vars_, rotation=30, ha='right', fontsize=6); A.set_title(kn, fontsize=8)
axs[0].set_ylabel('Median ΔPSNR vs 2 M (dB)', fontsize=7); axs[0].legend(fontsize=6, frameon=False)
plt.tight_layout(); fig.savefig(os.path.join(RESULTS, 'figureS1_ablation.png')); plt.close()
import sys, platform, importlib
vers = [f'python {sys.version.split()[0]} ({platform.platform()})']
for mod in ['numpy', 'scipy', 'pandas', 'matplotlib', 'PIL', 'torch', 'torchvision', 'lpips']:
    try:
        v = getattr(importlib.import_module(mod), '__version__', None)
        if v is None:
            from importlib.metadata import version as _ver
            v = _ver({'PIL': 'pillow'}.get(mod, mod))
        vers.append(f'{mod} {v}')
    except Exception: vers.append(f'{mod} (not installed)')
open(os.path.join(RESULTS, 'versions.txt'), 'w').write('\n'.join(vers) + '\n')
print('\n' + '\n'.join(vers))
print('\nwritten to', RESULTS)
