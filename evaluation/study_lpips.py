#!/usr/bin/env python3
"""
LPIPS for the output of the batch study (Run study in the evaluation build).

  pip install lpips torch torchvision pillow numpy
  python study_lpips.py OUTPUT_FOLDER [--weights FOLDER_WITH_vgg16-397923af.pth] [--alex] [--device cpu|cuda|mps]

OUTPUT_FOLDER is the folder chosen as "Output folder" (case/look/... subfolders), or a folder of
case__look.zip files (the download fallback). Each scene's images/<method>/<kind>_###.png is compared with
images/ref/<kind>_###.png. The backbone weights are downloaded by torchvision, or read from --weights:
vgg16-397923af.pth (and alexnet-owt-7be5be79.pth for --alex).

Writes, in OUTPUT_FOLDER:
  lpips_per_view.csv   case, look, variant, view_kind, view, lpips_vgg[, lpips_alex]
  summary_all.csv      per case/look/variant/view_kind: mean PSNR, PSNR(fg), SSIM (from metrics_per_view.csv) and LPIPS
Existing rows in lpips_per_view.csv are reused, so the script can be stopped and run again.
"""
import argparse, csv, io, os, re, shutil, sys, zipfile
from collections import defaultdict

import numpy as np
from PIL import Image


class Scene:
    def __init__(self, path):
        self.path = path
        self.zip = zipfile.ZipFile(path) if path.endswith('.zip') else None
        names = self.zip.namelist() if self.zip else [
            os.path.relpath(os.path.join(d, f), path).replace(os.sep, '/')
            for d, _, fs in os.walk(path) for f in fs]
        self.names = set(names)

    def read(self, name):
        if self.zip:
            return self.zip.read(name)
        with open(os.path.join(self.path, name), 'rb') as f:
            return f.read()

    def img(self, name):
        return np.array(Image.open(io.BytesIO(self.read(name))).convert('RGB'))


def find_scenes(root):
    out = []
    for d, _, fs in os.walk(root):
        if 'metrics_per_view.csv' in fs:
            out.append(d)
        out += [os.path.join(d, f) for f in fs if f.endswith('.zip') and '__' in f]
    return sorted(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('root')
    ap.add_argument('--weights', default='', help='folder with vgg16-397923af.pth (and alexnet-owt-7be5be79.pth)')
    ap.add_argument('--alex', action='store_true', help='also AlexNet LPIPS (supplementary)')
    ap.add_argument('--device', default=None)
    ap.add_argument('--time-limit', type=float, default=0, help='stop after this many seconds (results so far are saved; run again to continue)')
    ap.add_argument('--no-summary', action='store_true')
    a = ap.parse_args()

    # --weights: put the backbone weights where torchvision looks, so nothing is downloaded
    if a.weights:
        th = os.path.join(os.path.expanduser('~'), '.cache', 'ctbakegs_torch')
        ck = os.path.join(th, 'hub', 'checkpoints')
        os.makedirs(ck, exist_ok=True)
        need = ['vgg16-397923af.pth'] + (['alexnet-owt-7be5be79.pth'] if a.alex else [])
        for n in need:
            src = os.path.join(a.weights, n)
            if not os.path.exists(src):
                sys.exit(f'missing {src}')
            dst = os.path.join(ck, n)
            if not os.path.exists(dst):
                try:
                    os.symlink(os.path.abspath(src), dst)
                except OSError:
                    shutil.copy(src, dst)
        os.environ['TORCH_HOME'] = th

    import time
    T0 = time.time()
    import torch, lpips
    dev = a.device or ('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    nets = {'vgg': lpips.LPIPS(net='vgg', verbose=False).to(dev).eval()}
    if a.alex:
        nets['alex'] = lpips.LPIPS(net='alex', verbose=False).to(dev).eval()

    def tens(x):
        return torch.from_numpy(x).permute(2, 0, 1)[None].float().div(127.5).sub(1).to(dev)

    per_path = os.path.join(a.root, 'lpips_per_view.csv')
    cols = ['case', 'look', 'variant', 'view_kind', 'view'] + [f'lpips_{k}' for k in nets]
    done = {}
    if os.path.exists(per_path):
        with open(per_path) as f:
            for r in csv.DictReader(f):
                if all(c in r and r[c] != '' for c in cols):
                    done[(r['case'], r['look'], r['variant'], r['view_kind'], int(r['view']))] = r
    rows = list(done.values())
    metr = []

    pat = re.compile(r'^images/([^/]+)/(wide|close)_(\d+)\.png$')
    scenes = find_scenes(a.root)
    print(f'{len(scenes)} scenes, device {dev}')
    def save():
        with open(per_path + '.tmp', 'w', newline='') as f:
            w = csv.DictWriter(f, cols)
            w.writeheader()
            w.writerows(rows)
        os.replace(per_path + '.tmp', per_path)

    # collect every pending image of every scene first, then work through them by variant priority,
    # so that a partial run gives balanced results (all cases for the main comparisons first)
    PRI = ['2M', 'realtimePhotoreal', 'naiveOpacity', 'noAA', 'uniformCells', 'adaptiveMatched', '1M', '3M', 'voxelCells', 'standard']
    jobs = []
    for si, sp in enumerate(scenes):
        s = Scene(sp)
        if 'metrics_per_view.csv' not in s.names:
            continue
        mrows = list(csv.DictReader(io.StringIO(s.read('metrics_per_view.csv').decode())))
        metr += mrows
        case, look = mrows[0]['case'], mrows[0]['look']
        for n in sorted(s.names):
            m = pat.match(n)
            if not m or m.group(1) in ('ref', 'ref8') or m.group(1).startswith('diff'):
                continue
            meth, kind, v = m.group(1), m.group(2), int(m.group(3))
            ref = f'images/ref/{kind}_{v:03d}.png'
            var = meth[3:] if meth.startswith('gs_') else {'rt': 'realtimePhotoreal', 'std': 'standard'}.get(meth, meth)
            key = (case, look, var, kind, v)
            if ref in s.names and key not in done:
                jobs.append((PRI.index(var) if var in PRI else len(PRI), si, key, n, ref, s))
    jobs.sort(key=lambda j: (j[0], j[1], j[4]))
    print(f'{len(jobs)} images pending', flush=True)
    out_of_time = False
    refc = {}
    with torch.no_grad():
        for _, si, key, n, ref, s in jobs:
            rk = (si, ref)
            if rk not in refc:
                refc = {rk: tens(s.img(ref))}
            x = tens(s.img(n))
            r = dict(zip(cols[:5], key))
            for k, net in nets.items():
                r[f'lpips_{k}'] = f'{net(x, refc[rk]).item():.5f}'
            rows.append(r)
            done[key] = r
            if len(rows) % 25 == 0:
                save()
            if a.time_limit and time.time() - T0 > a.time_limit:
                out_of_time = True
                break
    if jobs:
        save()
    print('OUT_OF_TIME' if out_of_time else 'ALL_SCENES_PROCESSED', len(rows), 'rows', flush=True)
    if a.no_summary:
        return

    # merged per-scene summary (mean over views)
    acc = defaultdict(lambda: defaultdict(list))
    for r in metr:
        k = (r['case'], r['look'], r['variant'], r['view_kind'])
        for c in ('psnr', 'psnr_fg', 'ssim'):
            if r.get(c) not in (None, ''):
                acc[k][c].append(float(r[c]))
    for r in rows:
        k = (r['case'], r['look'], r['variant'], r['view_kind'])
        for c in cols[5:]:
            acc[k][c].append(float(r[c]))
    mc = ['psnr', 'psnr_fg', 'ssim'] + cols[5:]
    with open(os.path.join(a.root, 'summary_all.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['case', 'look', 'variant', 'view_kind'] + [f'{c}_mean' for c in mc] + [f'n_{c}' for c in mc])
        for k in sorted(acc):
            d = acc[k]
            w.writerow(list(k) + [f'{np.mean(d[c]):.5f}' if d[c] else '' for c in mc] + [len(d[c]) for c in mc])
    print('wrote', per_path, 'and summary_all.csv')


if __name__ == '__main__':
    main()
