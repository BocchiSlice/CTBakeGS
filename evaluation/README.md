# Evaluation

Scripts that reproduce the tables and figures of the paper.

The paper uses six public CTs from the TotalSegmentator dataset v2 (CC BY 4.0): s0138, s0289, s0931, s0970, s1121 and
s1340. s0289 is also evaluated with 5 mm slices.

## 1. Render and measure

Open `CTBakeGS.html?study=1` (Chrome, opened directly as a file). The batch panel takes:

- **CTs:** a folder of CTs. Every `ct.nii(.gz)` and every DICOM series is listed, and a 5 mm thick-slice copy can be
  added per case.
- **Output folder.**
- **Looks.**
- **Splat counts and ablations.**

For every CT × look it renders, from the same cameras at 800 × 800 on a black background, a converged reference
(progressive photoreal, 128 samples per view), the real-time photoreal view and every 3DGS variant, in 48 wide and 16
automatically chosen close-up views. It writes PNG images, `cameras.json`, `metrics_per_view.csv` (PSNR, foreground
PSNR, SSIM) and `scene.json` (settings, splat counts, conversion times, frame rates and the environment). Interrupted
runs resume where they stopped.

**Measure speed only** records conversion times and frame rates on another computer (no images), in `perf/`.

## 2. Tables and figures

```
pip install -r requirements.txt
python ctbakegs_tables_figures.py OUTPUT_FOLDER [--desktop SPEED_FOLDER] [--laptop SPEED_FOLDER] [--ct path/to/s0138/ct.nii.gz]
```

On the first run, LPIPS (VGG) is computed for every image with `study_lpips.py` and saved in the output folder. The
script then writes Tables 1–2, the supplementary tables, Figures 1–2 (PNG, PDF, SVG, TIFF) and `versions.txt` to
`OUTPUT_FOLDER/paper_results`.

- `--desktop`, `--laptop`: `perf` folders from *Measure speed only* on other computers (extra rows in Table 2).
- `--ct`: the CT of case s0138, for the CT slice in Figure 1 (needs `nibabel`).

The same steps are in `ctbakegs_tables_figures.ipynb` for Google Colab.

## Other scripts

- `study_lpips.py OUTPUT_FOLDER [--weights FOLDER] [--device cpu|cuda|mps]`: LPIPS only. `--weights` takes a folder
  with `vgg16-397923af.pth` to avoid the download.
- `error_maps.py OUTPUT_FOLDER`: error heatmaps and error by image region.
