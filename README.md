# Nanoparticle Photoluminescence (PL) Trace Extraction Pipeline

A computer vision pipeline for extracting time-resolved photoluminescence intensity traces from semiconductor nanoparticle microscopy videos. Designed to handle characteristic "blinking" behavior of quantum dots, perovskite nanocrystals, and similar emitters.

---

##  Methodology
### PL-Trace-Extractor

Extract photoluminescence (PL) intensity traces from microscopy videos of single nanoparticles (quantum dots, perovskites, etc.). The pipeline detects bright spots, corrects drift, extracts per-frame intensities, fits 2D Gaussian PSFs for sub-pixel localization, and produces labeled visual outputs so you can map CSV traces to physical particles.

Project and code: https://github.com/itz-sayak/PL-Trace-Extractor

---

## Highlights (what's new)

- Spot ID labeling across visual outputs: detected particles are annotated with `spot_0`, `spot_1`, ... in the average image, the output video (`output.mp4`), and saved fit figures. This makes it trivial to cross-reference CSV columns with images/videos.
- Improved `visualize_gauss_fit.py`: clearer saved filenames (`gauss_fit_SPOT{ID}_frame{N}.png`) and bold figure titles that show `SPOT #X` prominently.
- CSV now includes Gaussian-fit columns (`gauss_x`, `gauss_y`, `gauss_amp`, `gauss_sigma`, `gauss_bg`, `gauss_success`) and `loc_uncertainty` (estimated localization standard error).

---

## Quick start

1. Create and activate a virtual environment (Windows PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Run the main pipeline on a video (simple example):

```powershell
python process_video.py --input Perovskite.mp4 --output traces.csv --radius 3
```

3. Visualize a fitted Gaussian for a specific spot (spot IDs start at 0):

```powershell
python visualize_gauss_fit.py --input Perovskite.mp4 --spot 3 --frame 0 --save --fit-radius 7
```

The `--spot` argument selects which detected particle to visualize (e.g. `--spot 3` corresponds to `spot_3` in the CSV and `avg_with_spots.png`).

---

## Recommended workflow

- Run `process_video.py` to produce:
  - `traces.csv` — per-frame, per-spot table with intensity, fit results, and uncertainty
  - `traces_figs/avg_with_spots.png` — drift-corrected average image annotated with spot IDs
  - `output.mp4` — optional annotated video showing circular markers and spot ID numbers
  - example trace plots in `traces_figs/`
- Use `visualize_gauss_fit.py --spot <N>` to inspect the 2D Gaussian fit and residuals for a particular particle and frame.

---

## CLI options (short reference)

- `--input, -i` (required): input video or TIFF stack path
- `--output, -o`: output CSV path (default: `<input>_traces.csv`)
- `--fps`: target frame rate to subsample the input video (optional)
- `--radius`: aperture radius (px) for photometry (default 3)
- `--px-size`: pixel size `a` for localization uncertainty (units per pixel; default 1.0)
- `--min-sigma`, `--max-sigma`: LoG detection sigma range (px)
- `--threshold`: explicit normalized detection threshold (0–1); if unset, `--sensitivity` is used
- `--sensitivity`: sensitivity (0–100) mapped internally to a detection threshold
- `--upsample`: subpixel registration upsampling (default 10)

See `process_video.py --help` for full option descriptions.

---

## Output CSV schema (columns you'll find in `traces.csv`)

- `frame`: frame index
- `spot_id`: integer ID (0-indexed) matching annotated images/videos
- `y`, `x`: integer pixel coordinates (rounded) of detected spot
- `radius`: detected blob radius (px)
- `raw_signal`, `bg_median`, `net_signal`: aperture photometry values
- `gauss_x`, `gauss_y`: sub-pixel fitted center (if `gauss_success` is True)
- `gauss_amp`, `gauss_sigma`, `gauss_bg`: Gaussian fit parameters
- `gauss_success`: True/False fit converged
- `loc_uncertainty`: estimated localization standard error (same units as `--px-size`)

---

## Spot labeling & cross-referencing

- The average image `traces_figs/avg_with_spots.png` shows all detected spots with their `spot_id` numbers. Use this image to map visual particles to CSV columns.
- The annotated video (`output.mp4`) shows a yellow numeric label next to each detected spot on every frame, so you can visually follow `spot_5` across time.
- `visualize_gauss_fit.py --spot N` will load the detected coordinate for `spot_N` and produce a fit figure named `gauss_fit_SPOTN_frameM.png` (when `--save` is specified).

---

## Gaussian fitting and localization uncertainty

- 2D Gaussian PSF fitting is performed with bounded `scipy.optimize.curve_fit`. When successful, the fit provides sub-pixel coordinates and amplitude.
- Localization uncertainty is estimated using the formula (implemented in the pipeline):

$$
\sigma = \sqrt{\frac{s^{2}}{N} + \frac{a^{2}}{12N} + \frac{4\sqrt{\pi}\, s^{3} b^{2}}{a N^{2}} }
$$

where parameters are explained in the code and recorded in the CSV columns. Use `--px-size` to convert uncertainties into physical units (e.g. nm).

---

## Examples

- Process and generate labeled outputs (recommended):

```powershell
python process_video.py --input Perovskite.mp4 --output traces_full.csv --fps 10 --radius 3 --sensitivity 50
```

- Inspect spot 3 at frame 0 and save the fit figure:

```powershell
python visualize_gauss_fit.py --input Perovskite.mp4 --spot 3 --frame 0 --save
```

---

## Dependencies

- `numpy`, `scipy`, `pandas`, `matplotlib`, `imageio`, `imageio-ffmpeg`, `scikit-image`, `tqdm`, `opencv-python`

Install with `pip install -r requirements.txt`.

---

## Contact / contributions

If you find issues or want to contribute, please open an issue or pull request on GitHub: https://github.com/itz-sayak/PL-Trace-Extractor

If you'd like, I can also:
- Run the pipeline on your primary dataset and attach the labeled video + a mapping table of `spot_id -> (x,y)` and example traces.
- Add a short README subsection describing how to visually relabel or exclude spots manually.

---

License: MIT (see repository settings)
