# Nanoparticle Photoluminescence (PL) Trace Extraction Pipeline

A computer vision pipeline for extracting time-resolved photoluminescence intensity traces from semiconductor nanoparticle microscopy videos. Designed to handle characteristic "blinking" behavior of quantum dots, perovskite nanocrystals, and similar emitters.

---

##  Methodology

The pipeline applies the following image processing and computer vision techniques:

### 1. Frame Acquisition & Preprocessing
- **Video decoding**: Reads MP4, TIFF stacks, AVI, and other formats via `imageio` + FFmpeg backend
- **Grayscale conversion**: RGB frames are converted to single-channel grayscale by averaging color channels
- **Frame rate control**: Optional `--fps` parameter subsamples the video to a target frame rate (useful for reducing processing time on high-speed videos)

### 2. Drift Correction (Image Registration)
Microscopy images often exhibit translational drift due to stage instability or thermal effects. We correct this using:

- **Phase cross-correlation** (`skimage.registration.phase_cross_correlation`)
  - Computes the shift between each frame and a reference frame (first frame by default)
  - Works in Fourier space: finds the peak of the cross-power spectrum
  - Subpixel accuracy achieved via upsampling (default 10×)
- **Affine transformation**: Shifts are applied using `scipy.ndimage.shift` with bilinear interpolation
- **Gaussian pre-smoothing** (σ=1 px) before correlation to reduce noise sensitivity

### 3. Temporal Averaging
- After alignment, all frames are averaged to produce a high-SNR reference image
- This average image reveals persistent emitters while suppressing shot noise

### 4. Spot Detection (Blob Detection)
Bright nanoparticle spots are detected using the **Laplacian of Gaussian (LoG)** method:

- **`skimage.feature.blob_log`**: Convolves image with LoG kernels at multiple scales (σ range)
- Detects spots as local maxima in scale-space
- Parameters:
  - `--min-sigma` / `--max-sigma`: range of spot sizes (in pixels) to detect
  - `--threshold`: normalized intensity threshold (0–1) for detection sensitivity

### 5. False Positive Filtering
To remove spurious detections, we apply three filters:

1. **Edge exclusion**: Spots within 10 px of image boundaries are rejected
2. **Dark-region rejection**: Spots in regions where the smoothed background (Gaussian σ=20 px) falls below the 20th percentile are removed (eliminates detections in areas with no sample)
3. **Intensity thresholding**: Spots with peak intensity below the 25th percentile of all detections are discarded (removes dim noise peaks)

### 6. Aperture Photometry
For each detected spot in every frame, we measure intensity using circular aperture photometry:

- **Signal aperture**: Circular region of radius `r` (default 3 px) centered on spot
- **Background annulus**: Ring from `r+3` to `r+8` px
- **Background estimation**: Median of pixels in annulus (robust to outliers)
- **Net signal**: `raw_signal - (background_median × aperture_area)`

This approach accounts for local background variations and provides accurate intensity measurements even with non-uniform illumination.

### Localization uncertainty (standard error)

We compute an estimate of the localization standard error (uncertainty) for each fitted spot position following the formula shown in the project notes. In KaTeX/LaTeX notation the formula is:

$$
\sigma = \sqrt{\frac{s^{2}}{N} + \frac{a^{2}}{12N} + \frac{4\sqrt{\pi}\, s^{3} b^{2}}{a N^{2}} }
$$

where:
- $\sigma$ : estimated standard error of the fitted position (same units as $a$; default pixels)
- $s$ : estimated PSF Gaussian sigma (in pixels). In the code we approximate $s$ from the detected blob radius $r$ via $s = r/\sqrt{2}$.
- $N$ : total number of signal photons (we use the background-subtracted aperture sum `net_signal` as an estimate)
- $b$ : background per pixel (we use the local `bg_median` value)
- $a$ : effective pixel size (units of length per pixel, default `a = 1.0` so uncertainty is returned in pixels). Use `--px-size` to set `a` (e.g. nm/pixel) to report uncertainties in physical units.

Notes:
- The formula assumes Gaussian PSF and Poisson-dominated photon statistics; the third term accounts for the effect of background noise on localization precision.
- If `N <= 0` the uncertainty is undefined and recorded as `NaN` in the CSV.


### 7. Output Generation
- **CSV file**: Per-spot, per-frame data with columns: `frame, spot_id, y, x, radius, raw_signal, bg_median, net_signal`
- **Detection overlay**: Average image with detected spots marked (red circles)
- **Example traces**: Time series plots showing blinking dynamics for sample spots

---

## Installation

Requires Python 3.8+. Install dependencies in a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

---

## Usage

```powershell
python process_video.py --input YourVideo.mp4 --output traces.csv --radius 3
 
Simple example using your provided sample video:

```powershell
python process_video.py --input Perovskite.mp4 --fps 10 --output traces_10fps.csv
```
```

### Command-Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--input`, `-i` | (required) | Input video or TIFF stack path |
| `--output`, `-o` | `<input>_traces.csv` | Output CSV path |
| `--fps` | None | Target frame rate; subsamples video if set |
| `--radius` | 3 | Aperture radius in pixels for photometry |
| `--px-size` | 1.0 | Pixel size `a` used in localization uncertainty formula (units: pixels) |
| `--min-sigma` | 1.0 | Minimum blob sigma for LoG detection |
| `--max-sigma` | 4.0 | Maximum blob sigma for LoG detection |
| `--threshold` | None | Normalized detection threshold (0–1). If set, overrides `--sensitivity` |
| `--sensitivity` | 50.0 | Detection sensitivity (0–100). Higher = more sensitive; maps to an internal threshold if `--threshold` is not set |
| `--upsample` | 10 | Subpixel upsampling factor for registration |

### Example with FPS Control

```powershell
# Process at 10 fps (useful for long/high-framerate videos)
python process_video.py --input Perovskite.mp4 --fps 10 --output traces_10fps.csv
```

### Full command example (recommended)

This is a complete example showing the commonly used options. Adjust `--sensitivity`/`--threshold`, `--radius`, and `--px-size` for your dataset.

```powershell
python process_video.py \
  --input C:\Users\Sayakdutta\Downloads\Nano\Perovskite.mp4 \
  --output C:\Users\Sayakdutta\Downloads\Nano\traces_full.csv \
  --fps 10 \
  --radius 3 \
  --px-size 1.0 \
  --sensitivity 50 \
  --min-sigma 1.0 \
  --max-sigma 4.0 \
  --upsample 10
```

If you prefer to set an explicit detection threshold instead of using `--sensitivity`:

```powershell
python process_video.py --input Perovskite.mp4 --threshold 0.015 --radius 3 --px-size 1.0
```

---

## Output Files

| File | Description |
|------|-------------|
| `traces.csv` | Table with columns: `frame, spot_id, y, x, radius, raw_signal, bg_median, net_signal` |
| `traces_figs/avg_with_spots.png` | Drift-corrected average image with detected spots circled |
| `traces_figs/example_traces.png` | Sample intensity vs. frame plots showing blinking |

Note: the output CSV now includes an additional column `loc_uncertainty` which is the estimated standard error (localization precision) of the fitted position for each spot and frame, computed using the formula provided in the project documentation. The calculation uses the detected PSF width, the per-frame net signal, local background, and the pixel-size parameter `--px-size`.

---

## Dependencies

- `numpy`, `scipy` – numerical operations
- `pandas` – data handling
- `matplotlib` – plotting
- `imageio`, `imageio-ffmpeg` – video I/O
- `scikit-image` – registration, blob detection
- `tqdm` – progress bars

---

## References

- Phase cross-correlation: Guizar-Sicairos et al., *Optics Letters* 33, 156 (2008)
- Laplacian of Gaussian blob detection: Lindeberg, *IJCV* 30, 79 (1998)
- Aperture photometry: standard technique in astronomical imaging

---

## Future Enhancements

- 2D Gaussian PSF fitting for sub-pixel localization
- ON/OFF state detection and blinking statistics
- Spot tracking/linking across frames (via `trackpy`)
- Streaming mode for very large videos
