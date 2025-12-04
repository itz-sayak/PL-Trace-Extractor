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
```

### Command-Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--input`, `-i` | (required) | Input video or TIFF stack path |
| `--output`, `-o` | `<input>_traces.csv` | Output CSV path |
| `--fps` | None | Target frame rate; subsamples video if set |
| `--radius` | 3 | Aperture radius in pixels for photometry |
| `--min-sigma` | 1.0 | Minimum blob sigma for LoG detection |
| `--max-sigma` | 4.0 | Maximum blob sigma for LoG detection |
| `--threshold` | 0.02 | Normalized detection threshold (0–1) |
| `--upsample` | 10 | Subpixel upsampling factor for registration |

### Example with FPS Control

```powershell
# Process at 10 fps (useful for long/high-framerate videos)
python process_video.py --input Perovskite.mp4 --fps 10 --output traces_10fps.csv
```

---

## Output Files

| File | Description |
|------|-------------|
| `traces.csv` | Table with columns: `frame, spot_id, y, x, radius, raw_signal, bg_median, net_signal` |
| `traces_figs/avg_with_spots.png` | Drift-corrected average image with detected spots circled |
| `traces_figs/example_traces.png` | Sample intensity vs. frame plots showing blinking |

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
