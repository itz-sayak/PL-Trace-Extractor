#!/usr/bin/env python3
"""
process_video.py

Pipeline to extract per-particle photoluminescence traces from a video (TIFF/MP4/etc.).
- Registers frames to remove drift using phase cross-correlation
- Builds average image and detects bright spots (blob detection)
- Performs circular aperture photometry with local background subtraction
- Outputs `traces.csv` and example plots

Usage example:
python process_video.py --input path/to/movie.tif --output traces.csv --radius 3

"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import imageio
import cv2
from skimage.registration import phase_cross_correlation
from scipy.ndimage import shift as ndi_shift
from skimage.feature import blob_log
from scipy.ndimage import gaussian_filter, median_filter


def read_frames(path, target_fps=None):
    """Read frames from a video or multipage TIFF using imageio.
    
    Args:
        path: path to video file
        target_fps: if provided, subsample frames to achieve this frame rate.
                    If None, all frames are used.
    
    Returns a float32 numpy array shape (n_frames, H, W).
    """
    reader = imageio.get_reader(path)
    
    # Try to get video FPS metadata
    try:
        meta = reader.get_meta_data()
        video_fps = meta.get('fps', None)
    except Exception:
        video_fps = None
    
    # Compute frame step for subsampling
    frame_step = 1
    if target_fps is not None and video_fps is not None and video_fps > target_fps:
        frame_step = int(round(video_fps / target_fps))
        print(f"Video FPS: {video_fps:.1f}, target FPS: {target_fps}, using every {frame_step} frame(s)")
    elif target_fps is not None and video_fps is None:
        print(f"Warning: Could not detect video FPS, using all frames")
    
    frames = []
    for idx, im in enumerate(reader):
        if idx % frame_step != 0:
            continue
        arr = np.asarray(im)
        if arr.ndim == 3 and arr.shape[2] in (3, 4):
            # convert RGB -> grayscale if needed
            arr = np.mean(arr[..., :3], axis=2)
        frames.append(arr.astype(np.float32))
    reader.close()
    stack = np.stack(frames, axis=0)
    return stack, video_fps, frame_step


def compute_registration_shifts(frames, reference_index=0, upsample_factor=10):
    """Compute shift for each frame to align to reference frame.
    Returns shifts array shape (n_frames, 2) in (y, x) order.
    """
    ref = frames[reference_index]
    shifts = []
    for i in range(frames.shape[0]):
        if i == reference_index:
            shifts.append((0.0, 0.0))
            continue
        shift_est, error, diffphase = phase_cross_correlation(ref, frames[i], upsample_factor=upsample_factor)
        shifts.append(tuple(shift_est))
    return np.array(shifts, dtype=np.float32)


def apply_shifts(frames, shifts, order=1):
    """Apply shifts to frames. shifts in (y, x) order. Returns aligned frames.
    Uses `scipy.ndimage.shift` which shifts input by given offsets.
    """
    aligned = np.empty_like(frames)
    for i in range(frames.shape[0]):
        sy, sx = shifts[i]
        # shift() moves contents by given shift; to align moving->ref, we apply +shift
        aligned[i] = ndi_shift(frames[i], shift=(sy, sx), order=order, mode='nearest')
    return aligned


def detect_blobs(avg_image, min_sigma=1, max_sigma=5, num_sigma=10, threshold=0.02, 
                  min_intensity_percentile=20, edge_margin=10):
    """Detect blobs with additional filtering to remove false positives.
    
    Args:
        min_intensity_percentile: spots in regions darker than this percentile of the 
                                  local background are rejected (removes dark-region FPs)
        edge_margin: reject spots within this many pixels of image edge
    """
    # normalize image to [0,1]
    img = avg_image - np.min(avg_image)
    if np.max(img) > 0:
        img = img / np.max(img)
    blobs = blob_log(img, min_sigma=min_sigma, max_sigma=max_sigma, num_sigma=num_sigma, threshold=threshold)
    # blob_log returns (y, x, sigma). We'll convert sigma->radius ~ sqrt(2)*sigma
    if blobs.size == 0:
        return np.zeros((0, 2)), np.zeros((0,))
    
    coords = blobs[:, :2]
    sigmas = blobs[:, 2]
    radii = np.sqrt(2) * sigmas
    
    # Filter 1: remove spots too close to edges
    H, W = avg_image.shape
    valid = ((coords[:, 0] > edge_margin) & (coords[:, 0] < H - edge_margin) &
             (coords[:, 1] > edge_margin) & (coords[:, 1] < W - edge_margin))
    
    # Filter 2: remove spots in dark regions (background intensity threshold)
    # Use a smoothed version to estimate local background
    bg_smooth = gaussian_filter(avg_image, sigma=20)
    bg_threshold = np.percentile(bg_smooth, min_intensity_percentile)
    for i, (y, x) in enumerate(coords):
        local_bg = bg_smooth[int(y), int(x)]
        if local_bg < bg_threshold:
            valid[i] = False
    
    # Filter 3: remove spots that are too dim (peak intensity filter)
    # Check actual intensity at spot location
    peak_intensities = []
    for y, x in coords:
        yi, xi = int(y), int(x)
        # small window around spot
        y1, y2 = max(0, yi-2), min(H, yi+3)
        x1, x2 = max(0, xi-2), min(W, xi+3)
        peak_intensities.append(avg_image[y1:y2, x1:x2].max())
    peak_intensities = np.array(peak_intensities)
    
    # Keep only spots with peak intensity above median of all detected spots
    if len(peak_intensities) > 0:
        intensity_threshold = np.percentile(peak_intensities[valid], 25) if valid.sum() > 0 else 0
        valid = valid & (peak_intensities >= intensity_threshold)
    
    return coords[valid], radii[valid]


def aperture_photometry(frame, yx, radius):
    """Compute aperture sum and local background median (annulus) for a single spot in one frame.
    yx: (y,x) coordinates (float allowed). radius in pixels.
    Returns (signal, background_median, net)
    """
    H, W = frame.shape
    y0, x0 = yx
    # build bounding box (clamped)
    r_ext = int(np.ceil(radius + 8))
    y1 = max(0, int(np.floor(y0 - r_ext)))
    y2 = min(H, int(np.ceil(y0 + r_ext) + 1))
    x1 = max(0, int(np.floor(x0 - r_ext)))
    x2 = min(W, int(np.ceil(x0 + r_ext) + 1))
    sub = frame[y1:y2, x1:x2]
    yy, xx = np.mgrid[y1:y2, x1:x2]
    dist = np.sqrt((yy - y0) ** 2 + (xx - x0) ** 2)
    aperture_mask = dist <= radius
    # annulus for background
    ann_r_in = radius + 3
    ann_r_out = radius + 8
    ann_mask = (dist >= ann_r_in) & (dist <= ann_r_out)
    signal = sub[aperture_mask].sum()
    # background: median of annulus if available, else median of local box excluding aperture
    if np.any(ann_mask):
        b_med = np.median(sub[ann_mask])
    else:
        # fallback
        b_med = np.median(sub[~aperture_mask])
    net = signal - b_med * aperture_mask.sum()
    return float(signal), float(b_med), float(net)


def render_frame_with_spots(frame, coords, circle_radius=8, color=(0, 0, 255), thickness=1):
    """Render a single frame with detected spots overlaid as circles.
    
    Args:
        frame: 2D grayscale image (float32)
        coords: Nx2 array of (y, x) spot coordinates
        circle_radius: radius of circles to draw (pixels)
        color: BGR color tuple for circles
        thickness: line thickness for circles
    
    Returns:
        RGB uint8 image with circles drawn
    """
    # Normalize frame to 0-255
    img = frame - np.min(frame)
    if np.max(img) > 0:
        img = img / np.max(img)
    img = (img * 255).astype(np.uint8)
    
    # Convert grayscale to BGR for colored circles
    img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    # Draw circles at each spot location
    for (y, x) in coords:
        center = (int(round(x)), int(round(y)))  # cv2 uses (x, y) order
        cv2.circle(img_bgr, center, circle_radius, color, thickness)
    
    return img_bgr


def write_output_video(aligned_frames, coords, output_path, fps=25, circle_radius=8, verbose=True):
    """Write output video with spot overlays.
    
    Args:
        aligned_frames: (N, H, W) array of aligned frames
        coords: Nx2 array of spot coordinates (y, x)
        output_path: path to output .mp4 file
        fps: output video frame rate
        circle_radius: radius of spot circles
        verbose: print progress
    """
    n_frames, H, W = aligned_frames.shape
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (W, H))
    
    if verbose:
        print(f"Writing output video to {output_path}...")
    
    for i in tqdm(range(n_frames), desc='Rendering video', disable=not verbose):
        frame_bgr = render_frame_with_spots(aligned_frames[i], coords, circle_radius=circle_radius)
        out.write(frame_bgr)
    
    out.release()
    if verbose:
        print(f"Saved output video: {output_path}")


def run_pipeline(path, out_csv=None, radius=3, min_sigma=1, max_sigma=5, threshold=0.02, upsample=10, fps=None, verbose=True):
    frames, video_fps, frame_step = read_frames(path, target_fps=fps)
    n_frames, H, W = frames.shape
    if verbose:
        print(f"Loaded {n_frames} frames of size {H}x{W}")
        if video_fps:
            effective_fps = video_fps / frame_step
            print(f"Effective frame rate: {effective_fps:.2f} fps")

    # optional smoothing for registration (helps SNR)
    smooth_ref = gaussian_filter(frames, sigma=(0, 1, 1))

    if verbose:
        print("Computing registration shifts (this may take a bit)...")
    shifts = compute_registration_shifts(smooth_ref, upsample_factor=upsample)

    if verbose:
        print("Applying shifts to align frames...")
    aligned = apply_shifts(frames, shifts, order=1)

    avg = np.mean(aligned, axis=0)
    # small smoothing for blob detection
    avg_s = gaussian_filter(avg, sigma=1.0)

    if verbose:
        print("Detecting blobs on average image...")
    coords, radii = detect_blobs(avg_s, min_sigma=min_sigma, max_sigma=max_sigma, num_sigma=10, threshold=threshold)
    if verbose:
        print(f"Detected {len(coords)} candidate spots")

    # prepare DataFrame for traces
    rows = []
    for fid in tqdm(range(n_frames), desc='Photometry'):  # iterate frames
        frm = aligned[fid]
        for sid, (coord, rad) in enumerate(zip(coords, radii)):
            y, x = float(coord[0]), float(coord[1])
            # shift is already applied to frames, so coords measured on avg image are valid
            signal, bmed, net = aperture_photometry(frm, (y, x), radius)

            # Localization uncertainty (sigma) calculation
            # Using formula (from provided reference image):
            # sigma = sqrt( s^2 / N + (a^2/12) / N + (4 * sqrt(pi) * s^3 * b^2) / (a * N^2) )
            # where:
            #  s = PSF Gaussian sigma (px) -> estimated from detected radius: s = rad / sqrt(2)
            #  N = total photons (we use net signal as estimate)
            #  b = background per pixel (we use bg_median)
            #  a = pixel size (px) — default 1.0 unless overridden via CLI
            try:
                s = float(rad) / np.sqrt(2.0)
                N = float(net)
                b = float(bmed)
                a = float(run_pipeline._px_size) if hasattr(run_pipeline, '_px_size') else 1.0
                if N > 0:
                    term1 = (s ** 2) / N
                    term2 = (a ** 2 / 12.0) / N
                    term3 = (4.0 * np.sqrt(np.pi) * (s ** 3) * (b ** 2)) / (a * (N ** 2))
                    loc_sigma = float(np.sqrt(max(0.0, term1 + term2 + term3)))
                else:
                    loc_sigma = float('nan')
            except Exception:
                loc_sigma = float('nan')

            rows.append({
                'frame': int(fid),
                'spot_id': int(sid),
                'y': float(y),
                'x': float(x),
                'radius': float(radius),
                'raw_signal': float(signal),
                'bg_median': float(bmed),
                'net_signal': float(net),
                'loc_uncertainty': loc_sigma
            })

    df = pd.DataFrame(rows)
    if out_csv is None:
        out_csv = os.path.splitext(os.path.basename(path))[0] + '_traces.csv'
    df.to_csv(out_csv, index=False)
    if verbose:
        print(f"Saved traces to {out_csv}")

    # save a small summary plot: a few traces and the average image with detections
    fig_dir = os.path.splitext(out_csv)[0] + '_figs'
    os.makedirs(fig_dir, exist_ok=True)

    # plot average image with detected spots
    plt.figure(figsize=(6, 6))
    plt.imshow(avg, cmap='gray', origin='lower')
    if len(coords) > 0:
        ys = coords[:, 0]
        xs = coords[:, 1]
        plt.scatter(xs, ys, facecolors='none', edgecolors='r', s=40)
    plt.title('Average image with detected spots')
    plt.savefig(os.path.join(fig_dir, 'avg_with_spots.png'), dpi=150)
    plt.close()

    # plot example traces (up to 6 spots)
    nplot = min(6, len(coords))
    if nplot > 0:
        fig, ax = plt.subplots(nplot, 1, figsize=(6, 2 * nplot), sharex=True)
        for i in range(nplot):
            sub = df[df['spot_id'] == i].sort_values('frame')
            ax[i].plot(sub['frame'], sub['net_signal'], '-', color='C0')
            ax[i].set_ylabel(f'spot {i}')
        ax[-1].set_xlabel('frame')
        fig.suptitle('Example net traces')
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(os.path.join(fig_dir, 'example_traces.png'), dpi=150)
        plt.close()

    # Generate output video with spot overlays
    output_video_path = os.path.join(fig_dir, 'output.mp4')
    # Use effective fps for output video (or 25 as default)
    output_fps = video_fps / frame_step if video_fps else 25
    write_output_video(aligned, coords, output_video_path, fps=output_fps, circle_radius=radius*3, verbose=verbose)

    return df, shifts


def parse_args():
    p = argparse.ArgumentParser(description='Extract per-particle PL traces from a video')
    p.add_argument('--input', '-i', required=True, help='Input video/multipage TIFF (path)')
    p.add_argument('--output', '-o', required=False, help='Output CSV path (default: <input>_traces.csv)')
    p.add_argument('--fps', type=float, default=None, help='Target frame rate (fps). If set, subsamples video to this rate.')
    p.add_argument('--px-size', type=float, default=1.0, help='Pixel size `a` used in localization uncertainty formula (units: pixels).')
    p.add_argument('--radius', type=int, default=3, help='Aperture radius (pixels)')
    p.add_argument('--min-sigma', type=float, default=1.0, help='Min sigma for blob detection')
    p.add_argument('--max-sigma', type=float, default=4.0, help='Max sigma for blob detection')
    p.add_argument('--threshold', type=float, default=None, help='Normalized threshold for blob detection (0..1). Overrides --sensitivity if set.')
    p.add_argument('--sensitivity', type=float, default=50.0, help='Detection sensitivity 0-100 (higher=more spots). Maps to threshold internally.')
    p.add_argument('--upsample', type=int, default=10, help='Upsample factor for subpixel registration')
    p.add_argument('--no-plot', dest='plot', action='store_false', help='Do not write plots')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    
    # Convert sensitivity (0-100) to threshold (0.1 - 0.005)
    # Higher sensitivity -> lower threshold -> more spots detected
    # sensitivity=0 -> threshold=0.10 (very few spots)
    # sensitivity=50 -> threshold=0.02 (default)
    # sensitivity=100 -> threshold=0.005 (many spots)
    if args.threshold is not None:
        threshold = args.threshold
    else:
        # Map sensitivity [0, 100] to threshold [0.10, 0.005] (log scale)
        sens = max(0.0, min(100.0, args.sensitivity))
        threshold = 0.10 * (0.05 ** (sens / 100.0))
    
    print(f"Detection sensitivity: {args.sensitivity:.1f} -> threshold: {threshold:.4f}")
    
    # store pixel size on run_pipeline function so inner loop can access without changing signatures
    run_pipeline._px_size = args.px_size
    df, shifts = run_pipeline(args.input, out_csv=args.output, radius=args.radius,
                              min_sigma=args.min_sigma, max_sigma=args.max_sigma,
                              threshold=threshold, upsample=args.upsample,
                              fps=args.fps, verbose=True)
    print('Done.')
