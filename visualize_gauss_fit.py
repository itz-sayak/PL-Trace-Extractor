#!/usr/bin/env python3
"""
visualize_gauss_fit.py

Visualize 2D Gaussian PSF fits for selected spots from a video.
Shows: raw data, fitted Gaussian surface, residuals, and 1D cross-sections.

Usage:
    python visualize_gauss_fit.py --input Perovskite.mp4 --spot 0 --frame 0
    python visualize_gauss_fit.py --input Perovskite.mp4 --spot 0 --frame 0,10,50,100
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import imageio
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter, shift as ndi_shift
from skimage.registration import phase_cross_correlation
from skimage.feature import blob_log


def gaussian_2d(coords, amplitude, x0, y0, sigma, offset):
    """2D Gaussian function."""
    x, y = coords
    g = offset + amplitude * np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))
    return g.ravel()


def gaussian_2d_full(x, y, amplitude, x0, y0, sigma, offset):
    """2D Gaussian (non-flattened) for plotting."""
    return offset + amplitude * np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))


def read_and_align_frames(path, max_frames=None):
    """Read video and align frames."""
    reader = imageio.get_reader(path)
    frames = []
    for i, im in enumerate(reader):
        if max_frames and i >= max_frames:
            break
        arr = np.asarray(im)
        if arr.ndim == 3 and arr.shape[2] in (3, 4):
            arr = np.mean(arr[..., :3], axis=2)
        frames.append(arr.astype(np.float32))
    reader.close()
    frames = np.stack(frames, axis=0)
    
    # Compute shifts
    smooth = gaussian_filter(frames, sigma=(0, 1, 1))
    ref = smooth[0]
    shifts = []
    for i in range(frames.shape[0]):
        if i == 0:
            shifts.append((0.0, 0.0))
        else:
            shift_est, _, _ = phase_cross_correlation(ref, smooth[i], upsample_factor=10)
            shifts.append(tuple(shift_est))
    shifts = np.array(shifts, dtype=np.float32)
    
    # Apply shifts
    aligned = np.empty_like(frames)
    for i in range(frames.shape[0]):
        aligned[i] = ndi_shift(frames[i], shift=shifts[i], order=1, mode='nearest')
    
    return aligned


def detect_spots(avg_image, min_sigma=1, max_sigma=4, threshold=0.02):
    """Detect spots on average image."""
    img = avg_image - np.min(avg_image)
    if np.max(img) > 0:
        img = img / np.max(img)
    blobs = blob_log(img, min_sigma=min_sigma, max_sigma=max_sigma, num_sigma=10, threshold=threshold)
    if blobs.size == 0:
        return np.zeros((0, 2)), np.zeros((0,))
    coords = blobs[:, :2]
    radii = np.sqrt(2) * blobs[:, 2]
    return coords, radii


def fit_and_visualize(frame, y0, x0, spot_id, frame_id, fit_radius=7, save_path=None):
    """Fit 2D Gaussian and create visualization."""
    H, W = frame.shape
    
    # Extract sub-image
    y1 = max(0, int(np.floor(y0 - fit_radius)))
    y2 = min(H, int(np.ceil(y0 + fit_radius) + 1))
    x1 = max(0, int(np.floor(x0 - fit_radius)))
    x2 = min(W, int(np.ceil(x0 + fit_radius) + 1))
    
    sub = frame[y1:y2, x1:x2].astype(np.float64)
    yy, xx = np.mgrid[y1:y2, x1:x2]
    
    # Initial guesses
    offset_init = np.percentile(sub, 10)
    amp_init = sub.max() - offset_init
    sigma_init = 1.5
    
    # Fit
    bounds_lower = [0, x1, y1, 0.3, 0]
    bounds_upper = [np.inf, x2, y2, 10.0, np.inf]
    
    try:
        popt, pcov = curve_fit(
            gaussian_2d,
            (xx, yy),
            sub.ravel(),
            p0=[amp_init, x0, y0, sigma_init, offset_init],
            bounds=(bounds_lower, bounds_upper),
            maxfev=1000
        )
        amplitude, x_fit, y_fit, sigma_fit, bg_fit = popt
        fit_success = True
    except Exception as e:
        print(f"Fit failed: {e}")
        amplitude, x_fit, y_fit, sigma_fit, bg_fit = amp_init, x0, y0, sigma_init, offset_init
        fit_success = False
    
    # Generate fitted surface
    fitted = gaussian_2d_full(xx, yy, amplitude, x_fit, y_fit, sigma_fit, bg_fit)
    residuals = sub - fitted
    
    # Create figure with multiple panels
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(f'2D Gaussian PSF Fit — SPOT #{spot_id} — Frame {frame_id}\n'
                 f'Fit: x={x_fit:.2f}, y={y_fit:.2f}, σ={sigma_fit:.2f}, A={amplitude:.1f}, bg={bg_fit:.1f}',
                 fontsize=14, fontweight='bold')
    
    # 1. Raw data (2D image)
    ax1 = fig.add_subplot(2, 3, 1)
    im1 = ax1.imshow(sub, cmap='hot', origin='lower', extent=[x1, x2, y1, y2])
    ax1.scatter([x_fit], [y_fit], c='cyan', marker='+', s=100, linewidths=2, label='Fit center')
    ax1.scatter([x0], [y0], c='lime', marker='x', s=80, linewidths=2, label='Initial')
    ax1.set_title('Raw Data')
    ax1.set_xlabel('x (px)')
    ax1.set_ylabel('y (px)')
    ax1.legend(fontsize=8)
    plt.colorbar(im1, ax=ax1, shrink=0.8)
    
    # 2. Fitted Gaussian (2D image)
    ax2 = fig.add_subplot(2, 3, 2)
    im2 = ax2.imshow(fitted, cmap='hot', origin='lower', extent=[x1, x2, y1, y2])
    ax2.scatter([x_fit], [y_fit], c='cyan', marker='+', s=100, linewidths=2)
    ax2.set_title('Fitted 2D Gaussian')
    ax2.set_xlabel('x (px)')
    ax2.set_ylabel('y (px)')
    plt.colorbar(im2, ax=ax2, shrink=0.8)
    
    # 3. Residuals (2D image)
    ax3 = fig.add_subplot(2, 3, 3)
    vmax = max(abs(residuals.min()), abs(residuals.max()))
    im3 = ax3.imshow(residuals, cmap='RdBu_r', origin='lower', extent=[x1, x2, y1, y2],
                     vmin=-vmax, vmax=vmax)
    ax3.set_title(f'Residuals (RMS={np.std(residuals):.2f})')
    ax3.set_xlabel('x (px)')
    ax3.set_ylabel('y (px)')
    plt.colorbar(im3, ax=ax3, shrink=0.8)
    
    # 4. 3D surface plot
    ax4 = fig.add_subplot(2, 3, 4, projection='3d')
    xx_plot, yy_plot = np.meshgrid(np.arange(x1, x2), np.arange(y1, y2))
    ax4.plot_surface(xx_plot, yy_plot, sub, cmap='hot', alpha=0.7, linewidth=0)
    ax4.plot_wireframe(xx_plot, yy_plot, fitted, color='cyan', linewidth=0.5, alpha=0.8)
    ax4.set_title('3D: Data (surface) + Fit (wireframe)')
    ax4.set_xlabel('x')
    ax4.set_ylabel('y')
    ax4.set_zlabel('Intensity')
    
    # 5. X cross-section through center
    ax5 = fig.add_subplot(2, 3, 5)
    y_center_idx = int(round(y_fit - y1))
    y_center_idx = np.clip(y_center_idx, 0, sub.shape[0] - 1)
    x_vals = np.arange(x1, x2)
    ax5.plot(x_vals, sub[y_center_idx, :], 'ko-', markersize=4, label='Data')
    x_fine = np.linspace(x1, x2, 100)
    y_fine_x = gaussian_2d_full(x_fine, y_fit, amplitude, x_fit, y_fit, sigma_fit, bg_fit)
    ax5.plot(x_fine, y_fine_x, 'r-', linewidth=2, label='Gaussian fit')
    ax5.axvline(x_fit, color='cyan', linestyle='--', alpha=0.7, label=f'x₀={x_fit:.2f}')
    ax5.set_title(f'X cross-section (y={y_fit:.1f})')
    ax5.set_xlabel('x (px)')
    ax5.set_ylabel('Intensity')
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)
    
    # 6. Y cross-section through center
    ax6 = fig.add_subplot(2, 3, 6)
    x_center_idx = int(round(x_fit - x1))
    x_center_idx = np.clip(x_center_idx, 0, sub.shape[1] - 1)
    y_vals = np.arange(y1, y2)
    ax6.plot(y_vals, sub[:, x_center_idx], 'ko-', markersize=4, label='Data')
    y_fine = np.linspace(y1, y2, 100)
    y_fine_y = gaussian_2d_full(x_fit, y_fine, amplitude, x_fit, y_fit, sigma_fit, bg_fit)
    ax6.plot(y_fine, y_fine_y, 'r-', linewidth=2, label='Gaussian fit')
    ax6.axvline(y_fit, color='cyan', linestyle='--', alpha=0.7, label=f'y₀={y_fit:.2f}')
    ax6.set_title(f'Y cross-section (x={x_fit:.1f})')
    ax6.set_xlabel('y (px)')
    ax6.set_ylabel('Intensity')
    ax6.legend(fontsize=8)
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()
    
    return {
        'x_fit': x_fit, 'y_fit': y_fit, 'amplitude': amplitude,
        'sigma': sigma_fit, 'background': bg_fit, 'success': fit_success
    }


def main():
    parser = argparse.ArgumentParser(description='Visualize 2D Gaussian PSF fits')
    parser.add_argument('--input', '-i', required=True, help='Input video path')
    parser.add_argument('--spot', '-s', type=int, default=0, help='Spot ID to visualize')
    parser.add_argument('--frame', '-f', type=str, default='0', help='Frame number(s), comma-separated')
    parser.add_argument('--fit-radius', type=int, default=7, help='Fitting window half-size')
    parser.add_argument('--save', action='store_true', help='Save figures to files')
    args = parser.parse_args()
    
    frame_ids = [int(f.strip()) for f in args.frame.split(',')]
    
    print(f"Loading and aligning video: {args.input}")
    max_frame_needed = max(frame_ids) + 1
    aligned = read_and_align_frames(args.input, max_frames=max_frame_needed)
    
    print("Detecting spots on average image...")
    avg = np.mean(aligned, axis=0)
    avg_s = gaussian_filter(avg, sigma=1.0)
    coords, radii = detect_spots(avg_s)
    print(f"Detected {len(coords)} spots")
    
    if args.spot >= len(coords):
        print(f"Error: spot {args.spot} not found. Max spot ID is {len(coords)-1}")
        return
    
    y0, x0 = coords[args.spot]
    print(f"Spot {args.spot} initial position: x={x0:.1f}, y={y0:.1f}")
    
    for fid in frame_ids:
        if fid >= aligned.shape[0]:
            print(f"Warning: frame {fid} not available (max={aligned.shape[0]-1})")
            continue
        
        save_path = None
        if args.save:
            save_path = f"gauss_fit_SPOT{args.spot}_frame{fid}.png"
        
        print(f"\n--- Fitting SPOT #{args.spot}, frame {fid} ---")
        result = fit_and_visualize(aligned[fid], y0, x0, args.spot, fid, 
                                   fit_radius=args.fit_radius, save_path=save_path)
        print(f"Result: x={result['x_fit']:.3f}, y={result['y_fit']:.3f}, "
              f"σ={result['sigma']:.3f}, A={result['amplitude']:.1f}, bg={result['background']:.1f}")


if __name__ == '__main__':
    main()
