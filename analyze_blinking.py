#!/usr/bin/env python3
"""
analyze_blinking.py

Analyze photoluminescence blinking statistics from extracted traces.
- Classifies intensity time traces into ON / OFF / GRAY states
- Computes dwell-time distributions for each state
- Generates histograms and probability density plots

Usage examples:
    # Analyze all spots, save figures
    python analyze_blinking.py --input traces_gauss.csv --save

    # Analyze specific spots (0, 3, 5)
    python analyze_blinking.py --input traces_gauss.csv --spots 0,3,5 --save

    # Custom thresholds (as fraction of max intensity)
    python analyze_blinking.py --input traces_gauss.csv --on-thresh 0.5 --off-thresh 0.2

"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


def classify_states(intensity, on_thresh_frac=0.5, off_thresh_frac=0.2):
    """
    Classify intensity trace into ON / OFF / GRAY states.
    
    Args:
        intensity: 1D array of intensity values
        on_thresh_frac: fraction of (max - baseline) above which state is ON
        off_thresh_frac: fraction of (max - baseline) below which state is OFF
        
    Returns:
        states: array of same length with values 'ON', 'OFF', 'GRAY'
        thresholds: dict with 'on_threshold', 'off_threshold', 'baseline', 'max_int'
    """
    intensity = np.asarray(intensity, dtype=float)
    
    # Estimate baseline (background) as lower percentile
    baseline = np.percentile(intensity, 10)
    max_int = np.percentile(intensity, 95)  # Use 95th percentile to avoid outliers
    
    dynamic_range = max_int - baseline
    
    if dynamic_range <= 0:
        # No blinking detected, all OFF or constant
        return np.array(['OFF'] * len(intensity)), {
            'on_threshold': baseline,
            'off_threshold': baseline,
            'baseline': baseline,
            'max_int': max_int
        }
    
    on_threshold = baseline + on_thresh_frac * dynamic_range
    off_threshold = baseline + off_thresh_frac * dynamic_range
    
    states = np.empty(len(intensity), dtype=object)
    states[intensity >= on_threshold] = 'ON'
    states[intensity <= off_threshold] = 'OFF'
    states[(intensity > off_threshold) & (intensity < on_threshold)] = 'GRAY'
    
    thresholds = {
        'on_threshold': on_threshold,
        'off_threshold': off_threshold,
        'baseline': baseline,
        'max_int': max_int
    }
    
    return states, thresholds


def compute_dwell_times(states):
    """
    Compute dwell times (consecutive frames) for each state.
    
    Args:
        states: array of state labels ('ON', 'OFF', 'GRAY')
        
    Returns:
        dict with keys 'ON', 'OFF', 'GRAY', each containing list of dwell times
    """
    dwell_times = {'ON': [], 'OFF': [], 'GRAY': []}
    
    if len(states) == 0:
        return dwell_times
    
    current_state = states[0]
    current_duration = 1
    
    for i in range(1, len(states)):
        if states[i] == current_state:
            current_duration += 1
        else:
            dwell_times[current_state].append(current_duration)
            current_state = states[i]
            current_duration = 1
    
    # Don't forget the last segment
    dwell_times[current_state].append(current_duration)
    
    return dwell_times


def plot_trace_with_states(frames, intensity, states, thresholds, spot_id, ax=None):
    """Plot intensity trace colored by state."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 4))
    
    # Color mapping
    colors = {'ON': 'green', 'OFF': 'red', 'GRAY': 'gray'}
    
    # Plot segments by state
    for state in ['ON', 'OFF', 'GRAY']:
        mask = states == state
        ax.scatter(frames[mask], intensity[mask], c=colors[state], s=3, label=state, alpha=0.7)
    
    # Plot thresholds
    ax.axhline(thresholds['on_threshold'], color='green', linestyle='--', alpha=0.5, label=f"ON thresh ({thresholds['on_threshold']:.1f})")
    ax.axhline(thresholds['off_threshold'], color='red', linestyle='--', alpha=0.5, label=f"OFF thresh ({thresholds['off_threshold']:.1f})")
    
    ax.set_xlabel('Frame')
    ax.set_ylabel('Net Signal (a.u.)')
    ax.set_title(f'SPOT #{spot_id} — Intensity Trace with State Classification')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    return ax


def plot_dwell_histograms(dwell_times, spot_id, frame_time=1.0, ax_on=None, ax_off=None, ax_gray=None):
    """
    Plot histograms of dwell times for ON, OFF, GRAY states.
    
    Args:
        dwell_times: dict with 'ON', 'OFF', 'GRAY' keys
        spot_id: spot identifier for title
        frame_time: time per frame (for x-axis units)
    """
    fig = None
    if ax_on is None or ax_off is None or ax_gray is None:
        fig, (ax_on, ax_off, ax_gray) = plt.subplots(1, 3, figsize=(14, 4))
    
    for ax, state, color in [(ax_on, 'ON', 'green'), (ax_off, 'OFF', 'red'), (ax_gray, 'GRAY', 'gray')]:
        times = np.array(dwell_times[state]) * frame_time
        if len(times) > 0:
            # Use auto bins or fixed bins depending on data range
            max_time = times.max()
            if max_time > 0:
                bins = np.arange(0, max_time + frame_time * 2, frame_time)
                if len(bins) > 50:
                    bins = 30  # Fall back to auto bins if too many
            else:
                bins = 10
            
            ax.hist(times, bins=bins, color=color, alpha=0.7, edgecolor='black', linewidth=0.5)
            ax.set_xlabel(f'{state} Time (frames)')
            ax.set_ylabel('Count')
            ax.set_title(f'SPOT #{spot_id} — {state} Time Histogram\n(n={len(times)}, mean={times.mean():.1f})')
            
            # Add statistics text
            if len(times) > 1:
                ax.axvline(times.mean(), color='black', linestyle='--', linewidth=1.5, label=f'Mean: {times.mean():.1f}')
                ax.axvline(np.median(times), color='blue', linestyle=':', linewidth=1.5, label=f'Median: {np.median(times):.1f}')
                ax.legend(fontsize=8)
        else:
            ax.set_xlabel(f'{state} Time (frames)')
            ax.set_ylabel('Count')
            ax.set_title(f'SPOT #{spot_id} — {state} Time Histogram\n(No events)')
            ax.text(0.5, 0.5, 'No events', ha='center', va='center', transform=ax.transAxes, fontsize=12)
        
        ax.grid(True, alpha=0.3)
    
    if fig is not None:
        fig.tight_layout()
    
    return fig


def plot_probability_density(dwell_times, spot_id, frame_time=1.0, ax=None):
    """
    Plot probability density functions for ON and OFF times.
    Uses kernel density estimation (KDE) if enough data points.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = None
    
    for state, color in [('ON', 'green'), ('OFF', 'red')]:
        times = np.array(dwell_times[state]) * frame_time
        if len(times) >= 3:
            # Compute histogram-based PDF
            hist, bin_edges = np.histogram(times, bins='auto', density=True)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            ax.bar(bin_centers, hist, width=bin_edges[1]-bin_edges[0], alpha=0.4, color=color, 
                   label=f'{state} (n={len(times)})', edgecolor='black', linewidth=0.3)
            
            # Overlay KDE if enough points
            if len(times) >= 10:
                try:
                    kde = stats.gaussian_kde(times)
                    x_range = np.linspace(0, times.max() * 1.1, 200)
                    ax.plot(x_range, kde(x_range), color=color, linewidth=2, linestyle='-')
                except Exception:
                    pass  # KDE can fail for degenerate data
        elif len(times) > 0:
            # Just plot as stems if few points
            ax.stem(times, np.ones(len(times)) * 0.1, linefmt=color, markerfmt=f'{color[0]}o', 
                    basefmt=' ', label=f'{state} (n={len(times)})')
    
    ax.set_xlabel('Dwell Time (frames)')
    ax.set_ylabel('Probability Density')
    ax.set_title(f'SPOT #{spot_id} — Probability Density: ON vs OFF Times')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    
    if fig is not None:
        fig.tight_layout()
    
    return fig


def analyze_spot(df_spot, spot_id, on_thresh_frac, off_thresh_frac, frame_time=1.0, save_dir=None):
    """
    Full blinking analysis for a single spot.
    
    Returns:
        dict with statistics and figures
    """
    # Sort by frame
    df_spot = df_spot.sort_values('frame').reset_index(drop=True)
    frames = df_spot['frame'].values
    intensity = df_spot['net_signal'].values
    
    # Classify states
    states, thresholds = classify_states(intensity, on_thresh_frac, off_thresh_frac)
    
    # Compute dwell times
    dwell_times = compute_dwell_times(states)
    
    # Compute statistics
    stats_dict = {
        'spot_id': spot_id,
        'n_frames': len(frames),
        'n_on_events': len(dwell_times['ON']),
        'n_off_events': len(dwell_times['OFF']),
        'n_gray_events': len(dwell_times['GRAY']),
        'mean_on_time': np.mean(dwell_times['ON']) if dwell_times['ON'] else np.nan,
        'mean_off_time': np.mean(dwell_times['OFF']) if dwell_times['OFF'] else np.nan,
        'mean_gray_time': np.mean(dwell_times['GRAY']) if dwell_times['GRAY'] else np.nan,
        'on_fraction': np.sum(states == 'ON') / len(states),
        'off_fraction': np.sum(states == 'OFF') / len(states),
        'gray_fraction': np.sum(states == 'GRAY') / len(states),
        'on_threshold': thresholds['on_threshold'],
        'off_threshold': thresholds['off_threshold'],
    }
    
    # Create comprehensive figure using GridSpec for flexible layout
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(f'Blinking Analysis — SPOT #{spot_id}', fontsize=14, fontweight='bold', y=0.98)
    
    import matplotlib.gridspec as gridspec
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)
    
    # 1. Trace with states (top row, full width)
    ax_trace = fig.add_subplot(gs[0, :])
    plot_trace_with_states(frames, intensity, states, thresholds, spot_id, ax=ax_trace)
    
    # 2. Histograms (middle row)
    ax_on = fig.add_subplot(gs[1, 0])
    ax_off = fig.add_subplot(gs[1, 1])
    ax_gray = fig.add_subplot(gs[1, 2])
    plot_dwell_histograms(dwell_times, spot_id, frame_time, ax_on, ax_off, ax_gray)
    
    # 3. Probability density (bottom left, spans 2 columns)
    ax_pdf = fig.add_subplot(gs[2, 0:2])
    plot_probability_density(dwell_times, spot_id, frame_time, ax=ax_pdf)
    
    # 4. State pie chart (bottom right)
    ax_pie = fig.add_subplot(gs[2, 2])
    state_counts = [np.sum(states == s) for s in ['ON', 'OFF', 'GRAY']]
    colors_pie = ['green', 'red', 'gray']
    labels_pie = [f'ON ({stats_dict["on_fraction"]*100:.1f}%)',
                  f'OFF ({stats_dict["off_fraction"]*100:.1f}%)',
                  f'GRAY ({stats_dict["gray_fraction"]*100:.1f}%)']
    ax_pie.pie(state_counts, labels=labels_pie, colors=colors_pie, autopct='', startangle=90)
    ax_pie.set_title(f'SPOT #{spot_id} — State Distribution')
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'blinking_SPOT{spot_id}.png')
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()
    
    return {
        'stats': stats_dict,
        'dwell_times': dwell_times,
        'states': states,
        'thresholds': thresholds,
        'figure': fig
    }


def main():
    parser = argparse.ArgumentParser(description='Analyze PL blinking statistics')
    parser.add_argument('--input', '-i', required=True, help='Input traces CSV file')
    parser.add_argument('--spots', '-s', type=str, default=None, 
                        help='Comma-separated spot IDs to analyze (default: all spots)')
    parser.add_argument('--on-thresh', type=float, default=0.5,
                        help='ON threshold as fraction of dynamic range (default: 0.5)')
    parser.add_argument('--off-thresh', type=float, default=0.2,
                        help='OFF threshold as fraction of dynamic range (default: 0.2)')
    parser.add_argument('--frame-time', type=float, default=1.0,
                        help='Time per frame in your units (default: 1.0 = frames)')
    parser.add_argument('--save', action='store_true', help='Save figures to blinking_figs/')
    parser.add_argument('--output-dir', type=str, default='blinking_figs',
                        help='Output directory for figures (default: blinking_figs)')
    parser.add_argument('--summary', action='store_true', help='Print summary statistics table')
    args = parser.parse_args()
    
    # Load data
    print(f"Loading traces from: {args.input}")
    df = pd.read_csv(args.input)
    
    # Determine spots to analyze
    all_spots = sorted(df['spot_id'].unique())
    if args.spots:
        spot_ids = [int(s.strip()) for s in args.spots.split(',')]
        # Validate
        spot_ids = [s for s in spot_ids if s in all_spots]
        if not spot_ids:
            print(f"Error: None of the specified spots found. Available: {all_spots[:10]}...")
            return
    else:
        spot_ids = all_spots
    
    print(f"Analyzing {len(spot_ids)} spot(s): {spot_ids[:10]}{'...' if len(spot_ids) > 10 else ''}")
    print(f"Thresholds: ON > {args.on_thresh*100:.0f}%, OFF < {args.off_thresh*100:.0f}% of dynamic range")
    
    save_dir = args.output_dir if args.save else None
    
    all_stats = []
    for spot_id in spot_ids:
        df_spot = df[df['spot_id'] == spot_id]
        if len(df_spot) < 10:
            print(f"Skipping spot {spot_id}: only {len(df_spot)} frames")
            continue
        
        print(f"\n{'='*50}")
        print(f"SPOT #{spot_id}")
        print(f"{'='*50}")
        
        result = analyze_spot(
            df_spot, spot_id,
            on_thresh_frac=args.on_thresh,
            off_thresh_frac=args.off_thresh,
            frame_time=args.frame_time,
            save_dir=save_dir
        )
        
        stats = result['stats']
        all_stats.append(stats)
        
        print(f"  ON events: {stats['n_on_events']}, mean duration: {stats['mean_on_time']:.1f} frames")
        print(f"  OFF events: {stats['n_off_events']}, mean duration: {stats['mean_off_time']:.1f} frames")
        print(f"  GRAY events: {stats['n_gray_events']}, mean duration: {stats['mean_gray_time']:.1f} frames")
        print(f"  State fractions: ON={stats['on_fraction']*100:.1f}%, OFF={stats['off_fraction']*100:.1f}%, GRAY={stats['gray_fraction']*100:.1f}%")
    
    # Summary table
    if args.summary and all_stats:
        print("\n" + "="*80)
        print("SUMMARY STATISTICS")
        print("="*80)
        summary_df = pd.DataFrame(all_stats)
        print(summary_df.to_string(index=False))
        
        if args.save:
            summary_path = os.path.join(args.output_dir, 'blinking_summary.csv')
            summary_df.to_csv(summary_path, index=False)
            print(f"\nSaved summary: {summary_path}")


if __name__ == '__main__':
    main()
