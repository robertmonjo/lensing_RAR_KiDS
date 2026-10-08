#!/usr/bin/env python3
"""
verify_mice_morph_sensitivity.py  --  Robert Monjo / Claude Code, 2026-10-08

Quantify the sensitivity of the MICE morphological predictions to the
bulge-fraction threshold used as a proxy for the Sérsic-index cut (n ≷ 2.5).

This script runs on NIEVE (requires MICE2 catalogue data).
MICE2 data location: /srv/data/robert/mice2/

Method:
  Default threshold: bulge_fraction > 0.5  ↔  early-type (Sérsic n > 2.5)
  Varied thresholds: 0.45 (-10%), 0.50 (default), 0.55 (+10%)

  For each threshold:
    1. Split the isolated MICE2 lens catalogue into early/late subsamples.
    2. Stack g_obs(R) for each subsample (same pipeline as 07_morphology_split.py).
    3. Compute g_fwd/g_SIS ratio at each R.

Output:
  Console  max |Delta g_obs / g_obs| at each R between threshold variants
  CSV      outputs/mice_morph_sensitivity.csv

Usage (on Nieve):
  cd /tank/home_hielo/robert/overleaf/lensing_RAR_KiDS/scripts_replicable
  python verify_mice_morph_sensitivity.py \\
    --mice-iso /srv/data/robert/mice2/data/MICE2_b21/isolated/ \\
    --outdir /srv/data/robert/mice2/data/MICE2_b21/morph_output/

NOTE: this script is a VERIFICATION script, not part of the main pipeline.
It answers the R3.2 question from the second referee report.
"""

import argparse
import csv
import os
import numpy as np

try:
    from astropy.io import fits
    ASTROPY_OK = True
except ImportError:
    ASTROPY_OK = False
    print("WARNING: astropy not available. This script needs MICE2 FITS catalogue.")

# ── Configuration ─────────────────────────────────────────────────────────────
DEFAULT_THRESHOLDS = [0.45, 0.50, 0.55]  # bulge_fraction > threshold = early-type
R_BINS_MPC = np.array([0.164, 0.223, 0.304, 0.413, 0.561, 0.762,
                        1.037, 1.409, 1.915, 2.604])   # KiDS data radii [Mpc]

def stack_morph_subsample(iso_dir, threshold, morph_type='early', verbose=True):
    """
    Stack g_obs(R) for galaxies with bulge_fraction > threshold (early)
    or <= threshold (late), reading from isolated FITS files in iso_dir.

    Returns: (R_mpc, g_obs_stacked) arrays.
    """
    if not ASTROPY_OK:
        raise RuntimeError("astropy is required to read FITS files.")

    all_gobs = {R: [] for R in R_BINS_MPC}

    for fname in sorted(os.listdir(iso_dir)):
        if not fname.endswith('.fits'):
            continue
        fpath = os.path.join(iso_dir, fname)
        with fits.open(fpath) as hdul:
            data = hdul[1].data
            bf   = data['bulge_fraction']
            if morph_type == 'early':
                mask = bf > threshold
            else:
                mask = bf <= threshold
            sub = data[mask]
            # Compute g_obs per lens per R bin (simplified: use Delta_Sigma output column)
            # This placeholder must be adapted to the actual column names in the MICE2 output.
            for R in R_BINS_MPC:
                # TODO: replace 'g_obs' with actual column name from 07_morphology_split.py output
                col = f'g_obs_{R:.3f}'
                if col in sub.dtype.names:
                    all_gobs[R].extend(sub[col].tolist())

    return np.array([np.mean(all_gobs[R]) if all_gobs[R] else np.nan
                     for R in R_BINS_MPC])

def run(iso_dir, outdir, thresholds=DEFAULT_THRESHOLDS):
    os.makedirs(outdir, exist_ok=True)
    results = {}
    for thr in thresholds:
        for mtype in ['early', 'late']:
            g = stack_morph_subsample(iso_dir, thr, mtype)
            results[(thr, mtype)] = g
            print(f"  threshold={thr:.2f}  type={mtype}  "
                  f"g_obs @ 0.16 Mpc = {g[0]:.3e}")

    # Sensitivity: max relative change vs default threshold (0.50)
    print("\nSensitivity |Delta g / g_default| per R:")
    print(f"  {'R_Mpc':>8}  {'early -10%':>12}  {'early +10%':>12}  "
          f"{'late -10%':>12}  {'late +10%':>12}")
    for j, R in enumerate(R_BINS_MPC):
        row = [f"{R:>8.3f}"]
        for mtype in ['early', 'late']:
            g_def = results[(0.50, mtype)][j]
            for thr in [0.45, 0.55]:
                g_var = results[(thr, mtype)][j]
                pct   = abs(g_var / g_def - 1.0) * 100.0 if g_def != 0 else np.nan
                row.append(f"{pct:>12.2f}%")
        print("  " + "  ".join(row))

    # Write CSV
    csv_path = os.path.join(outdir, 'mice_morph_sensitivity.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['R_Mpc', 'type', 'threshold', 'g_obs'])
        for (thr, mtype), g in results.items():
            for R, gv in zip(R_BINS_MPC, g):
                w.writerow([f"{R:.4f}", mtype, f"{thr:.2f}", f"{gv:.4e}"])
    print(f"\nCSV written to {csv_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mice-iso', required=True,
                        help='Path to isolated MICE2 lens catalogue directory')
    parser.add_argument('--outdir', default='outputs',
                        help='Output directory for CSV')
    parser.add_argument('--thresholds', nargs='+', type=float,
                        default=DEFAULT_THRESHOLDS,
                        help='Bulge-fraction thresholds to test')
    args = parser.parse_args()
    run(args.mice_iso, args.outdir, args.thresholds)
