#!/usr/bin/env python3
"""
mice_morph_threshold_sensitivity.py  --  Robert Monjo, 2026

Sensitivity of the MICE2 morphological stacked RAR to proxy threshold choices.

The MICE2 morphological classification uses two proxies that introduce
an irreducible mapping uncertainty relative to the KiDS observational
classifiers:

  (i)  Sérsic proxy: bulge fraction B/T as proxy for Sérsic index n > 2.5.
       Default split: B/T >= 0.50 = early.
  (ii) Colour proxy: rest-frame g-r as proxy for observed u-r > 2.5.
       Default split: g-r >= 0.60 = red.

This script quantifies how much the stacked g_obs(R) (and hence the
fitted HMG scale s) changes when each threshold is varied by ±10%:
  B/T  ∈ {0.45, 0.50, 0.55}
  g-r  ∈ {0.55, 0.60, 0.65}

Method:
  Run 07_morphology_split.py for each threshold combination, read the
  output allbins TXT files, and compute the maximum fractional change
  |g_obs(R,threshold) / g_obs(R,default) − 1| over all radial bins.

Usage (on Nieve):
  cd /tank/home_hielo/robert/overleaf/lensing_RAR_KiDS/MICE_n-body/scripts
  python mice_morph_threshold_sensitivity.py \\
    --morphfile /srv/data/robert/mice2/data/MICE2_isolated/07_morph_raw.fits \\
    --isodir    /srv/data/robert/mice2/data/MICE2_isolated/ \\
    --outdir    /srv/data/robert/mice2/data/MICE2_isolated/sensitivity/

Output files:
  sensitivity/morph_sersic_{late,early}_allbins_bt{BT}.txt  (one per threshold)
  sensitivity/morph_color_{blue,red}_allbins_gr{GR}.txt
  tables/mice_morph_threshold_sensitivity.csv
"""

import argparse
import csv
import os
import subprocess
import sys

import numpy as np

SCRIPT_07 = os.path.join(os.path.dirname(__file__), "07_morphology_split.py")

BT_THRESHOLDS = [0.45, 0.50, 0.55]   # default = 0.50
GR_THRESHOLDS = [0.55, 0.60, 0.65]   # default = 0.60

TYPES_BT = {"sersic_late": "late", "sersic_early": "early"}
TYPES_GR = {"color_blue":  "blue", "color_red":   "red"}


def run_split(morphfile, isodir, outdir, bt_split, gr_split):
    """Run 07_morphology_split.py with given thresholds; return output directory."""
    tag = f"bt{int(round(bt_split*100)):03d}_gr{int(round(gr_split*100)):03d}"
    subdir = os.path.join(outdir, tag)
    os.makedirs(subdir, exist_ok=True)
    cmd = [
        sys.executable, SCRIPT_07,
        "--morphfile", morphfile,
        "--isodir",    isodir,
        "--outdir",    subdir,
        "--bt_split",  str(bt_split),
        "--gr_split",  str(gr_split),
    ]
    print(f"Running: bt_split={bt_split}  gr_split={gr_split}  → {subdir}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ERROR:\n{result.stderr[-2000:]}")
        raise RuntimeError(f"07_morphology_split.py failed for bt={bt_split}, gr={gr_split}")
    return subdir, tag


def load_allbins(subdir, key):
    """Load morph_{key}_allbins.txt; return (log10_gbar, log10_gobs, r_Mpc) or None."""
    path = os.path.join(subdir, f"morph_{key}_allbins.txt")
    if not os.path.exists(path):
        return None
    d = np.loadtxt(path, comments="#")
    if d.ndim < 2 or d.shape[1] < 2:
        return None
    return d[:, 0], d[:, 1]   # log10_gbar, log10_gobs


def max_relative_change(g_default, g_varied):
    """Max |g_varied/g_default − 1| over all radial points where both are finite."""
    ok = np.isfinite(g_default) & np.isfinite(g_varied) & (g_default != 0)
    if ok.sum() == 0:
        return np.nan
    return np.max(np.abs(g_varied[ok] / g_default[ok] - 1.0)) * 100.0  # percent


def run(morphfile, isodir, outdir, tables_dir):
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)

    # Run 07_morphology_split.py for each threshold combination
    outputs = {}
    for bt in BT_THRESHOLDS:
        for gr in GR_THRESHOLDS:
            subdir, tag = run_split(morphfile, isodir, outdir, bt, gr)
            outputs[(bt, gr)] = (subdir, tag)

    # Default outputs
    def_subdir, _ = outputs[(0.50, 0.60)]

    # Compute sensitivity
    rows = []
    print("\nSensitivity  |Δg_obs / g_obs|  (%) relative to default threshold")
    print(f"  {'proxy':>12}  {'type':>6}  {'threshold':>10}  {'max|Δg/g| (%)':>15}")
    print("  " + "-" * 52)

    for proxy_types, default_thr, varied_thrs, fixed_key in [
        (TYPES_BT, (0.50, 0.60), [(0.45, 0.60), (0.55, 0.60)], "B/T"),
        (TYPES_GR, (0.50, 0.60), [(0.50, 0.55), (0.50, 0.65)], "g-r"),
    ]:
        g_defaults = {}
        for key, label in proxy_types.items():
            r = load_allbins(def_subdir, key)
            g_defaults[key] = 10.0**r[1] if r is not None else None

        for bt_v, gr_v in varied_thrs:
            sub_v, _ = outputs[(bt_v, gr_v)]
            thr_val = bt_v if fixed_key == "B/T" else gr_v
            for key, label in proxy_types.items():
                g_def = g_defaults[key]
                r_v   = load_allbins(sub_v, key)
                g_var = 10.0**r_v[1] if r_v is not None else None
                if g_def is None or g_var is None:
                    pct = np.nan
                else:
                    # Align arrays (both have N radial points from allbins)
                    n = min(len(g_def), len(g_var))
                    pct = max_relative_change(g_def[:n], g_var[:n])
                print(f"  {fixed_key:>12}  {label:>6}  {thr_val:>10.2f}  {pct:>14.2f}%")
                rows.append({
                    "proxy": fixed_key,
                    "type": label,
                    "threshold": thr_val,
                    "max_delta_gobs_pct": f"{pct:.2f}" if np.isfinite(pct) else "nan",
                })

    # Write CSV
    csv_path = os.path.join(tables_dir, "mice_morph_threshold_sensitivity.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["proxy", "type", "threshold", "max_delta_gobs_pct"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nCSV written to {csv_path}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--morphfile", required=True,
                   help="Path to 07_morph_raw.fits")
    p.add_argument("--isodir", required=True,
                   help="Directory containing mice2_isolated_bin{1..4}.fits")
    p.add_argument("--outdir",
                   default="/srv/data/robert/mice2/data/MICE2_isolated/sensitivity/",
                   help="Output directory for per-threshold stacks")
    p.add_argument("--tables-dir",
                   default="../../tables/",
                   help="Output directory for CSV summary")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.morphfile, args.isodir, args.outdir, args.tables_dir)
