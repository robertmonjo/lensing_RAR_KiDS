"""Construct the B21-style MICE band from two isolation variants.

Brouwer+2021 Sect. 5.3: the MICE band in Fig. 9 spans between two stacked
signals that differ only in the isolation selection criterion:
  - Lower limit : isolation with TRUE simulation redshifts (z_true)
  - Upper limit : isolation with photo-z perturbed redshifts
                  z_photo = z_true + N(0, sigma_z*(1+z)), sigma_z = 0.02

Both variants are produced by the same pipeline (steps 02-05) with:
  step 02 run 1: default (--suffix "")       -> rar_stacked_bin{b}.txt
  step 02 run 2: --photoz_sigma 0.02         -> rar_stacked_photoz_bin{b}.txt
                 --suffix _photoz

This script combines them into rar_band_b21_bin{b}.txt (one file per mass bin)
and rar_band_b21_allbins.txt (all bins concatenated, for the main figure).

Output columns: r_Mpc  log10_gbar  log10_gobs_ztrue  log10_gobs_zphotoz

Usage:
    python 08_b21_band.py --indir ../data/MICE2_isolated/ \
                          --outdir ../data/MICE2_isolated/
"""

import argparse
import os
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="Build B21-style MICE band")
    p.add_argument("--indir",  default="../data/MICE2_isolated/")
    p.add_argument("--outdir", default="../data/MICE2_isolated/")
    p.add_argument("--suffix_ztrue",  default="",
                   help="Suffix for z-true stacked files (default: none)")
    p.add_argument("--suffix_zphotoz", default="_photoz",
                   help="Suffix for photo-z stacked files (default: _photoz)")
    return p.parse_args()


def load_stacked(indir, b, suffix):
    """Load rar_stacked{suffix}_bin{b}.txt -> (r_Mpc, log10_gbar, log10_gobs)."""
    fname = os.path.join(indir, f"rar_stacked{suffix}_bin{b}.txt")
    if not os.path.exists(fname):
        print(f"  WARNING: {fname} not found — skipping bin {b}")
        return None
    d = np.loadtxt(fname, comments="#")
    return d   # columns: r_Mpc, log10_gbar, log10_gobs


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    all_rows = []

    for b in range(1, 5):
        zt = load_stacked(args.indir, b, args.suffix_ztrue)
        zp = load_stacked(args.indir, b, args.suffix_zphotoz)

        if zt is None or zp is None:
            print(f"Bin {b}: missing data, skipping.")
            continue

        # Use z-true gbar as reference; gobs: ztrue=lower, zphotoz=upper
        # (photo-z misses some neighbours -> more contaminated isolated sample
        #  -> higher stacked g_obs)
        gobs_lo = np.minimum(zt[:, 2], zp[:, 2])   # usually zt
        gobs_hi = np.maximum(zt[:, 2], zp[:, 2])   # usually zp

        out = np.column_stack([zt[:, 0], zt[:, 1], gobs_lo, gobs_hi])
        fname = os.path.join(args.outdir, f"rar_band_b21_bin{b}.txt")
        np.savetxt(fname, out,
                   header=("r_Mpc  log10_gbar  log10_gobs_ztrue  log10_gobs_zphotoz\n"
                            f"(B21 band: ztrue=lower, zphotoz=upper, bin {b})"),
                   comments="# ")
        print(f"Bin {b}: saved {fname}")
        print(f"  gobs_ztrue  range: [{zt[:, 2].min():.3f}, {zt[:, 2].max():.3f}]")
        print(f"  gobs_zphotoz range: [{zp[:, 2].min():.3f}, {zp[:, 2].max():.3f}]")
        print(f"  Max band width: {(gobs_hi - gobs_lo).max():.4f} dex")

        all_rows.append(out)

    if all_rows:
        all_out = np.vstack(all_rows)
        fname_all = os.path.join(args.outdir, "rar_band_b21_allbins.txt")
        np.savetxt(fname_all, all_out,
                   header="r_Mpc  log10_gbar  log10_gobs_ztrue  log10_gobs_zphotoz",
                   comments="# ")
        print(f"\nAll bins combined: {fname_all}")

    print("\nDone.")


if __name__ == "__main__":
    main()
