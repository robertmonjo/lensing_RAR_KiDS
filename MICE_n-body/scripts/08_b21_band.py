"""Construct the MICE isolation band of Brouwer et al. (2021, Sect. 5.3).

The band spans the stacked lensing RAR of two isolated-lens samples that differ
only in the redshifts used for the isolation criterion:
  - z_true  : true simulation redshifts                     (02 + 04, no suffix)
  - z_photo : z_true + N(0, 0.02 (1 + z)), emulating KiDS   (02 + 04, suffix _photoz)

At each radius present in both stacks the band takes the minimum and maximum of
the two log10 g_obs values. g_bar is the stacked baryonic acceleration of the
z_true sample (04_lensing_stack.py). Radii below --r_min are discarded.

Output: rar_band_b21_bin{b}.txt and rar_band_b21_allbins.txt with columns
        r_Mpc  log10_gbar  log10_gobs_lo  log10_gobs_hi

Usage:
    python 08_b21_band.py --indir_ztrue ../data/MICE2_b21/stacked_ztrue \
                          --indir_photoz ../data/MICE2_b21/stacked_photoz \
                          --outdir ../data/MICE2_b21/band
"""

import argparse
import os
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="Build the MICE isolation band")
    p.add_argument("--indir", default="../data/MICE2_isolated/",
                   help="directory holding both stacks (used when the two below are not given)")
    p.add_argument("--indir_ztrue", default=None)
    p.add_argument("--indir_photoz", default=None)
    p.add_argument("--suffix_ztrue", default="")
    p.add_argument("--suffix_zphotoz", default="")
    p.add_argument("--outdir", default="../data/MICE2_isolated/")
    p.add_argument("--r_min", type=float, default=0.05,
                   help="minimum radius kept in the band [Mpc] (default 0.05)")
    return p.parse_args()


def load_stacked(indir, b, suffix):
    """rar_stacked{suffix}_bin{b}.txt -> array (r_Mpc, log10_gbar, log10_gobs) or None."""
    fname = os.path.join(indir, f"rar_stacked{suffix}_bin{b}.txt")
    if not os.path.exists(fname):
        print(f"  {fname} not found")
        return None
    return np.loadtxt(fname, comments="#", ndmin=2)


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    indir_zt = args.indir_ztrue or args.indir
    indir_zp = args.indir_photoz or args.indir

    all_rows = []
    for b in range(1, 5):
        zt = load_stacked(indir_zt, b, args.suffix_ztrue)
        zp = load_stacked(indir_zp, b, args.suffix_zphotoz)
        if zt is None or zp is None:
            print(f"Bin {b}: missing stack, skipped")
            continue

        # Pair the two stacks radius by radius (both use the same radial grid).
        common = np.intersect1d(np.round(zt[:, 0], 8), np.round(zp[:, 0], 8))
        common = common[common >= args.r_min]
        it = [int(np.argmin(np.abs(zt[:, 0] - r))) for r in common]
        ip = [int(np.argmin(np.abs(zp[:, 0] - r))) for r in common]
        zt_c, zp_c = zt[it], zp[ip]

        lo = np.minimum(zt_c[:, 2], zp_c[:, 2])
        hi = np.maximum(zt_c[:, 2], zp_c[:, 2])
        out = np.column_stack([zt_c[:, 0], zt_c[:, 1], lo, hi])
        fname = os.path.join(args.outdir, f"rar_band_b21_bin{b}.txt")
        np.savetxt(fname, out, comments="# ",
                   header=("r_Mpc  log10_gbar  log10_gobs_lo  log10_gobs_hi\n"
                           f"(MICE isolation band: envelope of z_true and z_photo stacks, "
                           f"r >= {args.r_min} Mpc, bin {b})"))
        print(f"Bin {b}: {len(out)} radii -> {fname} (max width {(hi - lo).max():.3f} dex)")
        all_rows.append(out)

    if all_rows:
        fname = os.path.join(args.outdir, "rar_band_b21_allbins.txt")
        np.savetxt(fname, np.vstack(all_rows), comments="# ",
                   header="r_Mpc  log10_gbar  log10_gobs_lo  log10_gobs_hi")
        print(f"All bins: {fname}")


if __name__ == "__main__":
    main()
