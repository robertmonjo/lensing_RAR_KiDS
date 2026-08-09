"""Chi-squared comparison: MICE stacked mean vs KiDS-1000 data.

Reproduces the chi^2_nu = 1.66 reported in Brouwer+2021 for the MICE
Lambda-CDM prediction compared to the KiDS isolated-galaxy RAR.

Comparison subset (following B21):
  Mass bins 2, 3, 4  (code indices 1, 2, 3; paper bins 2-4)
  Outer 10 radial points: rows 4..13 (0-indexed, inner->outer ordering)
  i.e. r > 0.164 Mpc
  Total: N = 3 * 10 = 30 data points

Chi^2 formula (log space):
  chi^2 = sum [ (log10(g_obs_KiDS) - log10(g_obs_MICE)) / sigma_i ]^2

where sigma_i is derived from the KiDS 1-sigma error bars.

Requirements:
    pip install numpy astropy

Usage:
    python 06_chi2_comparison.py \
        --kidsdir ../../data/brouwer2021_rar/ \
        --micesdir ../data/MICE2_isolated/
"""

import argparse
import os
import sys
import numpy as np

# Outer 10 points: rows 4..13 in inner->outer ordering
# (row 0 = innermost, row 14 = outermost)
OUTER_START = 4   # inclusive
OUTER_END   = 14  # exclusive  (10 points: rows 4,5,...,13)

# Bins to include (1-indexed paper numbering: bins 2, 3, 4)
BINS_PAPER = [2, 3, 4]


def load_kids(data_dir, b):
    """Load KiDS B21 data for mass bin b (1-indexed paper numbering).

    Returns (r, g_bar, g_obs, g_obs_lo, g_obs_hi) in inner->outer order [m/s^2].
    """
    fname = os.path.join(data_dir, f"Fig-9_RAR-KiDS-isolated_Massbin-{b}.txt")
    if not os.path.exists(fname):
        raise FileNotFoundError(f"KiDS file not found: {fname}")
    d = np.loadtxt(fname, comments="#")
    d = d[::-1]  # outer->inner to inner->outer
    r_mpc = d[:, 4] if d.shape[1] > 4 else np.full(len(d), np.nan)
    return r_mpc, d[:, 0], d[:, 1], d[:, 2], d[:, 3]


def load_mice_stacked(mice_dir, b):
    """Load MICE stacked g_obs for mass bin b.

    Expects file: rar_stacked_binN.txt with columns [r_Mpc, log10_gbar, log10_gobs].
    Returns g_obs in m/s^2 (same length as KiDS radial bins).
    """
    fname = os.path.join(mice_dir, f"rar_stacked_bin{b}.txt")
    if not os.path.exists(fname):
        return None
    d = np.loadtxt(fname, comments="#")
    log_go = d[:, 2]
    return 10.0 ** log_go


def sigma_log(g_obs, g_obs_lo, g_obs_hi):
    """
    Convert KiDS absolute error bars to 1-sigma in log10 space.
    sigma_log = 0.5 * (log10(g_obs_hi) - log10(g_obs_lo))
    """
    return 0.5 * (np.log10(np.abs(g_obs_hi)) - np.log10(np.abs(g_obs_lo)))


def chi2_comparison(kids_dir, mice_dir, verbose=True):
    """
    Compute chi^2 between MICE stacked mean and KiDS data.

    Returns chi2_total, n_dof, chi2_nu
    """
    chi2_total = 0.0
    n_dof = 0

    rows = []

    for b in BINS_PAPER:
        r_mpc, g_bar_k, g_obs_k, g_obs_lo_k, g_obs_hi_k = load_kids(kids_dir, b)
        g_obs_m = load_mice_stacked(mice_dir, b)

        if g_obs_m is None:
            print(f"  Bin {b}: MICE stacked file not found, skipping.")
            continue

        # Outer 10 points
        sl = slice(OUTER_START, OUTER_END)
        go_k  = g_obs_k[sl]
        go_lo = g_obs_lo_k[sl]
        go_hi = g_obs_hi_k[sl]
        go_m  = g_obs_m[sl]
        r_sl  = r_mpc[sl]

        sig = sigma_log(go_k, go_lo, go_hi)
        log_diff = np.log10(np.abs(go_k)) - np.log10(np.abs(go_m))
        chi2_bin = (log_diff / sig) ** 2

        chi2_total += chi2_bin.sum()
        n_dof += len(chi2_bin)

        if verbose:
            print(f"\n  Bin {b} (outer 10 pts, r = {r_sl[0]:.3f} .. {r_sl[-1]:.3f} Mpc):")
            print(f"  {'r[Mpc]':>8} {'go_KiDS':>12} {'go_MICE':>12} "
                  f"{'sigma':>8} {'chi2_i':>8}")
            for i in range(len(r_sl)):
                print(f"  {r_sl[i]:8.3f} {np.log10(abs(go_k[i])):12.4f} "
                      f"{np.log10(abs(go_m[i])):12.4f} "
                      f"{sig[i]:8.4f} {chi2_bin[i]:8.3f}")
            print(f"  Bin {b} chi^2 = {chi2_bin.sum():.3f}  (10 pts)")

        for i in range(len(r_sl)):
            rows.append((b, r_sl[i], go_k[i], go_m[i], sig[i], chi2_bin[i]))

    chi2_nu = chi2_total / n_dof if n_dof > 0 else np.nan

    if verbose:
        print(f"\n{'='*50}")
        print(f"Total chi^2 = {chi2_total:.3f}  ({n_dof} d.o.f.)")
        print(f"chi^2_nu    = {chi2_nu:.4f}")
        print(f"Expected (B21): chi^2 = 49.7, chi^2_nu = 1.66")
        rel_diff = abs(chi2_nu - 1.66) / 1.66 * 100
        print(f"Relative difference from B21: {rel_diff:.1f}%")
        if rel_diff < 10:
            print("PASS: chi^2_nu within 10% of B21 value.")
        else:
            print("FAIL: chi^2_nu differs by more than 10% from B21.")
            print("  Possible causes:")
            print("  - Isolation criterion differs from B21")
            print("  - ESD conversion factor (check factor of 4)")
            print("  - g_bar cold gas fraction (check Baldry+2012 relation)")
            print("  - Radial bin centres don't match B21 files")

    return chi2_total, n_dof, chi2_nu, rows


def save_comparison_table(rows, outdir):
    out = os.path.join(outdir, "chi2_comparison_table.txt")
    header = ("bin  r_Mpc  log10_go_KiDS  log10_go_MICE  "
              "sigma_log  chi2_i")
    data = np.array([[r[0], r[1],
                      np.log10(abs(r[2])), np.log10(abs(r[3])),
                      r[4], r[5]] for r in rows])
    np.savetxt(out, data, header=header, comments="# ",
               fmt=["%d", "%.4f", "%.6f", "%.6f", "%.6f", "%.6f"])
    print(f"\nSaved comparison table: {out}")


def parse_args():
    p = argparse.ArgumentParser(description="Chi^2: MICE stacked vs KiDS")
    p.add_argument("--kidsdir",
                   default="../../data/brouwer2021_rar/")
    p.add_argument("--micesdir", default="../data/MICE2_isolated/")
    p.add_argument("--outdir", default="../data/MICE2_isolated/")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    print("Chi^2 comparison: MICE stacked mean vs KiDS-1000 (Brouwer+2021)")
    print(f"  KiDS data: {args.kidsdir}")
    print(f"  MICE data: {args.micesdir}")
    print(f"  Bins: {BINS_PAPER}, outer {OUTER_END - OUTER_START} radial pts each")
    print()

    try:
        chi2, n_dof, chi2_nu, rows = chi2_comparison(args.kidsdir, args.micesdir)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    if rows:
        save_comparison_table(rows, args.outdir)


if __name__ == "__main__":
    main()
