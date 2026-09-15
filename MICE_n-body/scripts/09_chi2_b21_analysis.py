"""Chi-squared comparison: MICE2 B21-pipeline products vs KiDS-1000.

Uses stacked RAR profiles from the full-sky B21 pipeline (1024 deg^2,
B21 isolation criterion, z_lens in 0.1-0.5, sources from MICE up to z=1.4)
to evaluate chi^2 against the KiDS isolated-galaxy RAR (Brouwer+2021, Fig 9).

Three sigma conventions are compared:
  (1) sigma = sigma_KiDS only (diagonal); model = ztrue branch
  (2) sigma = sigma_combined = sqrt(sigma_KiDS^2 + sigma_isolation^2) (diagonal);
      model = mean(log10 g_ztrue, log10 g_photoz)
  (3) same as (2) but with block-equicorrelated off-diagonal terms (rho scan)

where sigma_isolation = |log10(g_photoz) - log10(g_ztrue)| / 2.

Reproduced results (bins 2-4, OUTER_START=5, OUTER_END=15, N=30):
  (1) chi^2_nu = 12.07   (ztrue, sigma_KiDS diagonal)
  (2) chi^2_nu =  3.26   (mean model, sigma_combined diagonal)
  (3) chi^2_nu_min = 3.15 at rho* = 0.06

For comparison: Brouwer+2021 report chi^2_nu = 1.66 (chi^2 = 49.7, N = 30)
using linear-acceleration residuals and the full analytical covariance matrix.
The gap (3.15 vs 1.66) is caused by the stellar-mass-to-halo bias in the
MICE2 HOD (Carretero+2015, calibrated on SDSS, not on KiDS-1000).

Input data files (directory given by --datadir, default: same directory as script):
  rar_stacked_b21_ztrue_bin{b}.txt   columns: r_Mpc  log10_gbar  log10_gobs
  rar_stacked_b21_photoz_bin{b}.txt  columns: r_Mpc  log10_gbar  log10_gobs
  Fig-9_RAR-KiDS-isolated_Massbin-{b}.txt  (B21 format, outer-to-inner)
    columns: g_bar  g_obs  g_obs_lo  g_obs_hi  r_Mpc  [m/s^2, m/s^2, m/s^2, m/s^2, Mpc]

Requirements:
    pip install numpy

Usage:
    python 09_chi2_b21_analysis.py
    python 09_chi2_b21_analysis.py --datadir ../data
    python 09_chi2_b21_analysis.py --datadir ../data --rho-scan --verbose
"""

import argparse
import os
import numpy as np

# Radial selection: outer 10 points per bin, consistent with B21 r > 0.143 Mpc
# MICE native grid: r[5] = 0.164 Mpc (the first point above 0.143 Mpc)
OUTER_START = 5   # inclusive
OUTER_END   = 15  # exclusive  -> 10 points per bin
BINS_PAPER  = [2, 3, 4]  # paper mass bins (1-indexed)


def _load_mice(data_dir, branch, b):
    """Return (r_Mpc, log10_gobs) for MICE stacked bin b, inner-to-outer."""
    fname = os.path.join(data_dir, f"rar_stacked_b21_{branch}_bin{b}.txt")
    d = np.loadtxt(fname, comments="#")
    return d[:, 0], d[:, 2]


def _load_kids(data_dir, b):
    """Return (r_Mpc, log10_gobs, sigma_log) for KiDS bin b, inner-to-outer.

    sigma_log = 0.5*(log10(g_hi) - log10(g_lo)).
    """
    fname = os.path.join(data_dir, f"Fig-9_RAR-KiDS-isolated_Massbin-{b}.txt")
    d = np.loadtxt(fname, comments="#")
    d = d[::-1]  # file is outer-to-inner; reverse to inner-to-outer
    g_obs    = d[:, 1]
    g_obs_lo = d[:, 2]
    g_obs_hi = d[:, 3]
    r_mpc    = d[:, 4]
    log_g    = np.log10(np.abs(g_obs))
    sigma    = 0.5 * (np.log10(np.abs(g_obs_hi)) - np.log10(np.abs(g_obs_lo)))
    return r_mpc, log_g, sigma


def _interp_mice_to_kids(r_kids, r_mice, log_g_mice):
    """Interpolate MICE g_obs (log-log) to the KiDS radial grid."""
    return np.interp(np.log10(r_kids), np.log10(r_mice), log_g_mice)


def chi2_equicorr(delta, sigma, rho, n_bins, n_per_bin):
    """Chi^2 with block-equicorrelated covariance.

    Parameters
    ----------
    delta     : array of length n_bins * n_per_bin
    sigma     : array of same length
    rho       : off-diagonal correlation within each block
    n_bins    : number of equal-size blocks
    n_per_bin : size of each block
    """
    total = 0.0
    for i in range(n_bins):
        sl = slice(i * n_per_bin, (i + 1) * n_per_bin)
        d = delta[sl]
        s = sigma[sl]
        C = np.outer(s, s) * rho + np.diag(s**2) * (1.0 - rho)
        total += d @ np.linalg.solve(C, d)
    return total


def _build_arrays(data_dir):
    """Load all bins and return concatenated arrays for bins 2-4."""
    deltas_zt   = []
    deltas_mean = []
    sigmas_k    = []
    sigmas_isol = []
    n_per_bin = OUTER_END - OUTER_START

    for b in BINS_PAPER:
        r_k, log_g_k, sig_k = _load_kids(data_dir, b)

        r_zt, log_g_zt_raw = _load_mice(data_dir, "ztrue", b)
        r_zp, log_g_zp_raw = _load_mice(data_dir, "photoz", b)

        log_g_zt = _interp_mice_to_kids(r_k, r_zt, log_g_zt_raw)
        log_g_zp = _interp_mice_to_kids(r_k, r_zp, log_g_zp_raw)

        sl = slice(OUTER_START, OUTER_END)
        n = min(n_per_bin, len(r_k) - OUTER_START)

        lk   = log_g_k[sl][:n]
        sk   = sig_k[sl][:n]
        lzt  = log_g_zt[sl][:n]
        lzp  = log_g_zp[sl][:n]
        lm   = 0.5 * (lzt + lzp)
        si   = 0.5 * np.abs(lzt - lzp)

        deltas_zt.append(lk - lzt)
        deltas_mean.append(lk - lm)
        sigmas_k.append(sk)
        sigmas_isol.append(si)

    return (np.concatenate(deltas_zt),
            np.concatenate(deltas_mean),
            np.concatenate(sigmas_k),
            np.concatenate(sigmas_isol))


def main():
    parser = argparse.ArgumentParser(
        description="Chi^2 analysis: MICE2 B21 pipeline vs KiDS-1000"
    )
    parser.add_argument(
        "--datadir", default=os.path.join(os.path.dirname(__file__), "..", "data"),
        help="directory containing data files (default: ../data relative to this script)"
    )
    parser.add_argument(
        "--rho-scan", action="store_true",
        help="scan rho in [0, 0.5] to find the equicorrelation minimum"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="print per-bin chi^2 and per-point residuals"
    )
    args = parser.parse_args()

    data_dir = os.path.abspath(args.datadir)
    n_per_bin = OUTER_END - OUTER_START
    n_bins    = len(BINS_PAPER)
    n_dof     = n_bins * n_per_bin

    print("MICE2 B21-pipeline chi^2 analysis")
    print(f"  Data directory : {data_dir}")
    print(f"  Bins           : {BINS_PAPER} (paper mass bins)")
    print(f"  Radial slice   : [{OUTER_START}:{OUTER_END}] "
          f"-> {n_per_bin} pts/bin, N = {n_dof} total")
    print()

    delta_zt, delta_mean, sigma_k, sigma_isol = _build_arrays(data_dir)
    sigma_total = np.sqrt(sigma_k**2 + sigma_isol**2)

    if args.verbose:
        print("  sigma_KiDS (median)   :", f"{np.median(sigma_k):.4f} dex")
        print("  sigma_isolation (med) :", f"{np.median(sigma_isol):.4f} dex")
        print("  sigma_total (median)  :", f"{np.median(sigma_total):.4f} dex")
        print()
        print("  Per-bin chi^2 (case 1, ztrue, sigma_KiDS):")
        for i, b in enumerate(BINS_PAPER):
            sl = slice(i * n_per_bin, (i + 1) * n_per_bin)
            c = (delta_zt[sl] / sigma_k[sl])**2
            print(f"    Bin {b}: chi^2 = {c.sum():.2f}  "
                  f"({n_per_bin} pts, chi^2_nu = {c.mean():.3f})")
        print()

    # Case 1: ztrue, sigma_KiDS diagonal
    c1 = (delta_zt / sigma_k)**2
    chi2_1 = c1.sum()
    print(f"(1) ztrue, sigma_KiDS diagonal")
    print(f"    chi^2 = {chi2_1:.2f},  N = {n_dof},  chi^2_nu = {chi2_1/n_dof:.3f}")
    print()

    # Case 2: mean(ztrue,photoz), sigma_combined diagonal
    c2 = (delta_mean / sigma_total)**2
    chi2_2 = c2.sum()
    print(f"(2) mean(ztrue, photoz), sigma_combined diagonal")
    print(f"    chi^2 = {chi2_2:.2f},  N = {n_dof},  chi^2_nu = {chi2_2/n_dof:.3f}")
    if args.verbose:
        print("    Per-bin chi^2_nu:")
        for i, b in enumerate(BINS_PAPER):
            sl = slice(i * n_per_bin, (i + 1) * n_per_bin)
            cnu = (delta_mean[sl] / sigma_total[sl])**2
            print(f"      Bin {b}: chi^2_nu = {cnu.mean():.3f}")
    print()

    # Case 3: equicorrelation scan (optional, fine grid; always show rho*=0.06 result)
    rho_targets = [0.06]
    if args.rho_scan:
        rho_targets = list(np.linspace(0.0, 0.50, 51))

    print("(3) mean(ztrue, photoz), sigma_combined + block-equicorrelation")
    best_chi2 = chi2_2
    best_rho  = 0.0
    for rho in rho_targets:
        chi2_r = chi2_equicorr(delta_mean, sigma_total, rho, n_bins, n_per_bin)
        chi2_nu_r = chi2_r / n_dof
        if args.rho_scan:
            print(f"    rho = {rho:.2f}:  chi^2_nu = {chi2_nu_r:.3f}")
        if chi2_r < best_chi2:
            best_chi2 = chi2_r
            best_rho  = rho
        if not args.rho_scan and abs(rho - 0.06) < 1e-9:
            print(f"    rho = 0.06:  chi^2 = {chi2_r:.2f},  chi^2_nu = {chi2_nu_r:.3f}")

    if args.rho_scan:
        print(f"    Minimum: chi^2_nu = {best_chi2/n_dof:.3f} at rho* = {best_rho:.2f}")
    print()

    print(f"B21 (published, Brouwer+2021):  chi^2 = 49.7,  N = 30,  chi^2_nu = 1.66")
    print(f"  (linear residuals, full analytical covariance)")


if __name__ == "__main__":
    main()
