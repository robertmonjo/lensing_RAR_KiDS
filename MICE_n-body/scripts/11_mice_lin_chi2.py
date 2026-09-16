"""MICE2 B21-pipeline chi2_nu in LOG and LIN space — Table 2 reference values.

Computes chi2_nu in both log10-space (LOG) and linear-space (LIN) for:
  - MICE reproduced (ztrue branch, diagonal sigma_KiDS):         N=30 and N=40
  - MICE best case  (mean ztrue+photoz, equicorr rho=0.065):     N=30 and N=40

These are the four MICE rows in Table 2.  The companion script
09_chi2_b21_analysis.py computes only LOG chi2 and a rho scan;
this script adds the LIN column and the N=40 (bins 1-4) variant.

Conventions:
  LOG: chi2_nu = sum[(log10 g_obs - log10 g_pred)^2 / sig_log^2] / N
       sig_log = 0.5 * (log10 g_hi - log10 g_lo)  [symmetric log sigma]
  LIN: chi2_nu = sum[(g_obs - g_pred)^2 / sig_lin^2] / N
       sig_lin = g_hi - g_obs  if g_pred < g_obs  (upward sigma)
               = g_obs - g_lo  otherwise           (downward sigma)
  Best case: g_pred = 10^[0.5*(log10 g_ztrue + log10 g_photoz)]
             sigma += isolation term in quadrature
             Equicorr off-diagonal block: rho = 0.065 per bin

Reproduced reference values (Table 2 hardcoded in 10_lin_chi2_table2.py):
  MICE reproduced N=30  LOG=12.07  LIN=7.09
  MICE reproduced N=40  LOG=16.42  LIN=7.61
  MICE best case  N=30  LOG= 3.15  LIN=1.85
  MICE best case  N=40  LOG= 2.66  LIN=1.65

Input files (--datadir, default: ../data relative to this script):
  Fig-9_RAR-KiDS-isolated_Massbin-{b}.txt   (outer-to-inner)
    cols: gbar  gobs  gobs_lo  gobs_hi  r_Mpc  [m/s^2, Mpc]
  rar_stacked_b21_ztrue_bin{b}.txt
    cols: r_Mpc  log10_gbar  log10_gobs
  rar_stacked_b21_photoz_bin{b}.txt
    cols: r_Mpc  log10_gbar  log10_gobs

Requirements:
    pip install numpy scipy

Usage:
    python 11_mice_lin_chi2.py
    python 11_mice_lin_chi2.py --datadir ../data
"""

import argparse
import os
import numpy as np
from scipy.interpolate import interp1d

OUTER_START = 5    # inclusive — first point above B21 cut r > 0.143 Mpc
OUTER_END   = 15   # exclusive  -> 10 outer-radii points per bin
RHO_BEST    = 0.065


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--datadir", default=None,
                   help="directory with input data files (default: ../data next to this script)")
    return p.parse_args()


def load_kids(datadir, b):
    fname = os.path.join(datadir, f"Fig-9_RAR-KiDS-isolated_Massbin-{b}.txt")
    d = np.loadtxt(fname, comments="#")
    d = d[::-1]              # flip outer->inner to inner->outer
    # cols: gbar, gobs, gobs_lo, gobs_hi, r_Mpc
    return d[:, 4], d[:, 1], d[:, 2], d[:, 3]   # r, obs, lo, hi


def load_mice(datadir, tag, b):
    fname = os.path.join(datadir, f"rar_stacked_b21_{tag}_bin{b}.txt")
    raw = np.loadtxt(fname, comments="#")
    return raw[:, 0], 10.0 ** raw[:, 2]          # r_Mpc, g [m/s^2]


def interp_mice(r_kids, r_mice, g_mice):
    itp = interp1d(np.log10(r_mice), np.log10(g_mice),
                   kind="linear", fill_value="extrapolate")
    return 10.0 ** itp(np.log10(r_kids))


def _z_log(go_k, go_lo, go_hi, g_pred):
    sig = 0.5 * (np.log10(go_hi) - np.log10(go_lo))
    return (np.log10(go_k) - np.log10(g_pred)) / sig


def _z_lin(go_k, go_lo, go_hi, g_pred):
    sig = np.where(g_pred < go_k, go_hi - go_k, go_k - go_lo)
    return (go_k - g_pred) / sig


def equicorr(z, rho):
    n = len(z)
    return z @ z / (1 - rho) - rho * z.sum() ** 2 / ((1 - rho) * (1 + (n - 1) * rho))


def compute_ztrue(datadir, bins, label):
    print(f"\n-- {label} (ztrue, diagonal) --")
    c_log = c_lin = n_tot = 0
    for b in bins:
        r_k, go_k, go_lo, go_hi = load_kids(datadir, b)
        r_m, g_m = load_mice(datadir, "ztrue", b)
        sl = slice(OUTER_START, OUTER_END)
        g_pred = interp_mice(r_k[sl], r_m, g_m)
        zl = _z_log(go_k[sl], go_lo[sl], go_hi[sl], g_pred)
        zn = _z_lin(go_k[sl], go_lo[sl], go_hi[sl], g_pred)
        c_log += zl @ zl;  c_lin += zn @ zn;  n_tot += len(r_k[sl])
        print(f"  bin {b}: chi2_log={zl@zl:.3f}  chi2_lin={zn@zn:.3f}")
    print(f"  => chi2_nu LOG={c_log/n_tot:.2f}  LIN={c_lin/n_tot:.2f}  (N={n_tot})")
    return n_tot, c_log, c_lin


def compute_best_case(datadir, bins, label):
    print(f"\n-- {label} (best case, equicorr rho={RHO_BEST}) --")
    ceq_log = ceq_lin = n_tot = 0
    for b in bins:
        r_k, go_k, go_lo, go_hi = load_kids(datadir, b)
        r_zt, g_zt = load_mice(datadir, "ztrue", b)
        r_zp, g_zp = load_mice(datadir, "photoz", b)
        sl = slice(OUTER_START, OUTER_END)
        g_zt_i = interp_mice(r_k[sl], r_zt, g_zt)
        g_zp_i = interp_mice(r_k[sl], r_zp, g_zp)

        log_zt = np.log10(g_zt_i);  log_zp = np.log10(g_zp_i)
        log_pred = 0.5 * (log_zt + log_zp)
        g_pred = 10.0 ** log_pred

        sig_log_tot = np.sqrt(
            (0.5 * (np.log10(go_hi[sl]) - np.log10(go_lo[sl]))) ** 2 +
            (0.5 * np.abs(log_zp - log_zt)) ** 2
        )
        sig_iso_lin = 0.5 * np.abs(g_zp_i - g_zt_i)
        sig_lin_up = np.sqrt((go_hi[sl] - go_k[sl]) ** 2 + sig_iso_lin ** 2)
        sig_lin_dn = np.sqrt((go_k[sl] - go_lo[sl]) ** 2 + sig_iso_lin ** 2)
        sig_lin_tot = np.where(g_pred < go_k[sl], sig_lin_up, sig_lin_dn)

        z_log = (np.log10(go_k[sl]) - log_pred) / sig_log_tot
        z_lin = (go_k[sl] - g_pred) / sig_lin_tot

        n_bin = len(r_k[sl]);  n_tot += n_bin
        eq_l = equicorr(z_log, RHO_BEST);  eq_n = equicorr(z_lin, RHO_BEST)
        ceq_log += eq_l;  ceq_lin += eq_n
        print(f"  bin {b}: log_eq={eq_l:.3f}  lin_eq={eq_n:.3f}")
    print(f"  => chi2_nu LOG={ceq_log/n_tot:.2f}  LIN={ceq_lin/n_tot:.2f}  (N={n_tot})")
    return n_tot, ceq_log, ceq_lin


def main():
    args = parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    datadir = args.datadir or os.path.join(here, "..", "data")

    W = "=" * 65
    print(W)
    print("MICE chi2_nu LOG + LIN — Table 2 reference values")
    print(f"datadir: {os.path.abspath(datadir)}")
    print(W)

    n30z, cl30z, cn30z = compute_ztrue(datadir,  [2, 3, 4], "N=30")
    n40z, cl40z, cn40z = compute_ztrue(datadir,  [1, 2, 3, 4], "N=40")
    n30b, cl30b, cn30b = compute_best_case(datadir, [2, 3, 4], "N=30")
    n40b, cl40b, cn40b = compute_best_case(datadir, [1, 2, 3, 4], "N=40")

    print()
    print(W)
    print("SUMMARY — Table 2 MICE rows")
    print(W)
    print(f"{'Row':<42} {'LOG':>6}  {'LIN':>6}  {'ref_log':>7}  {'ref_lin':>7}")
    print("-" * 65)
    ref = {
        ("ztrue", 30): (12.07, 7.09),
        ("ztrue", 40): (16.42, 7.61),
        ("best",  30): ( 3.15, 1.85),
        ("best",  40): ( 2.66, 1.65),
    }
    rows = [
        ("MICE reproduced N=30", n30z, cl30z, cn30z, "ztrue", 30),
        ("MICE reproduced N=40", n40z, cl40z, cn40z, "ztrue", 40),
        ("MICE best case  N=30", n30b, cl30b, cn30b, "best",  30),
        ("MICE best case  N=40", n40b, cl40b, cn40b, "best",  40),
    ]
    for name, n, cl, cn, key1, key2 in rows:
        rl, rn = ref[(key1, key2)]
        print(f"{name:<42} {cl/n:>6.2f}  {cn/n:>6.2f}  {rl:>7.2f}  {rn:>7.2f}")
    print(W)


if __name__ == "__main__":
    main()
