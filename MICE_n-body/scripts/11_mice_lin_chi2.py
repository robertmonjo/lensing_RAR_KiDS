"""MICE2 reduced chi^2 against KiDS-1000 on the outer-radii subsets (Table 2).

The MICE prediction is the stacked lensing RAR of the isolated MICE2 lenses selected
with true redshifts (04_lensing_stack.py), interpolated in log r to the KiDS radii.
No parameter is fitted to KiDS (k = 0), so chi2_nu = chi2 / N.

Both statistics follow Sect. 2.4 of the paper: diagonal KiDS uncertainties and the
one-sided width on the side of the data point facing the prediction,
  LOG: sigma = log10 g84 - log10 g_obs  if g_pred >= g_obs, else log10 g_obs - log10 g16
  LIN: sigma = g84 - g_obs              if g_pred >= g_obs, else g_obs - g16

Subsets: outer radii r >= 0.164 Mpc (10 points per bin);
  N = 30 for mass bins 2-4 and N = 40 for bins 1-4.

Input files (--datadir, default: ../data next to this script):
  Fig-9_RAR-KiDS-isolated_Massbin-{b}.txt   cols: gbar gobs gobs_lo gobs_hi r_Mpc (outer to inner)
  rar_stacked_b21_ztrue_bin{b}.txt          cols: r_Mpc log10_gbar log10_gobs

Usage:
    python 11_mice_lin_chi2.py [--datadir ../data]
"""

import argparse
import os
import numpy as np
from scipy.interpolate import interp1d

OUTER_START = 5    # first KiDS radius of the outer subset (r = 0.164 Mpc)
OUTER_END = 15     # 10 outer-radii points per bin


def default_datadir():
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def load_kids(datadir, b):
    d = np.loadtxt(os.path.join(datadir, f"Fig-9_RAR-KiDS-isolated_Massbin-{b}.txt"),
                   comments="#")[::-1]                       # inner to outer
    return d[:, 4], d[:, 1], d[:, 2], d[:, 3]                # r, g_obs, g16, g84


def load_mice(datadir, b):
    d = np.loadtxt(os.path.join(datadir, f"rar_stacked_b21_ztrue_bin{b}.txt"), comments="#")
    return d[:, 0], 10.0 ** d[:, 2]                          # r_Mpc, g_obs [m/s^2]


def interp_mice(r_kids, r_mice, g_mice):
    f = interp1d(np.log10(r_mice), np.log10(g_mice), kind="linear", fill_value="extrapolate")
    return 10.0 ** f(np.log10(r_kids))


def z_log(g, lo, hi, pred):
    sig = np.where(pred >= g, np.log10(hi) - np.log10(g), np.log10(g) - np.log10(lo))
    return (np.log10(g) - np.log10(pred)) / sig


def z_lin(g, lo, hi, pred):
    sig = np.where(pred >= g, hi - g, g - lo)
    return (g - pred) / sig


def mice_chi2(bins, datadir=None, verbose=False):
    """Return (N, chi2_log, chi2_lin) of the MICE z_true stack over the given mass bins."""
    datadir = datadir or default_datadir()
    n = c_log = c_lin = 0
    for b in bins:
        r, g, lo, hi = load_kids(datadir, b)
        sl = slice(OUTER_START, OUTER_END)
        pred = interp_mice(r[sl], *load_mice(datadir, b))
        zl, zn = z_log(g[sl], lo[sl], hi[sl], pred), z_lin(g[sl], lo[sl], hi[sl], pred)
        c_log += zl @ zl; c_lin += zn @ zn; n += len(zl)
        if verbose:
            print(f"  bin {b}: chi2_log = {zl @ zl:7.2f}  chi2_lin = {zn @ zn:7.2f}")
    return n, c_log, c_lin


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--datadir", default=None)
    args = p.parse_args()
    for bins, label in [([2, 3, 4], "bins 2-4"), ([1, 2, 3, 4], "bins 1-4")]:
        print(f"MICE z_true stack, {label}:")
        n, cl, cn = mice_chi2(bins, args.datadir, verbose=True)
        print(f"  N = {n}: chi2_nu LOG = {cl / n:.2f}  LIN = {cn / n:.2f}\n")


if __name__ == "__main__":
    main()
