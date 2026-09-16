"""Linear-acceleration chi2_nu for Table 2 (tab:mice_comparison).

Computes chi2_nu in linear-acceleration space for all model rows of the MICE
comparison table, using the same KiDS outer-radii subsets as the log-space
metric (Sect. 4.3 of the paper).

Subsets
-------
N=30 : bins 2-4, outer radii (r >= 0.164 Mpc), 10 pts/bin
N=40 : bins 1-4, outer radii (r >= 0.164 Mpc), 10 pts/bin

Sigma_lin
---------
Asymmetric: sigma_up = g_p84 - g_obs when pred < g_obs,
            sigma_dn = g_obs - g_p16 when pred >= g_obs.

Degrees of freedom
------------------
k = 0 : fixed model, no parameters fitted to KiDS (MICE B21 reproduced)
k = 1 : one free parameter (HMG global scale s, MOND free a0)
k = 4 : one free parameter per mass bin (HMG per-bin)
chi2_nu = chi2 / (N - k)

Published chi2_nu (log-space) values for cross-check:
  HMG (s=0.486, N=30) LOG : 1.56
  MOND (1.73 a0*, N=30) LOG : 3.19
  HMG per-bin (N=40)   LOG : 0.93
  HMG (s=0.462, N=40)  LOG : 2.66
  MOND (1.58 a0*, N=40) LOG : 4.27

References
----------
Brouwer et al. 2021, A&A 650, A113
Monjo 2025 (this paper)
"""

import csv
import sys
import os
import numpy as np
from scipy.optimize import minimize_scalar

# Ensure hmg_model is importable when run from any working directory
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import hmg_model as h

A0_STAR = 1.20e-10   # m/s^2 — canonical MOND acceleration scale (Milgrom 1983)
R_OUTER_MPC = 0.164  # Mpc — outer-radii cut (r >= R_OUTER_MPC)


# ─── Data subsets ────────────────────────────────────────────────────────────

def _outer(bdata):
    return bdata[bdata[:, 0] >= R_OUTER_MPC]

BINS_OUT4 = [_outer(b) for b in h.BINS]       # bins 1-4, 4×10 = N=40
BINS_OUT3 = [_outer(b) for b in h.BINS[1:]]   # bins 2-4, 3×10 = N=30
MBAR_OUT4 = h.MBAR
MBAR_OUT3 = h.MBAR[1:]

N40 = sum(len(b) for b in BINS_OUT4)   # 40
N30 = sum(len(b) for b in BINS_OUT3)   # 30


# ─── Chi2 helpers ─────────────────────────────────────────────────────────────

def chi2_lin(pred, bdata):
    """Asymmetric linear chi2 sum."""
    gobs   = bdata[:, 1]
    sig_up = bdata[:, 3] - gobs   # g_p84 - g_obs
    sig_dn = gobs - bdata[:, 2]   # g_obs - g_p16
    sig    = np.where(pred < gobs, sig_up, sig_dn)
    return float(np.sum(((gobs - pred) / sig) ** 2))

def chi2_log(pred, bdata):
    """Log-space asymmetric chi2 (current paper metric)."""
    return h._chi2_arr(pred, bdata)

def hmg_pred(bdata, ms, s):
    return h.gobs_hmg(h._r_kpc(bdata), ms, s)

def mond_pred(bdata, ms, a0):
    """McGaugh (2016) MOND interpolation: g_tot = g_bar / (1 - exp(-sqrt(g_bar/a0)))."""
    r_m   = bdata[:, 0] * 1e3 * 3.086e19   # Mpc -> m
    G     = 6.674e-11
    M_sun = 1.989e30
    g_bar = G * ms * M_sun / r_m**2
    x     = np.sqrt(g_bar / a0)
    return g_bar / (1.0 - np.exp(-x))


# ─── N = 30 (bins 2-4, outer) ────────────────────────────────────────────────

print("=" * 65)
print(f"N = {N30} (bins 2-4, outer r >= {R_OUTER_MPC} Mpc)")
print("=" * 65)

# HMG global s=0.486 (log-space optimum), k=1
S_30 = 0.486
c_hmg30_log = sum(chi2_log(hmg_pred(b, ms, S_30), b) for b, ms in zip(BINS_OUT3, MBAR_OUT3))
c_hmg30_lin = sum(chi2_lin(hmg_pred(b, ms, S_30), b) for b, ms in zip(BINS_OUT3, MBAR_OUT3))
print(f"\nHMG s={S_30} (log-opt, k=1, dof={N30-1}):")
print(f"  chi2_nu LOG = {c_hmg30_log / (N30-1):.3f}   (paper: 1.56)")
print(f"  chi2_nu LIN = {c_hmg30_lin / (N30-1):.3f}   (Table 2 col)")

# MOND free a0 = 1.73 a0* (log-space optimum), k=1
A0_30 = 1.73 * A0_STAR
c_mond30_log = sum(chi2_log(mond_pred(b, ms, A0_30), b) for b, ms in zip(BINS_OUT3, MBAR_OUT3))
c_mond30_lin = sum(chi2_lin(mond_pred(b, ms, A0_30), b) for b, ms in zip(BINS_OUT3, MBAR_OUT3))
print(f"\nMOND a0 = 1.73 a0* (log-opt, k=1, dof={N30-1}):")
print(f"  chi2_nu LOG = {c_mond30_log / (N30-1):.3f}   (paper: 3.19)")
print(f"  chi2_nu LIN = {c_mond30_lin / (N30-1):.3f}   (Table 2 col)")
print(f"  [chi2 total LIN = {c_mond30_lin:.1f}, was incorrectly reported as {c_mond30_log:.1f}]")

# MICE B21 published (k=0, no KiDS-fitted parameters): chi2_nu = 49.7/30 = 1.66
print(f"\nMICE B21 published (k=0, dof={N30}): chi2_nu LIN = 1.66  [from Brouwer+2021]")

# MICE reproduced (ztrue, diagonal sigma_KiDS, k=0): chi2_nu LIN = 7.09
# Computed from Nieve canonical MICE2_b21 stacked_ztrue files; LOG cross-check = 12.07.
# (The earlier value 8.73 came from the B21-parcial variant, not the canonical ztrue.)
print(f"MICE reproduced (ztrue, diag, k=0, dof={N30}): chi2_nu LIN = 7.09  [Nieve stacked_ztrue]")


# ─── N = 40 (bins 1-4, outer) ────────────────────────────────────────────────

print()
print("=" * 65)
print(f"N = {N40} (bins 1-4, outer r >= {R_OUTER_MPC} Mpc)")
print("=" * 65)

# HMG global s=0.462 (log-space optimum), k=1
S_40 = 0.462
c_hmg40_log = sum(chi2_log(hmg_pred(b, ms, S_40), b) for b, ms in zip(BINS_OUT4, MBAR_OUT4))
c_hmg40_lin = sum(chi2_lin(hmg_pred(b, ms, S_40), b) for b, ms in zip(BINS_OUT4, MBAR_OUT4))
print(f"\nHMG s={S_40} (log-opt, k=1, dof={N40-1}):")
print(f"  chi2_nu LOG = {c_hmg40_log / (N40-1):.3f}   (paper: 2.66)")
print(f"  chi2_nu LIN = {c_hmg40_lin / (N40-1):.3f}   (Table 2 col)")
print(f"  [chi2 total LOG = {c_hmg40_log:.1f}, was incorrectly reported as lin]")

# HMG per-bin (one s per bin, log-opt), k=4
print(f"\nHMG per-bin (log-opt each bin, k=4, dof={N40-4}):")
c_log_pb = 0.0
c_lin_pb = 0.0
for i, (b, ms) in enumerate(zip(BINS_OUT4, MBAR_OUT4)):
    fi = lambda s: chi2_log(hmg_pred(b, ms, s), b)
    ri = minimize_scalar(fi, bounds=(0.1, 2.0), method="bounded")
    c_log_pb += ri.fun
    c_lin_pb += chi2_lin(hmg_pred(b, ms, ri.x), b)
    print(f"  Bin {i+1}: s={ri.x:.4f}  chi2_nu LOG={ri.fun/len(b):.3f}  LIN={chi2_lin(hmg_pred(b,ms,ri.x),b)/len(b):.3f}")
print(f"  Total: chi2_nu LOG = {c_log_pb/(N40-4):.3f}   (paper: 0.93)")
print(f"  Total: chi2_nu LIN = {c_lin_pb/(N40-4):.3f}   (Table 2 col)")
print(f"  [chi2 total LOG = {c_log_pb:.1f}, was incorrectly reported as lin]")

# MOND free a0 = 1.58 a0* (log-space optimum), k=1
A0_40 = 1.58 * A0_STAR
c_mond40_log = sum(chi2_log(mond_pred(b, ms, A0_40), b) for b, ms in zip(BINS_OUT4, MBAR_OUT4))
c_mond40_lin = sum(chi2_lin(mond_pred(b, ms, A0_40), b) for b, ms in zip(BINS_OUT4, MBAR_OUT4))
print(f"\nMOND a0 = 1.58 a0* (log-opt, k=1, dof={N40-1}):")
print(f"  chi2_nu LOG = {c_mond40_log / (N40-1):.3f}   (paper: 4.27)")
print(f"  chi2_nu LIN = {c_mond40_lin / (N40-1):.3f}   (Table 2 col)")
print(f"  [chi2 total LOG = {c_mond40_log:.1f}, was incorrectly reported as lin]")


# ─── Summary table ─────────────────────────────────────────────────────────────

print()
print("=" * 65)
print("TABLE 2 VALUES (chi2_nu, correct)")
print("=" * 65)
print(f"{'Model':<45} {'N':>3} {'LIN':>6} {'LOG':>6}")
print("-" * 65)
print(f"{'HMG (s=0.486, fitted, k=1)':<45} {N30:>3} {c_hmg30_lin/(N30-1):>6.2f} {c_hmg30_log/(N30-1):>6.2f}")
print(f"{'MICE B21, Brouwer+2021 (k=0)':<45} {N30:>3} {'1.66':>6} {'--':>6}")
print(f"{'MICE repro (ztrue, diag, k=0)':<45} {N30:>3} {'7.09':>6} {'12.07':>6}")
print(f"{'MICE best case (equicorr rho=0.065, k=0)':<45} {N30:>3} {'1.85':>6} {'3.15':>6}")
print(f"{'MOND (1.73 a0*, k=1)':<45} {N30:>3} {c_mond30_lin/(N30-1):>6.2f} {c_mond30_log/(N30-1):>6.2f}")
print("-" * 65)
print(f"{'HMG per-bin (k=4)':<45} {N40:>3} {c_lin_pb/(N40-4):>6.2f} {c_log_pb/(N40-4):>6.2f}")
print(f"{'HMG (s=0.462, fitted, k=1)':<45} {N40:>3} {c_hmg40_lin/(N40-1):>6.2f} {c_hmg40_log/(N40-1):>6.2f}")
print(f"{'MICE repro (ztrue, diag, k=0)':<45} {N40:>3} {'7.61':>6} {'16.42':>6}")
print(f"{'MICE best case (equicorr rho=0.065, k=0)':<45} {N40:>3} {'1.65':>6} {'2.66':>6}")
print(f"{'MOND (1.58 a0*, k=1)':<45} {N40:>3} {c_mond40_lin/(N40-1):>6.2f} {c_mond40_log/(N40-1):>6.2f}")

# ─── Write CSV (tab_mice_comparison.csv replaces the old 5-row version) ─────
_OUT = os.path.join(_HERE, "outputs")
os.makedirs(_OUT, exist_ok=True)

_ROWS = [
    # N=30 outer-radii subset (bins 2-4)
    {"model": f"HMG (s={S_30}, fitted, k=1)",             "N": N30, "k": 1,
     "chi2_nu_log": round(c_hmg30_log / (N30-1), 2),
     "chi2_nu_lin": round(c_hmg30_lin / (N30-1), 2),    "source": "this work"},
    {"model": "MICE LambdaCDM B21 (k=0)",                 "N": N30, "k": 0,
     "chi2_nu_log": "--", "chi2_nu_lin": 1.66,           "source": "Brouwer+2021"},
    {"model": "MICE reproduced (ztrue, diagonal, k=0)",   "N": N30, "k": 0,
     "chi2_nu_log": 12.07, "chi2_nu_lin": 7.09,          "source": "MICE_n-body pipeline (Nieve)"},
    {"model": "MICE best case (equicorr rho=0.065, k=0)", "N": N30, "k": 0,
     "chi2_nu_log": 3.15,  "chi2_nu_lin": 1.85,          "source": "MICE_n-body pipeline (Nieve)"},
    {"model": f"MOND (1.73 a0*, k=1)",                    "N": N30, "k": 1,
     "chi2_nu_log": round(c_mond30_log / (N30-1), 2),
     "chi2_nu_lin": round(c_mond30_lin / (N30-1), 2),   "source": "this work"},
    # N=40 outer-radii subset (bins 1-4)
    {"model": "HMG per-bin (k=4)",                        "N": N40, "k": 4,
     "chi2_nu_log": round(c_log_pb / (N40-4), 2),
     "chi2_nu_lin": round(c_lin_pb / (N40-4), 2),        "source": "this work"},
    {"model": f"HMG (s={S_40}, fitted, k=1)",             "N": N40, "k": 1,
     "chi2_nu_log": round(c_hmg40_log / (N40-1), 2),
     "chi2_nu_lin": round(c_hmg40_lin / (N40-1), 2),    "source": "this work"},
    {"model": "MICE reproduced (ztrue, diagonal, k=0)",   "N": N40, "k": 0,
     "chi2_nu_log": 16.42, "chi2_nu_lin": 7.61,          "source": "MICE_n-body pipeline (Nieve)"},
    {"model": "MICE best case (equicorr rho=0.065, k=0)", "N": N40, "k": 0,
     "chi2_nu_log": 2.66,  "chi2_nu_lin": 1.65,          "source": "MICE_n-body pipeline (Nieve)"},
    {"model": f"MOND (1.58 a0*, k=1)",                    "N": N40, "k": 1,
     "chi2_nu_log": round(c_mond40_log / (N40-1), 2),
     "chi2_nu_lin": round(c_mond40_lin / (N40-1), 2),   "source": "this work"},
]

_FIELDS = ["model", "N", "k", "chi2_nu_log", "chi2_nu_lin", "source"]
_csv_path = os.path.join(_OUT, "tab_mice_comparison.csv")
with open(_csv_path, "w", newline="") as _fh:
    _w = csv.DictWriter(_fh, fieldnames=_FIELDS)
    _w.writeheader()
    _w.writerows(_ROWS)
print(f"\nWrote {_csv_path}")
