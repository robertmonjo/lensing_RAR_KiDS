#!/usr/bin/env python3
"""
fwd_global_refit.py  --  Robert Monjo, 2026-10-07

Find the FWD-kernel global optimum for HMG (single s) and MOND (free a0).

Method: evaluate FWD chi2_global at 5 s values (3 a0 values for MOND),
fit a quadratic, report s_fwd and chi2_fwd_min.

Each evaluation runs 4 Abel projections in parallel (one per mass bin).
Estimated time: ~40-80 min locally (8 cores), ~20-40 min on Nieve (32 cores).

Checkpoints are saved after each (s, bin) evaluation so the script can be
interrupted and resumed (partially computed bins are not re-run).

Output: fwd_global_refit.json  (s grid, chi2 values, fitted optimum)
        fwd_global_refit.csv   (same, tabular)
"""

import sys, os, json, csv, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from scipy.integrate import quad, IntegrationWarning
from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings('ignore', category=IntegrationWarning)

import hmg_model as h

# ── Paths ─────────────────────────────────────────────────────────────────────
HERE    = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

CHECKPOINT = os.path.join(OUT_DIR, "fwd_global_refit_checkpoint.json")
OUT_JSON   = os.path.join(OUT_DIR, "fwd_global_refit.json")
OUT_CSV    = os.path.join(OUT_DIR, "fwd_global_refit.csv")

# ── Physical constants ────────────────────────────────────────────────────────
G_SI    = h.G_SI
MSUN_KG = h.MSUN_KG
MPC_M   = h.MPC_M
PC_M    = 3.0857e16
ESD2G   = 4.0 * G_SI * MSUN_KG / PC_M**2
MPC_TO_KPC = 1000.0

# ── Data ──────────────────────────────────────────────────────────────────────
BINS  = h.BINS
MBAR  = h.MBAR
MSTAR = h.MSTAR
N_BINS = len(BINS)

# ── Chi2 / (N-1), log-space asymmetric errors ─────────────────────────────────
def chi2_nu(pred, bdata):
    gobs   = bdata[:, 1]
    sig_up = np.log10(bdata[:, 3]) - np.log10(gobs)
    sig_dn = np.log10(gobs)        - np.log10(bdata[:, 2])
    sig    = np.where(pred >= gobs, sig_up, sig_dn)
    return float(np.sum(((np.log10(gobs) - np.log10(pred)) / sig)**2)) / (len(bdata) - 1)

# ── Abel projection ──────────────────────────────────────────────────────────
def build_rho_eff(g_callable, r_min_mpc=3e-4, r_max_mpc=300.0, n_r=6000):
    r_mpc = np.geomspace(r_min_mpc, r_max_mpc, n_r)
    g3d   = np.array([g_callable(r) for r in r_mpc])
    M_eff = (r_mpc * MPC_M)**2 * g3d / G_SI / MSUN_KG
    dM_dr = np.gradient(M_eff, r_mpc)
    rho   = np.maximum(dM_dr / (4.0 * np.pi * r_mpc**2), 0.0)
    good  = rho > 0
    lr    = np.log(r_mpc[good])
    lrho  = np.log(rho[good])
    spl   = interp1d(lr, lrho, kind='cubic', bounds_error=False,
                     fill_value=(lrho[0], lrho[-1]))
    def rho_at(r_val):
        r = np.atleast_1d(np.asarray(r_val, float))
        out = np.zeros(r.shape)
        pos = r > 0
        out[pos] = np.exp(spl(np.log(r[pos])))
        return float(out[0]) if out.size == 1 else out
    return rho_at

def sigma_at_R(R_mpc, rho_at, z_max=200.0):
    val, _ = quad(lambda z: rho_at(np.sqrt(R_mpc**2 + z**2)),
                  0.0, z_max, limit=500, epsrel=1e-5)
    return 2.0 * val

def compute_fwd_chi2_bin(bdata, g_callable, mbar_msun, R_lo=0.001):
    """FWD chi2 for one bin at given model parameters."""
    R_arr  = bdata[:, 0]
    rho_at = build_rho_eff(g_callable)
    R_hi   = R_arr.max() * 4.0
    R_grid = np.logspace(np.log10(R_lo), np.log10(R_hi), 100)
    Sig_grid = np.array([sigma_at_R(R, rho_at) for R in R_grid]) / 1.0e12
    # Filter out non-positive Sig values (rare numerical integration artefacts).
    good_sig = Sig_grid > 0
    if not np.all(good_sig):
        R_spl  = R_grid[good_sig]
        Sg_spl = Sig_grid[good_sig]
    else:
        R_spl, Sg_spl = R_grid, Sig_grid
    Sig_spl  = interp1d(np.log(R_spl),
                        np.log(Sg_spl),
                        kind='cubic', bounds_error=False, fill_value='extrapolate')
    def sig_at(R): return np.exp(Sig_spl(np.log(R)))
    dS = np.zeros(len(R_arr))
    for i, R in enumerate(R_arr):
        R_int    = np.linspace(R_lo, R, 1000)
        integral = np.trapezoid(R_int * sig_at(R_int), R_int)
        dS[i]    = 2.0 / R**2 * integral - sig_at(R)
    # Point-mass baryonic contribution (consistent with compare_fwd_models_v2.py)
    g_point = ESD2G * mbar_msun / (np.pi * R_arr**2 * 1.0e12)
    g_fwd = ESD2G * dS + g_point
    return chi2_nu(g_fwd, bdata)

# ── HMG g function ────────────────────────────────────────────────────────────
def g_hmg_fn(r_mpc, mbar, s):
    r_kpc = np.array([r_mpc * MPC_TO_KPC])
    return float(h.gobs_hmg(r_kpc, mbar, s)[0])

# ── MOND g function with free a0 ──────────────────────────────────────────────
def g_mond_fn(r_mpc, mbar, a0_factor):
    """MOND g_tot with a0 = a0_factor * a0_canonical."""
    gbar  = h.gbar_si(r_mpc, mbar)
    a0    = a0_factor * h.A0_SI
    x     = np.sqrt(gbar / a0)
    return float(gbar / (1.0 - np.exp(-x)))

# ── Checkpoint helpers ────────────────────────────────────────────────────────
def load_checkpoint():
    if os.path.isfile(CHECKPOINT):
        with open(CHECKPOINT) as f:
            return json.load(f)
    return {}

def save_checkpoint(data):
    with open(CHECKPOINT, "w") as f:
        json.dump(data, f, indent=2)

# ── Main grid evaluations ─────────────────────────────────────────────────────
# HMG: s grid centred on global SIS optimum (0.445)
S_GRID    = [0.38, 0.41, 0.445, 0.48, 0.52]

# MOND free-a0: a0 grid centred on global SIS optimum (1.51 * a0*)
A0_GRID   = [1.30, 1.42, 1.51, 1.61, 1.72]   # in units of a0*

print("=" * 70)
print("  FWD global refit — HMG (s) and MOND (free a0)")
print("=" * 70)
print(f"  HMG  s grid  : {S_GRID}")
print(f"  MOND a0 grid : {A0_GRID} (× a0*)")
print(f"  Bins         : {N_BINS} ({[len(b) for b in BINS]} pts each)")
print()

cp = load_checkpoint()

# ── HMG grid ─────────────────────────────────────────────────────────────────
print("── HMG FWD global chi2 vs s ──────────────────────────────────────────")
hmg_results = {}   # key: str(s), value: list of per-bin chi2

for s_val in S_GRID:
    key = f"HMG_s_{s_val:.4f}"
    per_bin = cp.get(key, [])
    if len(per_bin) < N_BINS:
        for ibin in range(len(per_bin), N_BINS):
            bdata = BINS[ibin]
            mbar  = MBAR[ibin]
            t0    = time.time()
            print(f"  s={s_val:.4f}  bin{ibin+1}  ...", end=" ", flush=True)
            c = compute_fwd_chi2_bin(
                bdata,
                lambda r, _m=mbar, _s=s_val: g_hmg_fn(r, _m, _s),
                mbar
            )
            per_bin.append(c)
            cp[key] = per_bin
            save_checkpoint(cp)
            dt = time.time() - t0
            print(f"chi2={c:.4f}  ({dt/60:.1f} min)")
    hmg_results[s_val] = per_bin

# Global chi2 per s value (pooled, chi2_total / sum(N_i - 1))
N_tot = sum(len(b) - 1 for b in BINS)
hmg_global = {}
for s_val in S_GRID:
    total = sum(hmg_results[s_val][i] * (len(BINS[i]) - 1) for i in range(N_BINS))
    hmg_global[s_val] = total / N_tot
    print(f"  s={s_val:.4f}  global chi2_nu={hmg_global[s_val]:.4f}")

# Quadratic fit to find optimum
s_arr  = np.array(S_GRID)
c_arr  = np.array([hmg_global[s] for s in S_GRID])
coeffs = np.polyfit(s_arr, c_arr, 2)
a, b_c, c_c = coeffs
s_fwd_opt   = -b_c / (2 * a)
chi2_fwd_min = np.polyval(coeffs, s_fwd_opt)

print()
print(f"  Quadratic fit: s_opt = {s_fwd_opt:.4f},  chi2_min = {chi2_fwd_min:.4f}")
print(f"  (SIS reference: s_SIS = 0.445, chi2_SIS_global = 2.32 [all 60 pts])")
print(f"  (FWD at per-bin SIS optima: chi2 = 1.51 [same normalisation as Table A.2])")

# ── MOND free-a0 grid ────────────────────────────────────────────────────────
print()
print("── MOND FWD global chi2 vs a0/a0* ────────────────────────────────────")
mond_results = {}

for a0f in A0_GRID:
    key = f"MOND_a0_{a0f:.3f}"
    per_bin = cp.get(key, [])
    if len(per_bin) < N_BINS:
        for ibin in range(len(per_bin), N_BINS):
            bdata = BINS[ibin]
            mbar  = MBAR[ibin]
            t0    = time.time()
            print(f"  a0={a0f:.2f}*a0*  bin{ibin+1}  ...", end=" ", flush=True)
            c = compute_fwd_chi2_bin(
                bdata,
                lambda r, _m=mbar, _a=a0f: g_mond_fn(r, _m, _a),
                mbar
            )
            per_bin.append(c)
            cp[key] = per_bin
            save_checkpoint(cp)
            dt = time.time() - t0
            print(f"chi2={c:.4f}  ({dt/60:.1f} min)")
    mond_results[a0f] = per_bin

mond_global = {}
for a0f in A0_GRID:
    total = sum(mond_results[a0f][i] * (len(BINS[i]) - 1) for i in range(N_BINS))
    mond_global[a0f] = total / N_tot
    print(f"  a0={a0f:.2f}*a0*  global chi2_nu={mond_global[a0f]:.4f}")

a0_arr  = np.array(A0_GRID)
cm_arr  = np.array([mond_global[a] for a in A0_GRID])
coeffs_m = np.polyfit(a0_arr, cm_arr, 2)
am, bm, cm = coeffs_m
a0_fwd_opt    = -bm / (2 * am)
chi2_mond_min = np.polyval(coeffs_m, a0_fwd_opt)

print()
print(f"  Quadratic fit: a0_opt = {a0_fwd_opt:.4f} × a0*,  chi2_min = {chi2_mond_min:.4f}")
print(f"  (SIS reference: a0_SIS = 1.51 × a0*, chi2_MOND_FWD at SIS opt = 6.66)")

# ── Summary ───────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("SUMMARY — changes relative to Table A.2 (SIS-kernel per-bin optima):")
print()
print(f"  HMG  SIS  (per-bin s, Table A.2): chi2 = 1.07")
print(f"  HMG  FWD  (per-bin s, Table A.2): chi2 = 1.51")
print(f"  HMG  FWD  (global s_fwd = {s_fwd_opt:.3f}): chi2 = {chi2_fwd_min:.4f}  [NEW]")
print()
print(f"  MOND SIS  (fixed a0,   Table A.2): chi2 = 6.74")
print(f"  MOND FWD  (fixed a0,   Table A.2): chi2 = 6.66")
print(f"  MOND FWD  (a0={a0_fwd_opt:.3f}×a0*, refitted): chi2 = {chi2_mond_min:.4f}  [NEW]")
print()

# ── Save outputs ─────────────────────────────────────────────────────────────
output = {
    "s_grid":           S_GRID,
    "hmg_global_chi2":  {str(s): v for s, v in hmg_global.items()},
    "hmg_s_fwd":        float(s_fwd_opt),
    "hmg_chi2_fwd_min": float(chi2_fwd_min),
    "a0_grid":          A0_GRID,
    "mond_global_chi2": {str(a): v for a, v in mond_global.items()},
    "mond_a0_fwd":      float(a0_fwd_opt),
    "mond_chi2_fwd_min":float(chi2_mond_min),
}
with open(OUT_JSON, "w") as f:
    json.dump(output, f, indent=2)

rows = []
for s_val in S_GRID:
    rows.append({
        "model": "HMG", "param": "s", "param_val": s_val,
        "global_chi2_fwd": hmg_global[s_val]
    })
for a0f in A0_GRID:
    rows.append({
        "model": "MOND", "param": "a0/a0*", "param_val": a0f,
        "global_chi2_fwd": mond_global[a0f]
    })
with open(OUT_CSV, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["model","param","param_val","global_chi2_fwd"])
    w.writeheader()
    w.writerows(rows)

print(f"Saved: {OUT_JSON}")
print(f"Saved: {OUT_CSV}")
