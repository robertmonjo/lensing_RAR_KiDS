#!/usr/bin/env python3
"""
fwd_global_refit_v2.py  --  Robert Monjo, 2026-10-08

FWD-kernel global refit for HMG (s), MOND (free a0) and CDM (free xi0).

Changes vs v1:
  - nu = N_total - k_global = 60 - 1 = 59  (not 56 = 4*(N-1))
  - HMG and MOND per-bin chi2 are read from the existing v1 checkpoint
    (fwd_global_refit_checkpoint.json); only CDM is newly computed.
  - CDM: single free parameter xi0 (global SHMR normalization) with
    SHMR halo masses fixed (Moster+2013).

Output:
  outputs/fwd_global_refit_v2.json
  outputs/fwd_global_refit_v2.csv
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
HERE          = os.path.dirname(os.path.abspath(__file__))
OUT_DIR       = os.path.join(HERE, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

CHECKPOINT_V1 = os.path.join(OUT_DIR, "fwd_global_refit_checkpoint.json")
CHECKPOINT_V2 = os.path.join(OUT_DIR, "fwd_global_refit_v2_checkpoint.json")
OUT_JSON      = os.path.join(OUT_DIR, "fwd_global_refit_v2.json")
OUT_CSV       = os.path.join(OUT_DIR, "fwd_global_refit_v2.csv")

# ── Physical constants ────────────────────────────────────────────────────────
G_SI       = h.G_SI
MSUN_KG    = h.MSUN_KG
PC_M       = 3.0857e16
ESD2G      = 4.0 * G_SI * MSUN_KG / PC_M**2
MPC_TO_KPC = 1000.0

# ── Data ──────────────────────────────────────────────────────────────────────
BINS   = h.BINS
MBAR   = h.MBAR
MSTAR  = h.MSTAR
N_BINS = len(BINS)

# ── dof: nu = N_total - k_global (one global parameter per model) ─────────────
N_total  = sum(len(b) for b in BINS)  # = 60
K_GLOBAL = 1
N_DOF    = N_total - K_GLOBAL         # = 59

print(f"nu = {N_total} - {K_GLOBAL} = {N_DOF}  (one global free parameter)")

# ── Chi2 helpers ──────────────────────────────────────────────────────────────
def chi2_nu_bin(pred, bdata):
    """chi2/(N-1) for one bin — same formula as v1 for checkpoint compatibility."""
    gobs   = bdata[:, 1]
    sig_up = np.log10(bdata[:, 3]) - np.log10(gobs)
    sig_dn = np.log10(gobs)        - np.log10(bdata[:, 2])
    sig    = np.where(pred >= gobs, sig_up, sig_dn)
    return float(np.sum(((np.log10(gobs) - np.log10(pred)) / sig)**2)) / (len(bdata) - 1)

def global_chi2_nu(per_bin_chi2nu):
    """Pool per-bin chi2_nu -> global chi2_nu with nu=N_DOF=59."""
    total_chi2 = sum(c * (len(BINS[i]) - 1) for i, c in enumerate(per_bin_chi2nu))
    return total_chi2 / N_DOF

# ── Abel projection ───────────────────────────────────────────────────────────
def build_rho_eff(g_callable, r_min_mpc=3e-4, r_max_mpc=300.0, n_r=6000):
    r_mpc = np.geomspace(r_min_mpc, r_max_mpc, n_r)
    g3d   = np.array([g_callable(r) for r in r_mpc])
    M_eff = (r_mpc * h.MPC_M)**2 * g3d / G_SI / MSUN_KG
    dM_dr = np.gradient(M_eff, r_mpc)
    rho   = np.maximum(dM_dr / (4.0 * np.pi * r_mpc**2), 0.0)
    good  = rho > 0
    lr    = np.log(r_mpc[good])
    lrho  = np.log(rho[good])
    spl   = interp1d(lr, lrho, kind='cubic', bounds_error=False,
                     fill_value=(lrho[0], lrho[-1]))
    def rho_at(r_val):
        r   = np.atleast_1d(np.asarray(r_val, float))
        out = np.zeros(r.shape)
        pos = r > 0
        out[pos] = np.exp(spl(np.log(r[pos])))
        return float(out[0]) if out.size == 1 else out
    return rho_at

def sigma_at_R(R_mpc, rho_at, z_max=200.0):
    val, _ = quad(lambda z: rho_at(np.sqrt(R_mpc**2 + z**2)),
                  0.0, z_max, limit=500, epsrel=1e-5)
    return 2.0 * val

def compute_fwd_chi2nu_bin(bdata, g_callable, mbar_msun, R_lo=0.001):
    R_arr    = bdata[:, 0]
    rho_at   = build_rho_eff(g_callable)
    R_hi     = R_arr.max() * 4.0
    R_grid   = np.logspace(np.log10(R_lo), np.log10(R_hi), 100)
    Sig_grid = np.array([sigma_at_R(R, rho_at) for R in R_grid]) / 1.0e12
    good_sig = Sig_grid > 0
    R_spl    = R_grid[good_sig]
    Sg_spl   = Sig_grid[good_sig]
    Sig_spl  = interp1d(np.log(R_spl), np.log(Sg_spl),
                        kind='cubic', bounds_error=False, fill_value='extrapolate')
    def sig_at(R):
        return np.exp(Sig_spl(np.log(R)))
    dS = np.zeros(len(R_arr))
    for i, R in enumerate(R_arr):
        R_int    = np.linspace(R_lo, R, 1000)
        try:
            integral = np.trapezoid(R_int * sig_at(R_int), R_int)
        except AttributeError:
            integral = np.trapz(R_int * sig_at(R_int), R_int)
        dS[i] = 2.0 / R**2 * integral - sig_at(R)
    g_point = ESD2G * mbar_msun / (np.pi * R_arr**2 * 1.0e12)
    g_fwd   = ESD2G * dS + g_point
    return chi2_nu_bin(g_fwd, bdata)

# ── CDM g function ────────────────────────────────────────────────────────────
def g_cdm_fn(r_mpc, mbar, mh, xi0):
    r_kpc = np.array([r_mpc * MPC_TO_KPC])
    return float(h.gobs_cdm(r_kpc, mbar, mh, xi0)[0])

# ── SHMR halo masses ──────────────────────────────────────────────────────────
MH_MOSTER = [h.mhalo_from_mstar(ms) for ms in MSTAR]
print(f"SHMR Mh/1e12: {[f'{m/1e12:.2f}' for m in MH_MOSTER]}")

# ── Checkpoint helpers ────────────────────────────────────────────────────────
def load_cp(path):
    return json.load(open(path)) if os.path.isfile(path) else {}

def save_cp(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

# =============================================================================
# 1. HMG and MOND: reuse v1 checkpoints, recompute global chi2_nu with nu=59
# =============================================================================
cp_v1   = load_cp(CHECKPOINT_V1)
S_GRID  = [0.38, 0.41, 0.445, 0.48, 0.52]
A0_GRID = [1.30, 1.42, 1.51, 1.61, 1.72]

print("\n── HMG (v1 checkpoints, nu=59) ─────────────────────────────────────────")
hmg_global = {}
for s_val in S_GRID:
    key     = f"HMG_s_{s_val:.4f}"
    per_bin = cp_v1.get(key, [])
    if len(per_bin) < N_BINS:
        print(f"  MISSING: {key} ({len(per_bin)}/{N_BINS} bins)")
        continue
    g_nu = global_chi2_nu(per_bin)
    hmg_global[s_val] = g_nu
    old_nu = sum(c * (len(BINS[i]) - 1) for i, c in enumerate(per_bin)) / 56
    print(f"  s={s_val:.4f}  chi2_nu={g_nu:.4f}  (old nu=56: {old_nu:.4f})")

s_arr    = np.array(sorted(hmg_global.keys()))
c_arr    = np.array([hmg_global[s] for s in s_arr])
coeffs   = np.polyfit(s_arr, c_arr, 2)
s_fwd    = -coeffs[1] / (2 * coeffs[0])
chi2_hmg = np.polyval(coeffs, s_fwd)
print(f"\n  -> s_fwd = {s_fwd:.4f},  chi2_nu_min = {chi2_hmg:.4f}")

print("\n── MOND (v1 checkpoints, nu=59) ────────────────────────────────────────")
mond_global = {}
for a0f in A0_GRID:
    key     = f"MOND_a0_{a0f:.3f}"
    per_bin = cp_v1.get(key, [])
    if len(per_bin) < N_BINS:
        print(f"  MISSING: {key} ({len(per_bin)}/{N_BINS} bins)")
        continue
    g_nu = global_chi2_nu(per_bin)
    mond_global[a0f] = g_nu
    print(f"  a0={a0f:.2f}*a0*  chi2_nu={g_nu:.4f}")

a0_arr    = np.array(sorted(mond_global.keys()))
cm_arr    = np.array([mond_global[a] for a in a0_arr])
coeffs_m  = np.polyfit(a0_arr, cm_arr, 2)
a0_fwd    = -coeffs_m[1] / (2 * coeffs_m[0])
chi2_mond = np.polyval(coeffs_m, a0_fwd)
print(f"\n  -> a0_fwd = {a0_fwd:.4f} x a0*,  chi2_nu_min = {chi2_mond:.4f}")

# =============================================================================
# 2. CDM FWD-opt: xi0 grid with SHMR Mh fixed
# =============================================================================
XI0_GRID = [10.0, 12.5, 14.8, 17.5, 21.0]

print("\n── CDM FWD-opt (xi0 grid, SHMR Mh fixed, nu=59) ───────────────────────")
cp_v2       = load_cp(CHECKPOINT_V2)
cdm_results = {}

for xi0 in XI0_GRID:
    key     = f"CDM_xi0_{xi0:.2f}"
    per_bin = cp_v2.get(key, [])
    if len(per_bin) < N_BINS:
        for ibin in range(len(per_bin), N_BINS):
            bdata = BINS[ibin]
            mbar  = MBAR[ibin]
            mh    = MH_MOSTER[ibin]
            t0    = time.time()
            print(f"  xi0={xi0:.1f}  bin{ibin+1}  ...", end=" ", flush=True)
            c = compute_fwd_chi2nu_bin(
                bdata,
                lambda r, _m=mbar, _mh=mh, _x=xi0: g_cdm_fn(r, _m, _mh, _x),
                mbar
            )
            per_bin.append(c)
            cp_v2[key] = per_bin
            save_cp(cp_v2, CHECKPOINT_V2)
            dt = time.time() - t0
            print(f"chi2_nu={c:.4f}  ({dt/60:.1f} min)")
    cdm_results[xi0] = per_bin

cdm_global = {}
for xi0 in XI0_GRID:
    pb = cdm_results.get(xi0, [])
    if len(pb) == N_BINS:
        g_nu = global_chi2_nu(pb)
        cdm_global[xi0] = g_nu
        print(f"  xi0={xi0:.1f}  chi2_nu={g_nu:.4f}")

xi0_arr  = np.array(sorted(cdm_global.keys()))
cx_arr   = np.array([cdm_global[x] for x in xi0_arr])
coeffs_c = np.polyfit(xi0_arr, cx_arr, 2)
xi0_fwd  = -coeffs_c[1] / (2 * coeffs_c[0])
chi2_cdm = np.polyval(coeffs_c, xi0_fwd)
print(f"\n  -> xi0_fwd = {xi0_fwd:.4f},  chi2_nu_min = {chi2_cdm:.4f}")

# =============================================================================
# 3. Summary
# =============================================================================
print("\n" + "=" * 70)
print(f"SUMMARY (nu={N_DOF}):")
print(f"  HMG  FWD-opt  s    = {s_fwd:.3f}         chi2_nu = {chi2_hmg:.4f}")
print(f"  MOND FWD-opt  a0   = {a0_fwd:.3f} x a0*  chi2_nu = {chi2_mond:.4f}")
print(f"  CDM  FWD-opt  xi0  = {xi0_fwd:.3f}        chi2_nu = {chi2_cdm:.4f}")
print(f"  Ranking: {chi2_hmg:.4f} < {chi2_mond:.4f} < {chi2_cdm:.4f}  "
      f"({'OK: HMG < MOND < CDM' if chi2_hmg < chi2_mond < chi2_cdm else 'UNEXPECTED ORDER'})")

# ── Save ──────────────────────────────────────────────────────────────────────
out = {
    "nu_dof": N_DOF,
    "hmg":  {"s_fwd":   float(s_fwd),   "chi2_nu": float(chi2_hmg),
             "grid": {str(s): v for s, v in hmg_global.items()}},
    "mond": {"a0_fwd":  float(a0_fwd),  "chi2_nu": float(chi2_mond),
             "grid": {str(a): v for a, v in mond_global.items()}},
    "cdm":  {"xi0_fwd": float(xi0_fwd), "chi2_nu": float(chi2_cdm),
             "grid": {str(x): v for x, v in cdm_global.items()}},
}
with open(OUT_JSON, "w") as f:
    json.dump(out, f, indent=2)

rows = []
for s_val in sorted(hmg_global):
    rows.append({"model": "HMG",  "param": "s",      "val": s_val, "chi2_nu_fwd": hmg_global[s_val]})
for a0f in sorted(mond_global):
    rows.append({"model": "MOND", "param": "a0/a0*", "val": a0f,   "chi2_nu_fwd": mond_global[a0f]})
for xi0 in sorted(cdm_global):
    rows.append({"model": "CDM",  "param": "xi0",    "val": xi0,   "chi2_nu_fwd": cdm_global[xi0]})
with open(OUT_CSV, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["model", "param", "val", "chi2_nu_fwd"])
    w.writeheader()
    w.writerows(rows)

print(f"\nSaved: {OUT_JSON}")
print(f"Saved: {OUT_CSV}")
