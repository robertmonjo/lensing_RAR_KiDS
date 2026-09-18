#!/usr/bin/env python3
"""
compare_fwd_models.py  --  Robert Monjo / Claude Code, 2026-09-18

Chi2 comparison of HMG, MOND and CDM with two lensing kernels:
  SIS  : g_obs = 4G DeltaSigma_SIS  (standard; what the paper uses)
  FWD  : g_fwd = 4G DeltaSigma_Abel (Abel projection of each model's rho_eff)

Tests whether the kernel bias changes the chi2_nu ranking across models.

Per-bin output:
  chi2_nu(SIS) and chi2_nu(FWD) for HMG (best-fit s), MOND (fixed a0), CDM (SHMR+xi0).

Note: the baryonic point-mass contribution M_bar/(pi R^2) to DeltaSigma is
omitted for all three models (consistent with forward_lensing_delta_sigma.py).
Its inclusion would systematically shift all FWD values upward at R < 0.1 Mpc,
but is < 2% at the signal-dominated scales (0.16-0.8 Mpc) and does not affect
the ranking.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from scipy.integrate import quad
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar
import hmg_model as h

# ── Physical constants ────────────────────────────────────────────────────────
G_SI     = h.G_SI          # m^3 kg^-1 s^-2
MSUN_KG  = h.MSUN_KG       # kg per M_sun
MPC_M    = h.MPC_M          # m per Mpc
PC_M     = 3.0857e16        # m per pc
MPC_TO_KPC = 1000.0
ESD2G    = 4.0 * G_SI * MSUN_KG / PC_M**2   # m/s^2 per M_sun/pc^2

# ── Data ─────────────────────────────────────────────────────────────────────
BINS  = h.BINS
MBAR  = h.MBAR
MSTAR = h.MSTAR

# Best-fit s values for HMG (SIS kernel, make_tables.py pipeline, N-1 chi2)
S_BIN_HMG = [0.2724, 0.4957, 0.5285, 0.4268]

# CDM: Moster+2013 SHMR halo masses
MH_MOSTER = [h.mhalo_from_mstar(ms) for ms in MSTAR]

# ── CDM global xi0 ───────────────────────────────────────────────────────────
def _fit_xi0_global():
    """Global xi0 fit with SHMR Mh fixed — same as make_tables.fit_xi0_global."""
    def chi2_total(xi0):
        return sum(
            h._chi2_arr(h.gobs_cdm(h._r_kpc(b), ms, mh, xi0), b)
            for b, ms, mh in zip(BINS, MBAR, MH_MOSTER)
        )
    res = minimize_scalar(chi2_total, bounds=(0.1, 30.0), method='bounded')
    return res.x

def _fit_mh_per_bin(bdata, ms, xi0):
    """Per-bin Mh fit (log scale) with xi0 fixed — same as make_tables.cdm_bin."""
    f = lambda lmh: h._chi2_arr(h.gobs_cdm(h._r_kpc(bdata), ms, 10**lmh, xi0), bdata)
    res = minimize_scalar(f, bounds=(10.0, 14.0), method='bounded')
    return 10**res.x   # best-fit Mh [M_sun]

print("Fitting CDM xi0 globally ... ", end="", flush=True)
XI0 = _fit_xi0_global()
print(f"xi0 = {XI0:.4f}")

print("Fitting CDM Mh per bin ... ", end="", flush=True)
MH_FIT = [_fit_mh_per_bin(b, ms, XI0) for b, ms in zip(BINS, MBAR)]
print("done  Mh/1e12 =", [f"{m/1e12:.2f}" for m in MH_FIT])

# ── Chi2 / (N-1), log-space asymmetric errors ─────────────────────────────────
def chi2_nu(pred, bdata):
    gobs   = bdata[:, 1]
    sig_up = np.log10(bdata[:, 3]) - np.log10(gobs)
    sig_dn = np.log10(gobs)        - np.log10(bdata[:, 2])
    sig    = np.where(pred >= gobs, sig_up, sig_dn)
    return float(np.sum(((np.log10(gobs) - np.log10(pred)) / sig)**2)) / (len(bdata) - 1)

# ── 3D acceleration functions ─────────────────────────────────────────────────
def g_hmg_fn(r_mpc, mbar, s):
    """HMG g_tot [m/s^2] at 3D radius r_mpc [Mpc]."""
    r_kpc = np.array([r_mpc * MPC_TO_KPC])
    return float(h.gobs_hmg(r_kpc, mbar, s)[0])

def g_mond_fn(r_mpc, mbar):
    """MOND g_tot [m/s^2] at 3D radius r_mpc [Mpc] (simple interp., fixed a0)."""
    gbar = h.gbar_si(r_mpc, mbar)   # G*M_bar/r^2 [m/s^2]
    return float(h.mond_gobs(gbar))

def g_cdm_fn(r_mpc, mbar, mh, xi0):
    """CDM g_tot [m/s^2] at 3D radius r_mpc [Mpc] (NFW + 2-halo)."""
    r_kpc = np.array([r_mpc * MPC_TO_KPC])
    return float(h.gobs_cdm(r_kpc, mbar, mh, xi0)[0])

# ── Abel projection machinery ─────────────────────────────────────────────────
def build_rho_eff(g_callable, r_min_mpc=3e-4, r_max_mpc=300.0, n_r=6000):
    """
    Numerical rho_eff(r) [M_sun/Mpc^3] from any g_callable(r_mpc) [m/s^2].
    Uses log-log cubic spline for smooth interpolation.
    """
    r_mpc = np.geomspace(r_min_mpc, r_max_mpc, n_r)
    g3d   = np.array([g_callable(r) for r in r_mpc])
    # M_eff(r) [M_sun] = r^2 g(r) / G
    M_eff = (r_mpc * MPC_M)**2 * g3d / G_SI / MSUN_KG
    dM_dr = np.gradient(M_eff, r_mpc)        # M_sun / Mpc
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

def compute_g_fwd(R_mpc_arr, g_callable, label="", R_lo=0.001):
    """
    Forward lensing g_fwd(R) [m/s^2] via Abel projection.
    R_lo = 0.001 Mpc (1 kpc): same truncation as forward_lensing_delta_sigma.py.
    """
    R_arr  = np.asarray(R_mpc_arr, float)
    rho_at = build_rho_eff(g_callable)
    R_hi   = R_arr.max() * 4.0
    R_grid = np.logspace(np.log10(R_lo), np.log10(R_hi), 100)
    print(f"    {label}: Abel ({len(R_grid)} pts)...", end=" ", flush=True)
    Sig_grid = np.array([sigma_at_R(R, rho_at) for R in R_grid]) / 1.0e12
    print("ESD...", end=" ", flush=True)
    Sig_spl = interp1d(np.log(R_grid), np.log(np.maximum(Sig_grid, 1e-40)),
                       kind='cubic', bounds_error=False, fill_value='extrapolate')
    def sig_at(R): return np.exp(Sig_spl(np.log(R)))
    dS = np.zeros(len(R_arr))
    for i, R in enumerate(R_arr):
        R_int    = np.linspace(R_lo, R, 1000)
        integral = np.trapezoid(R_int * sig_at(R_int), R_int)
        dS[i]    = 2.0 / R**2 * integral - sig_at(R)
    print("done")
    return ESD2G * dS

# ── Main comparison ───────────────────────────────────────────────────────────
print()
print("=" * 72)
print("  Bin  Model   chi2_nu(SIS)  chi2_nu(FWD)  FWD/SIS  Rank(SIS) Rank(FWD)")
print("=" * 72)

results = []
for i, (bdata, mbar, mstar, s, mh_shmr, mh) in enumerate(
        zip(BINS, MBAR, MSTAR, S_BIN_HMG, MH_MOSTER, MH_FIT), 1):
    R_arr  = bdata[:, 0]       # Mpc, all 15 points
    R_kpc  = R_arr * MPC_TO_KPC

    print(f"\nBin {i}: log M* = {np.log10(mstar):.2f}, Mbar = {mbar:.3e} M_sun, "
          f"s_HMG = {s}, Mh_fit = {mh:.3e} M_sun")

    # ── SIS chi2 (standard kernel) ──────────────────────────────────────────
    c_hmg_sis       = chi2_nu(h.gobs_hmg(R_kpc, mbar, s), bdata)
    c_mond_sis      = chi2_nu(h.mond_gobs(h.gbar_si(R_arr, mbar)), bdata)
    c_cdm_sis       = chi2_nu(h.gobs_cdm(R_kpc, mbar, mh, XI0), bdata)      # fitted Mh (Table 1)
    c_cdm_shmr_sis  = chi2_nu(h.gobs_cdm(R_kpc, mbar, mh_shmr, XI0), bdata) # SHMR Mh (§3)

    # ── FWD chi2 (Abel projection) ──────────────────────────────────────────
    g_fwd_hmg       = compute_g_fwd(R_arr, lambda r: g_hmg_fn(r, mbar, s),        "HMG     ")
    g_fwd_mond      = compute_g_fwd(R_arr, lambda r: g_mond_fn(r, mbar),           "MOND    ")
    g_fwd_cdm       = compute_g_fwd(R_arr, lambda r: g_cdm_fn(r, mbar, mh, XI0),      "CDM-fit ")
    g_fwd_cdm_shmr  = compute_g_fwd(R_arr, lambda r: g_cdm_fn(r, mbar, mh_shmr, XI0), "CDM-SHMR")

    c_hmg_fwd       = chi2_nu(g_fwd_hmg,       bdata)
    c_mond_fwd      = chi2_nu(g_fwd_mond,      bdata)
    c_cdm_fwd       = chi2_nu(g_fwd_cdm,       bdata)      # fitted Mh (Table 1)
    c_cdm_shmr_fwd  = chi2_nu(g_fwd_cdm_shmr,  bdata)      # SHMR Mh (§3 / A.4)

    # ── Ranking (SHMR CDM for global comparison, as in §3) ──────────────────
    models = {"HMG":  (c_hmg_sis,      c_hmg_fwd),
              "MOND": (c_mond_sis,     c_mond_fwd),
              "CDM":  (c_cdm_shmr_sis, c_cdm_shmr_fwd)}
    rank_sis = sorted(models, key=lambda m: models[m][0])
    rank_fwd = sorted(models, key=lambda m: models[m][1])

    for model, (cs, cf) in models.items():
        rs = rank_sis.index(model) + 1
        rf = rank_fwd.index(model) + 1
        print(f"  {i:4d}  {model:<5s}  {cs:11.3f}   {cf:11.3f}   {cf/cs:6.3f}   "
              f"  #{rs}         #{rf}")
    print(f"       CDM-fit SIS={c_cdm_sis:.3f}  FWD={c_cdm_fwd:.3f}  (per-bin fitted Mh, matches Table 1)")

    rank_changed = (rank_sis != rank_fwd)
    print(f"  -> Rank SIS: {' > '.join(rank_sis)}")
    print(f"  -> Rank FWD: {' > '.join(rank_fwd)}  {'*** CHANGED ***' if rank_changed else '(unchanged)'}")

    results.append({
        "bin": i, "logMstar": np.log10(mstar), "s": s,
        "HMG_SIS": c_hmg_sis, "MOND_SIS": c_mond_sis,
        "CDM_SIS": c_cdm_sis, "CDM_SHMR_SIS": c_cdm_shmr_sis,
        "HMG_FWD": c_hmg_fwd, "MOND_FWD": c_mond_fwd,
        "CDM_FWD": c_cdm_fwd, "CDM_SHMR_FWD": c_cdm_shmr_fwd,
        "rank_changed": rank_changed,
    })

# ── Global summary ────────────────────────────────────────────────────────────
print()
print("=" * 72)
print("GLOBAL (all 4 bins combined, chi2_total / (4*14) = chi2_total / 56)")
print("CDM uses Moster+2013 SHMR halo masses (consistent with paper §3 and A.4)")
N_tot = sum(len(b) for b in BINS) - len(BINS)   # 4*15 - 4 = 56

c_hmg_sis_g      = sum(h._chi2_arr(h.gobs_hmg(b[:,0]*MPC_TO_KPC, mb, s), b)
                       for b, mb, s in zip(BINS, MBAR, S_BIN_HMG)) / N_tot
c_mond_sis_g     = sum(h._chi2_arr(h.mond_gobs(h.gbar_si(b[:,0], mb)), b)
                       for b, mb in zip(BINS, MBAR)) / N_tot
c_cdm_sis_shmr_g = sum(h._chi2_arr(h.gobs_cdm(b[:,0]*MPC_TO_KPC, mb, mh, XI0), b)
                       for b, mb, mh in zip(BINS, MBAR, MH_MOSTER)) / N_tot
c_cdm_sis_fit_g  = sum(h._chi2_arr(h.gobs_cdm(b[:,0]*MPC_TO_KPC, mb, mh, XI0), b)
                       for b, mb, mh in zip(BINS, MBAR, MH_FIT)) / N_tot

print(f"  chi2_nu SIS (SHMR CDM): HMG = {c_hmg_sis_g:.3f}  MOND = {c_mond_sis_g:.3f}  CDM = {c_cdm_sis_shmr_g:.3f}")
print(f"  chi2_nu SIS (fit  CDM): CDM = {c_cdm_sis_fit_g:.3f}  (per-bin fitted Mh, Table 1 reference)")

c_hmg_fwd_g      = sum(r["HMG_FWD"]       * 14 for r in results) / N_tot
c_mond_fwd_g     = sum(r["MOND_FWD"]      * 14 for r in results) / N_tot
c_cdm_fwd_shmr_g = sum(r["CDM_SHMR_FWD"] * 14 for r in results) / N_tot
c_cdm_fwd_fit_g  = sum(r["CDM_FWD"]       * 14 for r in results) / N_tot

print(f"  chi2_nu FWD (SHMR CDM): HMG = {c_hmg_fwd_g:.3f}  MOND = {c_mond_fwd_g:.3f}  CDM = {c_cdm_fwd_shmr_g:.3f}")
print(f"  chi2_nu FWD (fit  CDM): CDM = {c_cdm_fwd_fit_g:.3f}  (per-bin fitted Mh, Table 1 reference)")

rank_sis_g = sorted(["HMG","MOND","CDM"],
                     key=lambda m: {"HMG":c_hmg_sis_g,"MOND":c_mond_sis_g,"CDM":c_cdm_sis_shmr_g}[m])
rank_fwd_g = sorted(["HMG","MOND","CDM"],
                     key=lambda m: {"HMG":c_hmg_fwd_g,"MOND":c_mond_fwd_g,"CDM":c_cdm_fwd_shmr_g}[m])
print(f"  Ranking SIS (SHMR): {' < '.join(rank_sis_g)}")
print(f"  Ranking FWD (SHMR): {' < '.join(rank_fwd_g)}  {'*** CHANGED ***' if rank_sis_g != rank_fwd_g else '(unchanged)'}")

# ── Save CSV ──────────────────────────────────────────────────────────────────
import csv
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUT, exist_ok=True)
outfile = os.path.join(OUT, "compare_fwd_models.csv")
with open(outfile, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
    w.writeheader()
    w.writerows(results)
print(f"\nResults saved to {outfile}")
