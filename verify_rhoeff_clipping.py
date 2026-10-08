#!/usr/bin/env python3
"""
verify_rhoeff_clipping.py  --  Robert Monjo / Claude Code, 2026-10-08

Quantify the effect of clipping negative rho_eff values on the Abel-projected
lensing signal g_fwd(R), as claimed in the response to the second referee
report (R4.2):

  "At R < 1 Mpc the difference is below 1% for all four mass bins;
   at R > 1 Mpc it reaches <=8% in the most extreme bin (bin 4)."

Method:
  1. Build rho_eff WITH clipping    (current production code, max(rho, 0))
  2. Build rho_eff WITHOUT clipping (raw numerical derivative, may go negative)
  3. Compute g_fwd at R = [0.16 ... 2.6] Mpc for both
  4. Report |g_fwd_clipped / g_fwd_unclipped - 1| per bin and R value

Note: unclipped rho_eff can be negative at large r (Hubble-dominated regime).
Linear-space interpolation is used for the unclipped case (log fails on negatives).

Requires: numpy, scipy  (same environment as forward_lensing_delta_sigma.py)
"""

import numpy as np
from scipy import integrate
from scipy.interpolate import interp1d

# ── Physical constants (identical to forward_lensing_delta_sigma.py) ──────────
G_SI       = 6.674e-11
MPC_M      = 3.0857e22
MSUN_KG    = 1.989e30
PC_M       = 3.0857e16
T0         = 14.11
C_KMS      = 299792.458
CODE_TO_SI = 3.24078e-14
G_CODE     = 4.30091727e-6
MPC_TO_KPC = 1000.0
SIN2_GU    = np.sin(np.pi / 3.0)**2
ESD2G      = 4.0 * G_SI * MSUN_KG / PC_M**2

def _cos_over_gamma(xi2):
    xi2   = np.asarray(xi2, float)
    delta = np.abs(1.0 - xi2) / (1.0 + xi2 + 1e-30)
    s2    = SIN2_GU + (1.0 - SIN2_GU) * np.clip(delta, 0.0, 1.0)
    gs    = np.arcsin(np.sqrt(np.clip(s2, 0.0, 1.0)))
    return np.cos(gs) / np.maximum(gs, 1e-10)

def gobs_hmg(r_kpc, mbar_msun, s):
    r   = np.asarray(r_kpc, float)
    v2n = G_CODE * mbar_msun / r
    vh2 = (r / T0)**2
    xi2 = 1.0 / s**3 + vh2 / (12.0 * v2n + 1e-60)
    a_n = v2n / r
    return np.sqrt(np.maximum(a_n * (a_n + 2.0 * C_KMS / T0 * _cos_over_gamma(xi2)), 0.0)) * CODE_TO_SI

# ── Baryonic masses ───────────────────────────────────────────────────────────
MSTAR = np.array([1.5e10, 3.2e10, 4.6e10, 8.9e10])
def f_cold(ms): return 10.0**(-0.69 * np.log10(ms) + 6.63)
MBAR = MSTAR * (1.0 + f_cold(MSTAR))
S_BIN_SIS = [0.2724, 0.4957, 0.5285, 0.4268]   # Table 1 best-fit s values

# ── rho_eff builder (WITH clipping, production code) ─────────────────────────
def build_rho_eff_clipped(mbar_msun, s, r_min_kpc=0.3, r_max_mpc=300.0, n_r=6000):
    r_kpc = np.logspace(np.log10(r_min_kpc), np.log10(r_max_mpc * MPC_TO_KPC), n_r)
    r_mpc = r_kpc / MPC_TO_KPC
    g3d   = gobs_hmg(r_kpc, mbar_msun, s)
    M_eff = (r_mpc * MPC_M)**2 * g3d / G_SI / MSUN_KG
    dM_dr = np.gradient(M_eff, r_mpc)
    rho   = np.maximum(dM_dr / (4.0 * np.pi * r_mpc**2), 0.0)  # CLIPPED
    good  = rho > 0
    spl   = interp1d(np.log(r_mpc[good]), np.log(rho[good]),
                     kind='cubic', bounds_error=False, fill_value='extrapolate')
    def rho_at(r_val):
        r = np.atleast_1d(np.asarray(r_val, float))
        out = np.exp(spl(np.log(np.maximum(r, 1e-6))))
        return float(out[0]) if out.size == 1 else out
    return rho_at

# ── rho_eff builder (WITHOUT clipping) ───────────────────────────────────────
def build_rho_eff_unclipped(mbar_msun, s, r_min_kpc=0.3, r_max_mpc=300.0, n_r=6000):
    """Linear-space interpolant: handles negative rho_eff at large r."""
    r_kpc = np.logspace(np.log10(r_min_kpc), np.log10(r_max_mpc * MPC_TO_KPC), n_r)
    r_mpc = r_kpc / MPC_TO_KPC
    g3d   = gobs_hmg(r_kpc, mbar_msun, s)
    M_eff = (r_mpc * MPC_M)**2 * g3d / G_SI / MSUN_KG
    dM_dr = np.gradient(M_eff, r_mpc)
    rho   = dM_dr / (4.0 * np.pi * r_mpc**2)  # NOT clipped
    spl   = interp1d(r_mpc, rho, kind='cubic', bounds_error=False,
                     fill_value=(rho[0], rho[-1]))
    def rho_at(r_val):
        r = np.atleast_1d(np.asarray(r_val, float))
        out = spl(r)
        return float(out[0]) if out.size == 1 else out
    return rho_at

# ── Abel projection (same as production, generic rho_at) ─────────────────────
def sigma_at_R(R_mpc, rho_at, z_max=200.0):
    val, _ = integrate.quad(lambda z: rho_at(np.sqrt(R_mpc**2 + z**2)),
                            0.0, z_max, limit=200, epsrel=1e-4)
    return 2.0 * val

def compute_g_fwd_generic(R_mpc_arr, mbar_msun, s, rho_at):
    R_arr  = np.asarray(R_mpc_arr, float)
    R_lo   = 0.001
    R_hi   = R_arr.max() * 4.0
    R_grid = np.logspace(np.log10(R_lo), np.log10(R_hi), 80)
    Sig_grid = np.array([sigma_at_R(R, rho_at) for R in R_grid]) / 1.0e12
    Sig_spl  = interp1d(np.log(R_grid), Sig_grid, kind='cubic',
                        bounds_error=False, fill_value='extrapolate')
    def sig_at(R): return float(Sig_spl(np.log(R)))
    dS = np.zeros(len(R_arr))
    for i, R in enumerate(R_arr):
        R_int    = np.linspace(R_lo, R, 500)
        integral = np.trapezoid(R_int * np.array([sig_at(r) for r in R_int]), R_int)
        dS[i]   = 2.0 / R**2 * integral - sig_at(R)
    return ESD2G * dS

# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    # R values to evaluate: cover both R<1 Mpc and R>1 Mpc
    R_test = np.array([0.164, 0.304, 0.561, 0.762,   # inner: signal-dominated
                       1.037, 1.409, 1.915, 2.604])   # outer: R > 1 Mpc

    print("=" * 80)
    print("Clipping effect on g_fwd: |g_fwd_clipped/g_fwd_unclipped - 1|  [%]")
    print("Claim in R4.2: <1% at R<1 Mpc; <=8% at R>1 Mpc (extreme bin 4)")
    print("=" * 80)

    max_inner = {}
    max_outer = {}

    for i, (mbar, s) in enumerate(zip(MBAR, S_BIN_SIS)):
        print(f"\nBin {i+1}: Mbar={mbar:.3e} Msun  s={s:.4f}")
        print(f"  Building rho_eff (clipped + unclipped) ...", flush=True)

        rho_c  = build_rho_eff_clipped(mbar, s)
        rho_u  = build_rho_eff_unclipped(mbar, s)

        # Check where rho_eff goes negative
        r_check = np.logspace(-2, 2.5, 500)   # 0.01 -- 316 Mpc
        g3d_chk = gobs_hmg(r_check * MPC_TO_KPC, mbar, s)
        M_chk   = (r_check * MPC_M)**2 * g3d_chk / G_SI / MSUN_KG
        dM_dr_chk = np.gradient(M_chk, r_check)
        rho_chk = dM_dr_chk / (4.0 * np.pi * r_check**2)
        neg_idx = np.where(rho_chk < 0)[0]
        if len(neg_idx) > 0:
            r_neg_start = r_check[neg_idx[0]]
            print(f"  rho_eff goes negative at r > {r_neg_start:.2f} Mpc")
        else:
            print(f"  rho_eff stays positive out to {r_check.max():.1f} Mpc")

        print(f"  Computing g_fwd at {len(R_test)} radii (clipped) ...", flush=True)
        g_c = compute_g_fwd_generic(R_test, mbar, s, rho_c)
        print(f"  Computing g_fwd at {len(R_test)} radii (unclipped) ...", flush=True)
        g_u = compute_g_fwd_generic(R_test, mbar, s, rho_u)

        diff_pct = np.abs(g_c / g_u - 1.0) * 100.0

        inner = R_test < 1.0
        outer = R_test >= 1.0
        max_inner[i+1] = diff_pct[inner].max()
        max_outer[i+1] = diff_pct[outer].max()

        print(f"  {'R_Mpc':>8}  {'g_clipped':>14}  {'g_unclipped':>14}  {'|diff|%':>8}")
        for R, gc, gu, dp in zip(R_test, g_c, g_u, diff_pct):
            tag = " *outer*" if R >= 1.0 else ""
            print(f"  {R:>8.3f}  {gc:>14.4e}  {gu:>14.4e}  {dp:>8.3f}%{tag}")

    print("\n" + "=" * 80)
    print("Summary: max |diff| % per bin")
    print(f"  {'Bin':>4}  {'R<1 Mpc (claim: <1%)':>22}  {'R>1 Mpc (claim: <=8%)':>22}")
    for i in range(1, 5):
        ok_i = "OK" if max_inner[i] < 1.0 else "EXCEEDS 1%"
        ok_o = "OK" if max_outer[i] <= 8.0 else f"EXCEEDS 8%"
        print(f"  {i:>4}  {max_inner[i]:>18.3f}%  {ok_i:>4}  "
              f"{max_outer[i]:>18.3f}%  {ok_o:>4}")
    print("=" * 80)
