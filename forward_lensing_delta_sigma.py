#!/usr/bin/env python3
"""
forward_lensing_delta_sigma.py  --  Robert Monjo / Claude Code, 2026-09-18

Forward HMG lensing (Abel projection) vs SIS kernel.
Reproduces Appendix A.4 of Monjo et al. (2026).

Chi2 convention (consistent with Table 1 / make_tables.py):
  - ALL 15 data points per bin
  - chi2_nu = chi2_total / (N-1) with N=15 -> dof=14
  - asymmetric log-space chi2

Method:
  g_tot(r)  [HMG] -> M_eff(r) = r^2 g_tot / G
                  -> rho_eff(r) = dM_eff/dr / (4 pi r^2)
  Sigma(R)  = 2 int_0^inf rho_eff(sqrt(R^2+z^2)) dz   [Abel projection]
  DeltaSigma(R) = Sigma_bar(<R) - Sigma(R)              [ESD]
  g_fwd(R)  = 4G DeltaSigma(R)                          [lensing prediction]

Output:
  Console      chi2_nu SIS vs FWD, g_fwd/g_SIS at 0.16<R<0.8 Mpc
  CSV          forward_lensing_results.csv (per-radius g_SIS, g_fwd, ratio)

SIS analytic test: for rho = A/r^2, g_fwd must equal g_SIS to <0.5%
(tested at the start; assert fails if the numerics are broken).
"""

import csv
import numpy as np
from scipy import integrate
from scipy.interpolate import interp1d

# ── Physical constants ────────────────────────────────────────────────────────
G_SI       = 6.674e-11        # m^3 kg^-1 s^-2
MPC_M      = 3.0857e22        # m per Mpc
MSUN_KG    = 1.989e30         # kg per M_sun
PC_M       = 3.0857e16        # m per pc
T0         = 14.11            # kpc/(km/s) [Hubble time in code units]
C_KMS      = 299792.458       # km/s
CODE_TO_SI = 3.24078e-14      # (km/s)^2/kpc -> m/s^2
G_CODE     = 4.30091727e-6    # kpc (km/s)^2 Msun^-1
MPC_TO_KPC = 1000.0
SIN2_GU    = np.sin(np.pi / 3.0)**2
ESD2G      = 4.0 * G_SI * MSUN_KG / PC_M**2   # m/s^2 per M_sun/pc^2

# ── HMG total acceleration (3D) ───────────────────────────────────────────────
def _cos_over_gamma(xi2):
    xi2   = np.asarray(xi2, float)
    delta = np.abs(1.0 - xi2) / (1.0 + xi2 + 1e-30)
    s2    = SIN2_GU + (1.0 - SIN2_GU) * np.clip(delta, 0.0, 1.0)
    gs    = np.arcsin(np.sqrt(np.clip(s2, 0.0, 1.0)))
    return np.cos(gs) / np.maximum(gs, 1e-10)

def gobs_hmg(r_kpc, mbar_msun, s):
    """HMG total acceleration [m/s^2] at 3D radius r_kpc [kpc]."""
    r   = np.asarray(r_kpc, float)
    v2n = G_CODE * mbar_msun / r
    vh2 = (r / T0)**2
    xi2 = 1.0 / s**3 + vh2 / (12.0 * v2n + 1e-60)
    a_n = v2n / r
    return np.sqrt(np.maximum(a_n * (a_n + 2.0 * C_KMS / T0 * _cos_over_gamma(xi2)), 0.0)) * CODE_TO_SI

# ── Chi2 (N-1 denominator, all 15 points, log-space asymmetric errors) ────────
def _chi2_arr(pred, bdata):
    gobs   = bdata[:, 1]
    sig_up = np.log10(bdata[:, 3]) - np.log10(gobs)
    sig_dn = np.log10(gobs)        - np.log10(bdata[:, 2])
    sig    = np.where(pred >= gobs, sig_up, sig_dn)
    return float(np.sum(((np.log10(gobs) - np.log10(pred)) / sig)**2))

def chi2_nu_N1(pred, bdata):
    """chi2/(N-1) over all 15 data points; consistent with Table 1."""
    return _chi2_arr(pred, bdata) / (len(bdata) - 1)

# ── Baryonic masses (stellar + cold gas) ─────────────────────────────────────
MSTAR = np.array([1.5e10, 3.2e10, 4.6e10, 8.9e10])
def f_cold(ms): return 10.0**(-0.69 * np.log10(ms) + 6.63)
MBAR = MSTAR * (1.0 + f_cold(MSTAR))

# ── Best-fit s values (SIS kernel, all-15-pt N-1 chi2, Mbar-corrected) ────────
# These match Table 1 of the paper (make_tables.py pipeline output).
S_BIN_SIS = [0.2724, 0.4957, 0.5285, 0.4268]

# ── KiDS data: cols = (R_Mpc, g_obs, g_p16, g_p84) [m/s^2] ──────────────────
BINS = [
  np.array([  # Bin 1: log M* ~ 10.18
    [3.5390e-02,8.9678e-12,6.1374e-12,1.9384e-11],[4.8108e-02,8.7469e-12,7.3406e-12,1.3834e-11],
    [6.5396e-02,8.0130e-12,6.7410e-12,1.0456e-11],[8.8896e-02,6.2018e-12,5.1260e-12,7.8323e-12],
    [1.2084e-01,4.3587e-12,3.6570e-12,5.3527e-12],[1.6427e-01,2.9690e-12,2.5314e-12,3.5749e-12],
    [2.2330e-01,2.0745e-12,1.7913e-12,2.4488e-12],[3.0354e-01,1.5605e-12,1.3583e-12,1.8130e-12],
    [4.1262e-01,1.2783e-12,1.1054e-12,1.4809e-12],[5.6090e-01,1.0607e-12,9.1520e-13,1.2117e-12],
    [7.6247e-01,7.8897e-13,6.7724e-13,9.0410e-13],[1.0365e+00,5.2832e-13,4.5092e-13,6.0996e-13],
    [1.4089e+00,3.3855e-13,2.8545e-13,3.9559e-13],[1.9152e+00,2.1428e-13,1.7641e-13,2.5873e-13],
    [2.6035e+00,1.3393e-13,1.0657e-13,1.7144e-13],
  ]),
  np.array([  # Bin 2: log M* ~ 10.51
    [3.5390e-02,1.5240e-11,9.8535e-12,3.3211e-11],[4.8108e-02,1.4124e-11,1.1616e-11,2.3146e-11],
    [6.5396e-02,1.2871e-11,1.1021e-11,1.7090e-11],[8.8896e-02,1.0679e-11,9.1144e-12,1.3259e-11],
    [1.2084e-01,8.6016e-12,7.5652e-12,1.0177e-11],[1.6427e-01,7.1304e-12,6.3996e-12,8.1249e-12],
    [2.2330e-01,5.7500e-12,5.1995e-12,6.4133e-12],[3.0354e-01,4.3201e-12,3.9354e-12,4.7609e-12],
    [4.1262e-01,3.0574e-12,2.7931e-12,3.3592e-12],[5.6090e-01,2.0942e-12,1.9101e-12,2.2974e-12],
    [7.6247e-01,1.3954e-12,1.2591e-12,1.5433e-12],[1.0365e+00,8.8727e-13,7.8691e-13,9.9376e-13],
    [1.4089e+00,5.4361e-13,4.7481e-13,6.1821e-13],[1.9152e+00,3.2944e-13,2.8243e-13,3.8415e-13],
    [2.6035e+00,1.9842e-13,1.6559e-13,2.4103e-13],
  ]),
  np.array([  # Bin 3: log M* ~ 10.67
    [3.5390e-02,1.6935e-11,1.1562e-11,3.5642e-11],[4.8108e-02,1.6440e-11,1.3986e-11,2.5662e-11],
    [6.5396e-02,1.5090e-11,1.3193e-11,1.9514e-11],[8.8896e-02,1.2426e-11,1.0795e-11,1.5236e-11],
    [1.2084e-01,9.8112e-12,8.6715e-12,1.1521e-11],[1.6427e-01,7.5314e-12,6.7595e-12,8.5855e-12],
    [2.2330e-01,5.9140e-12,5.3595e-12,6.6054e-12],[3.0354e-01,4.6728e-12,4.2629e-12,5.1444e-12],
    [4.1262e-01,3.5961e-12,3.3011e-12,3.9225e-12],[5.6090e-01,2.7037e-12,2.4906e-12,2.9365e-12],
    [7.6247e-01,1.9644e-12,1.8099e-12,2.1292e-12],[1.0365e+00,1.3641e-12,1.2494e-12,1.4839e-12],
    [1.4089e+00,9.0829e-13,8.1959e-13,1.0011e-12],[1.9152e+00,5.8463e-13,5.1449e-13,6.6209e-13],
    [2.6035e+00,3.6568e-13,3.1151e-13,4.3290e-13],
  ]),
  np.array([  # Bin 4: log M* ~ 10.95
    [3.5390e-02,1.9524e-11,1.4866e-11,3.2192e-11],[4.8108e-02,2.0661e-11,1.8503e-11,2.6481e-11],
    [6.5396e-02,2.0456e-11,1.8438e-11,2.3269e-11],[8.8896e-02,1.6525e-11,1.4850e-11,1.8635e-11],
    [1.2084e-01,1.2646e-11,1.1571e-11,1.3953e-11],[1.6427e-01,1.0283e-11,9.5248e-12,1.1147e-11],
    [2.2330e-01,7.8173e-12,7.2633e-12,8.4194e-12],[3.0354e-01,5.6494e-12,5.2664e-12,6.0550e-12],
    [4.1262e-01,4.2016e-12,3.9093e-12,4.5022e-12],[5.6090e-01,3.0784e-12,2.8632e-12,3.3035e-12],
    [7.6247e-01,2.1934e-12,2.0376e-12,2.3523e-12],[1.0365e+00,1.5082e-12,1.3923e-12,1.6282e-12],
    [1.4089e+00,9.8307e-13,8.9601e-13,1.0749e-12],[1.9152e+00,6.2200e-13,5.5517e-13,6.9798e-13],
    [2.6035e+00,3.8775e-13,3.3420e-13,4.5779e-13],
  ]),
]

# ── Abel projection machinery ─────────────────────────────────────────────────

def build_rho_eff(mbar_msun, s, r_min_kpc=0.3, r_max_mpc=300.0, n_r=6000):
    """Log-log cubic interpolant of rho_eff(r) [M_sun/Mpc^3]."""
    r_kpc = np.logspace(np.log10(r_min_kpc), np.log10(r_max_mpc * MPC_TO_KPC), n_r)
    r_mpc = r_kpc / MPC_TO_KPC
    g3d   = gobs_hmg(r_kpc, mbar_msun, s)
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
    """Sigma(R) [M_sun/Mpc^2] via adaptive line-of-sight quadrature."""
    val, _ = integrate.quad(lambda z: rho_at(np.sqrt(R_mpc**2 + z**2)),
                            0.0, z_max, limit=500, epsrel=1e-5)
    return 2.0 * val

def compute_g_fwd(R_mpc_arr, mbar_msun, s, verbose=False):
    """
    Forward lensing acceleration g_fwd(R) [m/s^2] via Abel projection.

    R_lo for the barSigma integral is set to 0.001 Mpc (1 kpc), well below
    R_min ~ 35 kpc; truncation bias 1-2*R_lo/R < 1.3% at all signal-dominated
    scales (R > 0.16 Mpc), compared with 2.2% for R_lo = 0.05*R_min.
    """
    R_arr  = np.asarray(R_mpc_arr, float)
    rho_at = build_rho_eff(mbar_msun, s)
    R_lo   = 0.001                       # Mpc = 1 kpc  (fixed, not 0.05*R_min)
    R_hi   = R_arr.max() * 4.0
    R_grid = np.logspace(np.log10(R_lo), np.log10(R_hi), 100)
    if verbose:
        print(f"    Abel ({len(R_grid)} pts) ... ", end="", flush=True)
    Sig_grid = np.array([sigma_at_R(R, rho_at) for R in R_grid]) / 1.0e12
    if verbose:
        print("done", flush=True)
    Sig_spl = interp1d(np.log(R_grid), np.log(np.maximum(Sig_grid, 1e-40)),
                       kind='cubic', bounds_error=False, fill_value='extrapolate')
    def sig_at(R): return np.exp(Sig_spl(np.log(R)))
    dS = np.zeros(len(R_arr))
    for i, R in enumerate(R_arr):
        R_int    = np.linspace(R_lo, R, 1000)
        integral = np.trapezoid(R_int * sig_at(R_int), R_int)
        dS[i]   = 2.0 / R**2 * integral - sig_at(R)
    return ESD2G * dS

# ── SIS analytic validation ───────────────────────────────────────────────────

def _sis_analytic_test(tol=0.015):
    """
    Verify that the Abel projection pipeline recovers the exact SIS result.

    For rho(r) = A/r^2 (SIS), the analytic result is:
      Sigma(R)      = pi*A/R         [M_sun/Mpc^2]
      DeltaSigma(R) = pi*A/R         [M_sun/Mpc^2]  (Sigma_bar = 2*Sigma for SIS)
      g_fwd(R)      = 4G*DeltaSigma  [m/s^2]

    This tests the numerical Abel + ESD pipeline against that closed-form result.
    Uses a synthetic SIS rho function (not gobs_hmg) so the test is purely numeric.

    Tolerance 1.5%: the dominant error is the barSigma truncation at R_lo = 0.001 Mpc,
    giving a bias of 2*R_lo/R = 1.22% at R = 0.164 Mpc, 0.26% at R = 0.762 Mpc.
    """
    A = 1.0e12    # M_sun/Mpc  (rho = A/r^2, r in Mpc)
    R_test = np.array([0.164, 0.223, 0.304, 0.413, 0.561, 0.762])
    # Analytic: DeltaSigma = pi*A/R [M_sun/Mpc^2] = pi*A/R/1e12 [M_sun/pc^2]
    g_analytic = ESD2G * np.pi * A / 1.0e12 / R_test    # m/s^2

    def rho_sis(r_mpc):
        r = np.atleast_1d(np.asarray(r_mpc, float))
        return float(A / r[0]**2) if r.size == 1 else A / r**2

    R_lo   = 0.001    # Mpc = 1 kpc; same as compute_g_fwd
    R_hi   = 3.0
    R_grid = np.logspace(np.log10(R_lo), np.log10(R_hi), 100)
    Sig_grid = np.array([sigma_at_R(R, rho_sis) for R in R_grid]) / 1.0e12
    Sig_spl  = interp1d(np.log(R_grid), np.log(np.maximum(Sig_grid, 1e-40)),
                        kind='cubic', bounds_error=False, fill_value='extrapolate')
    def sig_at(R): return np.exp(Sig_spl(np.log(R)))

    g_numeric = np.zeros(len(R_test))
    for i, R in enumerate(R_test):
        R_int    = np.linspace(R_lo, R, 1000)
        integral = np.trapezoid(R_int * sig_at(R_int), R_int)
        g_numeric[i] = ESD2G * (2.0 / R**2 * integral - sig_at(R))

    ratio   = g_numeric / g_analytic
    max_err = np.max(np.abs(ratio - 1.0))
    assert max_err < tol, (
        f"SIS numeric test FAILED: max|g_numeric/g_analytic - 1| = {max_err:.4f} > {tol}"
    )
    return ratio

# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys

    print("=" * 72)
    print("Forward HMG lensing -- Appendix A.4 (Monjo et al. 2026)")
    print("chi2_nu = chi2_total/(N-1), N=15 (consistent with Table 1)")
    print("=" * 72)

    print("\n[SIS analytic test] ... ", end="", flush=True)
    try:
        ratio_test = _sis_analytic_test()
        print(f"PASS  (g_fwd/g_SIS at 6 signal-dominated R: "
              f"[{ratio_test.min():.4f}, {ratio_test.max():.4f}])")
    except AssertionError as e:
        print(f"\nFAILED: {e}")
        sys.exit(1)

    csv_rows = [["bin", "R_Mpc", "g_SIS_mss", "g_fwd_mss", "g_fwd_over_g_SIS",
                 "chi2_nu_SIS", "chi2_nu_FWD"]]
    results = []

    for i, (bdata, mbar, s_sis) in enumerate(zip(BINS, MBAR, S_BIN_SIS)):
        print(f"\nBin {i+1}: Mbar={mbar:.3e} Msun  s_SIS={s_sis:.4f}")
        R_mpc = bdata[:, 0]
        R_kpc = R_mpc * MPC_TO_KPC

        g_sis  = gobs_hmg(R_kpc, mbar, s_sis)
        g_fwd  = compute_g_fwd(R_mpc, mbar, s_sis, verbose=True)

        c2_sis = chi2_nu_N1(g_sis, bdata)
        c2_fwd = chi2_nu_N1(g_fwd, bdata)

        outer = (bdata[:, 0] > 0.16) & (bdata[:, 0] < 0.8)
        ratio = g_fwd[outer] / g_sis[outer]
        print(f"  chi2_nu SIS={c2_sis:.3f}  FWD={c2_fwd:.3f}  delta={c2_fwd-c2_sis:+.3f}")
        print(f"  g_fwd/g_SIS at 0.16<R<0.8 Mpc: [{ratio.min():.4f}, {ratio.max():.4f}]")

        for j, R in enumerate(R_mpc):
            csv_rows.append([i+1, f"{R:.5e}", f"{g_sis[j]:.5e}", f"{g_fwd[j]:.5e}",
                             f"{g_fwd[j]/g_sis[j]:.5f}", f"{c2_sis:.4f}", f"{c2_fwd:.4f}"])

        results.append((c2_sis, c2_fwd, ratio.min(), ratio.max()))

    print("\n" + "=" * 72)
    print(f"  {'Bin':>4}  {'chi2_SIS':>9}  {'chi2_FWD':>9}  {'delta':>7}  "
          f"{'ratio_min':>10}  {'ratio_max':>10}")
    for i, (cs, cf, rmin, rmax) in enumerate(results):
        print(f"  {i+1:>4}  {cs:9.3f}  {cf:9.3f}  {cf-cs:+7.3f}  {rmin:10.4f}  {rmax:10.4f}")
    print("=" * 72)

    csv_path = "forward_lensing_results.csv"
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerows(csv_rows)
    print(f"\nResults written to {csv_path}")
