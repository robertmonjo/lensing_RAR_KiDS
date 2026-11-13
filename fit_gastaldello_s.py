"""
fit_gastaldello_s.py
====================
Fit HMG parameter s for each of the 16 Gastaldello+2007 X-ray groups
(ApJ 669, 158) using NFW-reconstructed mass profiles.

Strategy per group:
  1. NFW total mass profile from (r_s, c, M_Delta):
       M_tot(<r) = M_Delta * [f(r/r_s)] / [f(c)]  where f(x)=ln(1+x)-x/(1+x)
  2. Baryonic mass:
       M_bar(<r) = M_gas(<r) + M_star
     Gas: constant gas fraction f_gas within r_Delta (i.e. M_gas = f_gas * M_tot)
     Stars: point mass at center (negligible at r > 50 kpc)
  3. 15 log-spaced radii from 0.1*r_s to r_Delta
  4. Fit s minimizing chi2 in log10(g_obs) with sigma_log=0.10 (10% scatter)
  5. Uncertainty: Delta chi2 = 1 interval

Output:
  - Table printed to stdout (s_best, chi2_nu, s_lo, s_hi per group)
  - CSV saved to scripts_replicable/outputs/gastaldello_s_fit.csv
  - Plot saved to scripts_replicable/outputs/gastaldello_s_fit.png
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar, brentq
from pathlib import Path

# ── Constants (match hmg_model.py exactly) ────────────────────────────────
G_SI        = 6.674e-11      # m3 kg-1 s-2
G_CODE      = 4.30091727e-6  # (km/s)2 kpc / Msun
T0          = 14.11          # kpc/(km/s): Hubble time
C_KMS       = 299792.458     # speed of light km/s (used in g_s)
CODE_TO_SI  = 3.24078e-14    # (km/s)2/kpc → m/s2
KPC_M       = 3.0857e19      # m per kpc
MSUN_KG     = 1.989e30
SIN2_GU     = np.sin(np.pi / 3) ** 2
SIN2_GCEN   = 1.0

# ── HMG model: copied from hmg_model.py (gobs_hmg / _cos_over_gamma) ────
def _cos_over_gamma(xi2):
    xi2   = np.asarray(xi2, float)
    delta = np.abs(1.0 - xi2) / (1.0 + xi2 + 1e-30)
    s2    = SIN2_GU + (SIN2_GCEN - SIN2_GU) * np.clip(delta, 0.0, 1.0)
    gs    = np.arcsin(np.sqrt(np.clip(s2, 0.0, 1.0)))
    return np.cos(gs) / np.maximum(gs, 1e-10)

def gobs_hmg_si(r_kpc, mbar_msun, s):
    """HMG g_obs [m/s²]: formula: sqrt(a_N*(a_N + 2c/T0*q(ξ²)))*CODE_TO_SI."""
    r    = np.asarray(r_kpc, float)
    v2n  = G_CODE * mbar_msun / r       # (km/s)²: Newtonian circular vel²
    vh2  = (r / T0) ** 2
    xi2  = 1.0 / s ** 3 + vh2 / (12.0 * v2n + 1e-60)
    a_n  = v2n / r                      # (km/s)²/kpc
    g_s  = 2.0 * C_KMS / T0 * _cos_over_gamma(xi2)   # (km/s)²/kpc
    return np.sqrt(np.maximum(a_n * (a_n + g_s), 0.0)) * CODE_TO_SI

# ── NFW profile ────────────────────────────────────────────────────────────
def nfw_f(x):
    """f(x) = ln(1+x) - x/(1+x)"""
    return np.log1p(x) - x / (1.0 + x)

def nfw_enclosed(r_kpc, r_s_kpc, c, m_delta_msun):
    """Enclosed NFW mass at r_kpc, given scale radius r_s=r_Delta/c and M_Delta."""
    x       = r_kpc / r_s_kpc
    x_vir   = c              # r_Delta / r_s = c
    return m_delta_msun * nfw_f(x) / nfw_f(x_vir)

# ── σ_eff per radial point from observational errors ─────────────────────
def sigma_eff_arr(r_ok, r_s_kpc, c, c_err, r_delta_kpc,
                  m_delta_1e13, m_delta_err,
                  m_gas_1e13, m_gas_err, f_gas, f_gas_err,
                  m_star_1e11, m_star_err, M_bar_ok):
    """
    Propagate Gastaldello+2007 measurement errors into σ_eff per radial point.

    In Δ(r) = log g_obs(r) − log g_HMG(g_bar(r), s) both g_obs and g_bar derive
    from the same NFW profile (M_tot(r) and f_gas · M_tot(r)).  A fractional shift
    δ in M_tot shifts log g_obs by δ/ln10 but log g_HMG by ≈ 0.5·δ/ln10 (because
    ∂ log g_HMG / ∂ log g_bar ≈ 0.5 in the DM-dominated regime).  Net effect on Δ:
    0.5·δ/ln10: a factor-2 cancellation for correlated sources.

    Sources:
      (a) M_Delta_err  : normalization, fully correlated across r; 0.5× cancellation
      (b) c_err        : changes NFW profile shape; varies with r; 0.5× cancellation
      (c) f_gas_err    : shifts only g_bar independently;   0.5× from ∂ log g_HMG/∂ log g_bar
      (d) M_star_err   : shifts only g_bar independently;   0.5× same as (c)

    Result: σ_eff per data point in dex.
    """
    ln10 = np.log(10.0)

    # (a) M_Delta normalization
    sig_Mdelta = 0.5 * m_delta_err / (m_delta_1e13 * ln10)  # scalar, same for all r

    # (b) c_err → NFW shape uncertainty (numerical)
    c_lo = max(c - c_err, 0.5)
    c_hi = c + c_err
    M_tot_msun = m_delta_1e13 * 1e13
    M_plus  = nfw_enclosed(r_ok, r_delta_kpc / c_hi, c_hi, M_tot_msun)
    M_minus = nfw_enclosed(r_ok, r_delta_kpc / c_lo, c_lo, M_tot_msun)
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = M_plus / M_minus
    ratio = np.where(ratio > 0, ratio, 1.0)
    sig_c = 0.5 * np.abs(np.log10(ratio)) / 2.0  # half the ±1σ range, ×0.5 cancellation

    # (c) f_gas_err: independent source, only shifts g_bar
    sig_fgas = 0.5 * f_gas_err / (f_gas * ln10)  # scalar

    # (d) M_star_err: independent source, only shifts g_bar
    if (not np.isnan(m_star_err)) and (not np.isnan(m_star_1e11)) and (m_star_1e11 > 0):
        sig_mstar = 0.5 * m_star_err * 1e11 / (M_bar_ok * ln10)
    else:
        sig_mstar = np.zeros_like(r_ok)

    return np.sqrt(sig_Mdelta**2 + sig_c**2 + sig_fgas**2 + sig_mstar**2)


# ── Fit s for one group ────────────────────────────────────────────────────
def fit_s(name, r_s_kpc, c, c_err, r_delta_kpc,
          m_delta_1e13, m_delta_err, m_gas_1e13, m_gas_err,
          f_gas, f_gas_err, m_star_1e11, m_star_err,
          sigma_log=0.10, n_pts=15, s_lo_bound=0.05, s_hi_bound=6.0):
    """
    Returns: s_best, chi2_nu_fix, chi2_nu_obs, chi2_nu_s1_obs,
             n_sigma_s1, s_lo, s_hi, r_ok, gbar_si, gobs_si

    chi2_nu_fix   : original χ²_ν with fixed σ=0.10
    chi2_nu_obs   : χ²_ν at s_best using σ_eff from measurement errors
    chi2_nu_s1_obs: χ²_ν at s=1 using σ_eff from measurement errors
    n_sigma_s1    : mean pull |<Δ/σ_eff>| at s=1 (single-measurement proxy)
    """
    M_tot_msun  = m_delta_1e13 * 1e13
    M_gas_msun  = m_gas_1e13  * 1e13
    M_star_msun = (0.0 if np.isnan(m_star_1e11) else m_star_1e11 * 1e11)
    f_gas_val   = M_gas_msun / M_tot_msun

    r_min = max(0.10 * r_s_kpc, 20.0)
    r_arr = np.geomspace(r_min, r_delta_kpc, n_pts)

    M_tot_arr = nfw_enclosed(r_arr, r_s_kpc, c, M_tot_msun)
    M_gas_arr = f_gas_val * M_tot_arr
    M_bar_arr = M_gas_arr + M_star_msun

    ok = (M_bar_arr < 0.99 * M_tot_arr) & (M_bar_arr > 0) & (M_tot_arr > 0)
    if ok.sum() < 3:
        return name, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, r_arr, None, None

    r_ok     = r_arr[ok]
    M_tot_ok = M_tot_arr[ok]
    M_bar_ok = M_bar_arr[ok]

    r_si     = r_ok * KPC_M
    gobs_si  = G_SI * M_tot_ok * MSUN_KG / r_si ** 2
    gbar_si  = G_SI * M_bar_ok * MSUN_KG / r_si ** 2
    log_gobs = np.log10(gobs_si)

    # ── Fixed-σ fit (original) ─────────────────────────────────────────────
    def chi2_fix(s):
        gm = gobs_hmg_si(r_ok, M_bar_ok, s)
        return float(np.sum(((log_gobs - np.log10(gm + 1e-60)) / sigma_log) ** 2))

    res       = minimize_scalar(chi2_fix, bounds=(s_lo_bound, s_hi_bound), method="bounded")
    s_best    = res.x
    chi2_min  = res.fun
    nu        = ok.sum() - 1
    chi2_nu_fix = chi2_min / max(nu, 1)

    # 1-sigma interval (fixed-σ based)
    thr = chi2_min + 1.0
    def delta_fix(s): return chi2_fix(s) - thr
    try:
        s_lo = brentq(delta_fix, s_lo_bound, s_best) if delta_fix(s_lo_bound) > 0 else s_lo_bound
    except ValueError:
        s_lo = s_lo_bound
    try:
        s_hi = brentq(delta_fix, s_best, s_hi_bound) if delta_fix(s_hi_bound) > 0 else s_hi_bound
    except ValueError:
        s_hi = s_hi_bound

    # ── Observational-error σ_eff ──────────────────────────────────────────
    sig = sigma_eff_arr(r_ok, r_s_kpc, c, c_err, r_delta_kpc,
                        m_delta_1e13, m_delta_err,
                        m_gas_1e13, m_gas_err, f_gas_val, f_gas_err,
                        m_star_1e11, m_star_err, M_bar_ok)

    def chi2_obs(s):
        gm = gobs_hmg_si(r_ok, M_bar_ok, s)
        resid = (log_gobs - np.log10(gm + 1e-60)) / sig
        return float(np.sum(resid ** 2))

    chi2_nu_obs    = chi2_obs(s_best) / max(nu, 1)
    chi2_nu_s1_obs = chi2_obs(1.0)   / max(nu, 1)

    # Mean pull at s=1 (single-measurement proxy for correlated points)
    gm_s1  = gobs_hmg_si(r_ok, M_bar_ok, 1.0)
    Delta  = log_gobs - np.log10(gm_s1 + 1e-60)   # residuals at s=1
    n_sigma_s1 = float(np.mean(Delta / sig))       # signed mean pull

    return (name, s_best, chi2_nu_fix, chi2_nu_obs, chi2_nu_s1_obs, n_sigma_s1,
            s_lo, s_hi, r_ok, gbar_si, gobs_si)

# ── Load data ──────────────────────────────────────────────────────────────
import pandas as pd

DATA_FILE = Path(__file__).resolve().parent / "data" / "gastaldello2007.csv"
df = pd.read_csv(DATA_FILE, comment="#")

# ── Run fits ───────────────────────────────────────────────────────────────
results = []
HDR = (f"{'Group':20s}  {'f_bar':>5}  {'s_best':>7}  {'[s_lo':>6}  {'s_hi]':>5}  "
       f"{'χ²ν(fix)':>9}  {'σ_eff':>6}  {'χ²ν(obs)':>9}  "
       f"{'χ²ν(s=1)':>9}  {'n_σ(s=1)':>9}")
print(HDR)
print("-" * len(HDR))

for _, row in df.iterrows():
    name       = str(row['name'])
    Delta      = int(row['Delta'])
    r_s        = float(row['r_s_kpc'])
    c          = float(row['c'])
    c_err      = float(row['c_err'])
    r_delta    = float(row['r_Delta_kpc'])
    M_delta    = float(row['M_Delta_1e13'])
    M_delta_e  = float(row['M_Delta_err'])
    M_gas      = float(row['M_gas_1e13'])
    M_gas_e    = float(row['M_gas_err'])
    f_gas      = float(row['f_gas'])
    f_gas_e    = float(row['f_gas_err'])
    M_star     = float(row['M_star_1e11'])   # may be nan
    M_star_e   = float(row['M_star_err'])    # may be nan

    res = fit_s(name, r_s, c, c_err, r_delta,
                M_delta, M_delta_e, M_gas, M_gas_e, f_gas, f_gas_e,
                M_star, M_star_e)
    (n_name, s_b, c2nu_fix, c2nu_obs, c2nu_s1, nsig_s1,
     s_lo, s_hi, r_ok, gbar_arr, gobs_arr) = res

    f_bar = f_gas + (0 if np.isnan(M_star) else M_star * 1e11 / (M_delta * 1e13))
    n_pts = len(r_ok) if r_ok is not None else 0

    # mean σ_eff across radial grid (for display)
    if r_ok is not None and not np.isnan(s_b):
        M_tot_msun  = M_delta * 1e13
        M_gas_msun  = M_gas  * 1e13
        M_star_msun = 0.0 if np.isnan(M_star) else M_star * 1e11
        f_gas_val   = M_gas_msun / M_tot_msun
        M_tot_arr   = nfw_enclosed(r_ok, r_s, c, M_tot_msun)
        M_bar_arr   = f_gas_val * M_tot_arr + M_star_msun
        sig_arr = sigma_eff_arr(r_ok, r_s, c, c_err, r_delta,
                                M_delta, M_delta_e, M_gas, M_gas_e,
                                f_gas_val, f_gas_e, M_star, M_star_e, M_bar_arr)
        sig_mean = float(np.mean(sig_arr))
    else:
        sig_mean = np.nan

    if not np.isnan(s_b):
        print(f"{name:20s}  {f_bar:5.3f}  {s_b:7.3f}  [{s_lo:5.3f}  {s_hi:5.3f}]  "
              f"{c2nu_fix:9.3f}  {sig_mean:6.3f}  {c2nu_obs:9.3f}  "
              f"{c2nu_s1:9.3f}  {nsig_s1:+9.2f}")
        results.append((name, Delta, r_s, c, r_delta, M_delta, M_gas, M_star, s_b, s_lo, s_hi,
                        c2nu_fix, c2nu_obs, c2nu_s1, nsig_s1, n_pts, f_bar))
    else:
        print(f"{name:20s}  {f_bar:5.3f}  {'---':>7}")

# ── Summary statistics ─────────────────────────────────────────────────────
s_vals  = np.array([r[8]  for r in results if not np.isnan(r[8])])
c2_fix  = np.array([r[11] for r in results if not np.isnan(r[11])])
c2_obs  = np.array([r[12] for r in results if not np.isnan(r[12])])
c2_s1   = np.array([r[13] for r in results if not np.isnan(r[13])])
nsig_arr= np.array([r[14] for r in results if not np.isnan(r[14])])
print()
print(f"Median s            = {np.median(s_vals):.3f}")
print(f"Median chi2_nu(fix) = {np.median(c2_fix):.3f}")
print(f"Median chi2_nu(obs) = {np.median(c2_obs):.3f}")
print(f"Median chi2_nu(s=1) = {np.median(c2_s1):.3f}")
print(f"Median |n_sigma(s=1)| = {np.median(np.abs(nsig_arr)):.2f}  [correlated pull proxy]")
print(f"Groups with |n_sigma|<2: {np.sum(np.abs(nsig_arr)<2)}/{len(nsig_arr)}")
print()
print("Groups with |n_sigma(s=1)|>2:")
for r in results:
    if abs(r[14]) > 2:
        print(f"  {r[0]:20s}  n_sigma={r[14]:+.2f}  chi2_nu_obs(s=1)={r[13]:.2f}  f_bar={r[16]:.3f}")

# Global joint fit
print("\n=== Global joint fit (all groups, single s) ===")
all_r, all_gbar, all_gobs, all_mbar = [], [], [], []
for _, row in df.iterrows():
    r_s      = float(row['r_s_kpc'])
    c        = float(row['c'])
    c_err    = float(row['c_err'])
    r_delta  = float(row['r_Delta_kpc'])
    M_delta  = float(row['M_Delta_1e13'])
    M_delta_e= float(row['M_Delta_err'])
    M_gas    = float(row['M_gas_1e13'])
    M_gas_e  = float(row['M_gas_err'])
    f_gas    = float(row['f_gas'])
    f_gas_e  = float(row['f_gas_err'])
    M_star   = float(row['M_star_1e11'])
    M_star_e = float(row['M_star_err'])
    name_g   = str(row['name'])
    res = fit_s(name_g, r_s, c, c_err, r_delta,
                M_delta, M_delta_e, M_gas, M_gas_e, f_gas, f_gas_e, M_star, M_star_e)
    (_, _, _, _, _, _, _, _, r_ok, gbar_arr, gobs_arr) = res
    if r_ok is not None and gbar_arr is not None:
        m_star_msun = 0.0 if np.isnan(M_star) else M_star * 1e11
        f_gas_val   = M_gas * 1e13 / (M_delta * 1e13)
        M_tot_arr   = nfw_enclosed(r_ok, r_s, c, M_delta * 1e13)
        M_bar_arr   = f_gas_val * M_tot_arr + m_star_msun
        all_r.append(r_ok)
        all_gbar.append(gbar_arr)
        all_gobs.append(gobs_arr)
        all_mbar.append(M_bar_arr)

r_all    = np.concatenate(all_r)
gbar_all = np.concatenate(all_gbar)
gobs_all = np.concatenate(all_gobs)
mbar_all = np.concatenate(all_mbar)
log_gobs_all = np.log10(gobs_all)
sigma_log = 0.10

def chi2_global(s):
    gm = gobs_hmg_si(r_all, mbar_all, s)
    return float(np.sum(((log_gobs_all - np.log10(gm + 1e-60)) / sigma_log) ** 2))

res_g = minimize_scalar(chi2_global, bounds=(0.05, 6.0), method="bounded")
s_glob  = res_g.x
c2_glob = res_g.fun / (len(r_all) - 1)
thr_g   = res_g.fun + 1.0
try:
    s_glob_lo = brentq(lambda s: chi2_global(s) - thr_g, 0.05, s_glob)
except Exception:
    s_glob_lo = np.nan
try:
    s_glob_hi = brentq(lambda s: chi2_global(s) - thr_g, s_glob, 6.0)
except Exception:
    s_glob_hi = np.nan

print(f"s_global = {s_glob:.3f} +{s_glob_hi - s_glob:.3f} -{s_glob - s_glob_lo:.3f}")
print(f"chi2_nu  = {c2_glob:.3f}  (N={len(r_all)}, nu={len(r_all)-1})")

# ── Save CSV ───────────────────────────────────────────────────────────────
OUT_DIR = Path(__file__).resolve().parent / "outputs"
OUT_DIR.mkdir(exist_ok=True)
out_csv = OUT_DIR / "gastaldello_s_fit.csv"

with open(out_csv, "w", encoding="utf-8") as f:
    f.write("# Gastaldello+2007: HMG s fit per group (NFW reconstruction, f_gas constant)\n")
    f.write("# sigma_log=0.10 (10% scatter on log10(g_obs)); dof=N-1\n")
    f.write("name,Delta,r_s_kpc,c,r_Delta_kpc,M_Delta_1e13,M_gas_1e13,M_star_1e11,"
            "s_best,s_lo,s_hi,chi2_nu,N_pts,f_bar\n")
    for r in results:
        f.write(",".join(str(x) for x in r) + "\n")
    f.write(f"# GLOBAL: s={s_glob:.4f} +{s_glob_hi-s_glob:.4f} -{s_glob-s_glob_lo:.4f},"
            f" chi2_nu={c2_glob:.4f}, N={len(r_all)}\n")

print(f"\nSaved: {out_csv}")

# ── Plot: s per group vs M_Delta ──────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
s_arr   = np.array([r[8]  for r in results])
slo_arr = np.array([r[9]  for r in results])
shi_arr = np.array([r[10] for r in results])
M_arr   = np.array([r[5]  for r in results])   # M_Delta 1e13 Msun
Delta_arr = np.array([r[1] for r in results])

ax.axhline(s_glob, color="#e74c3c", lw=1.5, ls="--", zorder=1,
           label=rf"Global $s={s_glob:.3f}^{{+{s_glob_hi-s_glob:.3f}}}_{{-{s_glob-s_glob_lo:.3f}}}$")
ax.axhspan(s_glob_lo, s_glob_hi, color="#e74c3c", alpha=0.12, zorder=0)

sc = ax.scatter(M_arr, s_arr, c=Delta_arr, cmap="viridis", s=50, zorder=3,
                vmin=500, vmax=2500)
ax.errorbar(M_arr, s_arr, yerr=[s_arr - slo_arr, shi_arr - s_arr],
            fmt="none", ecolor="0.5", lw=0.8, capsize=2, zorder=2)
plt.colorbar(sc, ax=ax, label=r"$\Delta$ (overdensity)")

ax.set_xscale("log")
ax.set_xlabel(r"$M_\Delta\;[10^{13}\,M_\odot]$", fontsize=11)
ax.set_ylabel(r"$s_\mathrm{best}$", fontsize=11)
ax.set_title("Gastaldello+2007: s per group", fontsize=11)
ax.legend(fontsize=9)

# RAR plot: all points + HMG model
ax2 = axes[1]
ax2.scatter(gbar_all, gobs_all, s=8, alpha=0.5, color="#2980b9", zorder=3,
            label=f"All groups ({len(r_all)} pts)")
gbar_c = np.logspace(np.log10(gbar_all.min()*0.5), np.log10(gbar_all.max()*2), 200)
# HMG global (using median M_bar scale for model curve: illustrative)
ax2.set_xscale("log"); ax2.set_yscale("log")
ax2.plot([gbar_all.min()*0.5, gbar_all.max()*2],
         [gbar_all.min()*0.5, gbar_all.max()*2],
         "gray", lw=0.8, ls="--", zorder=1, label="Newton 1:1")
ax2.set_xlabel(r"$g_\mathrm{bar}\;[\mathrm{m\,s^{-2}}]$", fontsize=11)
ax2.set_ylabel(r"$g_\mathrm{obs}\;[\mathrm{m\,s^{-2}}]$", fontsize=11)
ax2.set_title(rf"RAR: 16 groups, global $s={s_glob:.3f}$, $\chi^2_\nu={c2_glob:.2f}$", fontsize=11)
ax2.legend(fontsize=9)

plt.tight_layout()
out_png = OUT_DIR / "gastaldello_s_fit.png"
plt.savefig(str(out_png), dpi=150, bbox_inches="tight")
print(f"Saved: {out_png}")
