"""
xcop_rnei_nsig.py -- X-COP cluster goodness-of-fit for the CORRECT HMG "r_nei"
model, reproducing the sample median n_sigma = 0.918.

WHY TWO MODELS EXIST, AND WHY ONLY ONE IS VALID FOR X-COP
---------------------------------------------------------
The Hyperconical Modified Gravity (HMG; Monjo 2025, ApJ 981, 195,
DOI 10.3847/1538-4357/adb723) prediction for a cluster depends on the local
projection factor eps0(r). Two ways to set it have been compared on X-COP:

  * Model B -- "scalar s" (one CONSTANT eps per cluster).  A single scalar
    s (equivalently one constant eps) is fitted per cluster, exactly the
    lensing/KiDS morphological submodel that is used when NO radial density
    profile is available.  Applied to X-COP it gives median n_sigma = 2.731.
    It is the WRONG model here: a cluster is NOT a point source -- forcing a
    single constant eps throws away the radial fall-off of the gas density
    that X-COP actually measures, so the outer bins are systematically
    mispredicted.

  * Model A / C -- "r_nei" (eps0 varies RADIALLY with the local density).
    eps0(r)^2 = 1/6 + rho_enc(r) / rho_vac, evaluated point-by-point along
    the measured gas profile and clipped at the fitted neighbour radius
    r_nei (inner cap at r_grav, outer cap at r_nei).  This is the model of
    the X-COP paper (scripts/python/fig2_table2.py + hmg_common.py).  It has
    the SAME number of free parameters as Model B (one per cluster: r_nei
    instead of s); the gain in goodness-of-fit comes purely from letting
    eps0 track the density profile radially, not from extra parameters.
    It gives median n_sigma = 0.918 -- reproduced here.

Model A and Model C are mathematically equivalent (verified numerically in
model_comparison.md); the "s-framework" can also reach 0.918 ONLY if xi(r) is
allowed to vary per point following the FITS profile -- i.e. it is no longer a
scalar.  A single scalar epsilon (Model B, 2.731) can NOT.

WHAT THIS SCRIPT COMPUTES
-------------------------
Per cluster, over the published fit window R_REF >= 1000 kpc:
  1. eps0(r) = sqrt(1/6 + rho_enc(r)/rho_vac), rho_enc from the enclosed hot
     GAS mass (gas-only baryonic source, as in fig2_table2.py), clipped so
     eps1[r<r_grav]=max(eps0|window) and eps1[r>r_nei]=min(eps0|window),
     window = (r_grav, r_nei).  The clip needs the FULL radial profile, so
     this script reads the complete EIN3 grid, not only the R>=1000 bins.
  2. the full HMG total acceleration a_pred1 (time-like + spatial).
  3. the reduced chi^2 in log-g space vs the total hydrostatic acceleration
     (dark + gas), df = N_i - 1, hence n_sigma_i = sqrt(chi2_nu_i).
  4. the MEDIAN n_sigma over the 12 clusters = 0.918.

The physics is a verbatim port of hmg_common.py:hmg_predict (all in SI;
add_e02=True, r_grav=50 as in fig2_table2.py).  NOTE on the units bug flagged
in model_comparison.md: that bug concerns the SCALAR s-model helper
hmg_gtot_s (gs must stay in CODE units, do NOT multiply by CODE_TO_SI).  The
r_nei model below never uses code units at all -- every quantity is SI from
the start -- so the bug does not apply here.

SELF-CONTAINED
--------------
Reads ONLY data/external_reference/xcop_profiles.txt (the full EIN3 profiles
and per-cluster r_nei, extracted once with astropy.io.fits from the X-COP Einasto
mass profiles (Eckert et al. 2022)).
No astropy or absolute paths at run time.  Writes outputs/xcop_rnei_nsig.txt.

Reference: model_comparison.md (xcop repo, table column "n_sigma(r_nei)").
"""
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
PROF = HERE / "data" / "external_reference" / "xcop_profiles.txt"
OUT  = HERE / "outputs" / "xcop_rnei_nsig.txt"

# ---- Physical constants, verbatim from hmg_common.py (SI unless noted) -------
c0   = 3e8                                    # speed of light, m/s (paper value)
Msol = 1.9891e30                              # solar mass, kg
kpc  = 3261.8116478174 * 365*24*3600 * c0     # kpc in metres
T0   = 13.7e9 * 365*24*3600                   # age of the universe, s (13.7 Gyr)
GN   = 6.674e-11                              # gravitational constant, SI
rho_vac = 3.0 / (8*np.pi*GN*T0**2)            # HMG vacuum/critical density
GA_EMPTY  = np.pi/3                            # gamma_U universal angle
GA_CENTER = np.pi/2                            # gamma_center (1-parameter model)
R_GRAV    = 50.0                               # inner grav.-domination scale, kpc
R_FIT_MIN = 1000.0                             # published fit window, kpc

# Cross-check against the lensing-pipeline conventions (not used in the SI r_nei
# model, kept here only so the two frameworks are visible side by side):
#   G_CODE=4.30091727e-6, T0_pipeline=14.11, C_KMS=299792.458,
#   CODE_TO_SI=3.24078e-14, A0_SI=1.20e-10.  The X-COP r_nei fit is done in SI
#   with the paper's T0=13.7 Gyr, so rho_vac above is the hmg_common.py value.


def hmg_gamma0(quotient, gaempty=GA_EMPTY, gacenter=GA_CENTER):
    """HMG projection factor gamma_0 = gamma_sys / cos(gamma_sys)."""
    g_sys = np.arcsin(np.sqrt(np.sin(gaempty)**2 +
                              (np.sin(gacenter)**2 - np.sin(gaempty)**2) * quotient))
    return g_sys / np.cos(g_sys)


def hmg_predict(M_enc_kg, R_ref_kpc, R0_kpc, r_grav=R_GRAV):
    """Verbatim port of hmg_common.py:hmg_predict (add_e02=True).
    Returns a_pred1 [m/s^2]: full HMG total acceleration on the radial grid."""
    M = np.asarray(M_enc_kg, float)
    R_ref_m = np.asarray(R_ref_kpc, float) * kpc
    acc_newton = GN * M / R_ref_m**2
    ve2  = 2 * acc_newton * R_ref_m           # v_N^2 = 2 G M / r
    vh2  = R_ref_m**2 / T0**2                  # v_H^2 = (r/t)^2
    dens = M / (4/3 * np.pi * R_ref_m**3)      # rho_enc(r)

    eps0 = np.sqrt(1/6 + dens / rho_vac)       # radially-varying projection factor

    R = np.asarray(R_ref_kpc)
    win = (R > r_grav) & (R < R0_kpc)
    eps0_low = np.nanmax(eps0[win])            # inner cap
    eps0_hig = np.nanmin(eps0[win])            # outer cap
    eps1 = eps0.copy()
    eps1[R < r_grav] = eps0_low
    eps1[R > R0_kpc] = eps0_hig

    q1 = np.abs(ve2 - vh2*eps1**2) / (vh2*eps1**2 + ve2)
    g1 = hmg_gamma0(q1)
    a_pred1 = np.sqrt(acc_newton**2 + 2*acc_newton*((c0/T0)/g1))   # total
    return a_pred1


def load_profiles(path):
    """Read the self-contained EIN3 extract: per-cluster r_nei (from the
    header comments) and per-row profile columns.  No astropy at run time."""
    rnei = {}
    rows = {}          # cluster -> list of [R_IN,R_OUT,R_REF,M_DM,M_DM_LO,M_DM_HI,MGAS,MGAS_LO,MGAS_HI]
    order = []
    with open(path, "r") as fh:
        for line in fh:
            s = line.strip()
            if s.startswith("#"):
                p = s.split()
                if len(p) >= 4 and p[1] == "rnei":
                    rnei[p[2]] = float(p[3])
                continue
            if not s:
                continue
            p = s.split()
            c = p[0]
            if c not in rows:
                rows[c] = []
                order.append(c)
            rows[c].append([float(x) for x in p[1:10]])
    prof = {c: np.asarray(rows[c], float) for c in order}
    return order, rnei, prof


def cluster_nsigma(arr, r_nei):
    """log-g reduced chi^2 and n_sigma = sqrt(chi2_nu) for one cluster,
    over R_REF >= 1000 kpc, df = N - 1.  Mirrors fig2_table2.py's obs/err."""
    R_IN, R_OUT, R_REF, M_DM, M_DM_LO, M_DM_HI, MGAS, MGAS_LO, MGAS_HI = arr.T

    M_gas = Msol * MGAS
    a_pred = hmg_predict(M_gas, R_REF, r_nei)                       # gas-only source

    obs = GN*Msol*(M_DM + MGAS) / (R_REF*kpc)**2                    # total hydrostatic accel
    with np.errstate(divide="ignore", invalid="ignore"):
        obs_HI = GN*Msol*(M_DM_HI + MGAS_HI) / (R_IN *kpc)**2
    obs_LO = GN*Msol*(M_DM_LO + MGAS_LO) / (R_OUT*kpc)**2
    obs_err = (obs_HI - obs_LO) / 2

    m = R_REF >= R_FIT_MIN
    N = int(m.sum())
    sig_log = obs_err[m] / (obs[m] * np.log(10))                    # log-g space error
    chi2 = np.nansum(((np.log10(obs[m]) - np.log10(a_pred[m])) / sig_log)**2)
    chi2_nu = chi2 / (N - 1)
    return N, chi2_nu, np.sqrt(chi2_nu)


def main():
    order, rnei, prof = load_profiles(PROF)
    print(f"{'Cluster':<9}{'r_nei':>7}{'N':>5}{'chi2_nu':>10}{'n_sigma':>10}")
    ns = []
    for c in order:
        N, chi2_nu, nsig = cluster_nsigma(prof[c], rnei[c])
        ns.append(nsig)
        print(f"{c:<9}{int(rnei[c]):>7}{N:>5}{chi2_nu:>10.4f}{nsig:>10.4f}")
    median = float(np.median(ns))
    print(f"\nMEDIAN n_sigma (r_nei model) = {median:.4f}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as fh:
        fh.write("# X-COP HMG r_nei-model goodness-of-fit (radially-varying eps0).\n")
        fh.write("# Median n_sigma over the 12 X-COP clusters, log-g reduced chi^2,\n")
        fh.write("# R_REF>=1000 kpc, df=N-1, n_sigma_i=sqrt(chi2_nu_i).\n")
        fh.write("# Reproduces model_comparison.md (xcop repo) Model A/C = 0.918.\n")
        fh.write("# The invalid scalar-s model (Model B, one constant eps/cluster) gives 2.731.\n")
        fh.write("# Generated by xcop_rnei_nsig.py from data/external_reference/xcop_profiles.txt\n")
        fh.write("# columns: per-cluster n_sigma, then the sample median on the last line.\n")
        for c, nsig in zip(order, ns):
            fh.write(f"{c:<9} {nsig:.4f}\n")
        fh.write(f"median_nsigma {median:.4f}\n")
    print(f"wrote {OUT}")
    return median


if __name__ == "__main__":
    main()
