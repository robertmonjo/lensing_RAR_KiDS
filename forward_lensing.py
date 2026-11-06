"""Appendix A.4: forward lensing projection and Table A.2.

Under the assumption that the lensing and dynamical potentials coincide (eta = 1), a
model acceleration g(r) defines an effective density
    rho_eff(r) = (4 pi r^2)^-1 d[r^2 g(r) / G] / dr,
which is projected along the line of sight (Abel integral, cut-off z_max = 200 Mpc),
    Sigma(R) = 2 int_0^z_max rho_eff(sqrt(R^2 + z^2)) dz,
    DeltaSigma(R) = Sigma_bar(<R) - Sigma(R),
    g_fwd(R) = 4 G [DeltaSigma(R) + M_bar / (pi R^2)],
where the last term restores the central baryonic point mass. The SIS kernel used for
the data is g_SIS(R) = g(r = R).

Models (as in Table 1): HMG with one s per mass bin (s < 1 branch); MOND with the
McGaugh (2016) interpolation and a0* = 1.2e-10 m/s^2; CDM with Moster et al. (2013)
halo masses and one global two-halo amplitude zeta_0.

Statistic: chi2_nu_log (Sect. 2.4). Per bin chi2_nu = chi2 / (15 - k_bin) with
k_bin = 1 for HMG and 0 for MOND and CDM; pooled rows divide the summed chi2 by
nu = 56 (HMG), 60 (MOND), 59 (CDM). Global-fit rows refit one parameter to all 60
points separately under each kernel (nu = 59): HMG s, MOND a0, CDM zeta_0.

Outputs
  outputs/forward_lensing_table_A2.csv   Table A.2
  outputs/forward_lensing_kernel.csv     g_fwd / g_SIS ranges quoted in Appendix A.4
"""
import csv
import os
import warnings
import numpy as np
from scipy.integrate import quad, IntegrationWarning
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar

import hmg_model as h

warnings.filterwarnings("ignore", category=IntegrationWarning)
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")

PC_M = 3.0857e16
ESD2G = 4.0 * h.G_SI * h.MSUN_KG / PC_M**2          # m/s^2 per Msun/pc^2
A0_STAR = 1.2e-10

DEFAULTS = dict(r_min=3e-4, r_max=300.0, n_r=6000, z_max=200.0, n_Rg=100,
                n_int=1000, R_lo=0.001, epsrel=1e-5, split=True)


# ── Projection ────────────────────────────────────────────────────────────────
def project(g_of_r, R_arr, mbar, p=DEFAULTS, point_mass=True, return_clip=False):
    """g_fwd(R) [m/s^2] at projected radii R_arr [Mpc] for an acceleration g_of_r(r_Mpc)."""
    r = np.geomspace(p["r_min"], p["r_max"], p["n_r"])
    M = (r * h.MPC_M) ** 2 * g_of_r(r) / h.G_SI / h.MSUN_KG
    rho_raw = np.gradient(M, r) / (4.0 * np.pi * r ** 2)
    n_clip = int(np.sum(rho_raw[r <= p["z_max"]] <= 0.0))
    good = rho_raw > 0.0
    spl = interp1d(np.log(r[good]), np.log(rho_raw[good]), kind="cubic", bounds_error=False,
                   fill_value=(np.log(rho_raw[good][0]), np.log(rho_raw[good][-1])))
    rho_at = lambda x: float(np.exp(spl(np.log(x))))

    def sigma(R):
        f = lambda z: rho_at(np.sqrt(R * R + z * z))
        if not p["split"]:
            return 2.0 * quad(f, 0.0, p["z_max"], limit=500, epsrel=p["epsrel"])[0]
        br = sorted({0.0, min(10.0 * R, 1.0), 1.0, p["z_max"]})
        return 2.0 * sum(quad(f, a, b, limit=500, epsrel=p["epsrel"])[0] for a, b in zip(br[:-1], br[1:]))

    Rg = np.logspace(np.log10(p["R_lo"]), np.log10(np.max(R_arr) * 4.0), p["n_Rg"])
    Sg = np.array([sigma(R) for R in Rg]) / 1e12                       # Msun/pc^2
    Ss = interp1d(np.log(Rg), np.log(np.maximum(Sg, 1e-40)), kind="cubic",
                  bounds_error=False, fill_value="extrapolate")
    sig = lambda R: np.exp(Ss(np.log(R)))
    dS = np.array([2.0 / R ** 2 * np.trapezoid(Ri * sig(Ri), Ri) - sig(R)
                   for R in R_arr for Ri in [np.linspace(p["R_lo"], R, p["n_int"])]])
    g = ESD2G * dS
    if point_mass:
        g = g + ESD2G * mbar / (np.pi * np.asarray(R_arr) ** 2 * 1e12)
    return (g, n_clip) if return_clip else g


# ── Models: g(r) for r in Mpc ─────────────────────────────────────────────────
def g_hmg(mbar, s):
    return lambda r: h.gobs_hmg(np.asarray(r) * h.MPC_TO_KPC, mbar, s)


def g_mond(mbar, a0):
    def f(r):
        gb = h.gbar_si(np.asarray(r), mbar)
        return gb / (1.0 - np.exp(-np.sqrt(gb / a0)))
    return f


def g_cdm(mbar, mh, zeta0):
    return lambda r: h.gobs_cdm(np.asarray(r) * h.MPC_TO_KPC, mbar, mh, zeta0)


def chi2(pred, b):
    return h._chi2_arr(pred, b)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(OUT, exist_ok=True)
    bins, mbars = h.BINS, h.MBAR
    mh = [h.mhalo_from_mstar(ms) for ms in h.MSTAR]
    s_bin = [h.fit_s_bin(b, m) for b, m in zip(bins, mbars)]
    zeta0 = h.fit_zeta0_global()
    N = sum(len(b) for b in bins)
    print(f"per-bin s = {[round(s, 4) for s in s_bin]},  zeta_0 = {zeta0:.3f}")

    models = {
        "HMG": [g_hmg(m, s) for m, s in zip(mbars, s_bin)],
        "MOND": [g_mond(m, A0_STAR) for m in mbars],
        "CDM": [g_cdm(m, x, zeta0) for m, x in zip(mbars, mh)],
    }
    k_bin = {"HMG": 1, "MOND": 0, "CDM": 0}
    nu_pool = {"HMG": N - len(bins), "MOND": N, "CDM": N - 1}

    rows, chi = [], {m: {"SIS": [], "FWD": []} for m in models}
    kernel = []
    for i, (b, mb) in enumerate(zip(bins, mbars)):
        R = b[:, 0]
        row = {"row": f"bin {i + 1}"}
        for name, gl in models.items():
            g_sis = gl[i](R)
            g_fwd = project(gl[i], R, mb)
            c_s, c_f = chi2(g_sis, b), chi2(g_fwd, b)
            chi[name]["SIS"].append(c_s); chi[name]["FWD"].append(c_f)
            row[f"{name}_SIS"] = c_s / (len(b) - k_bin[name])
            row[f"{name}_FWD"] = c_f / (len(b) - k_bin[name])
            if name == "HMG":
                ratio = g_fwd / g_sis
                g_abel = project(gl[i], R, mb, point_mass=False)
                mid, far, near = (R > 0.16) & (R < 0.8), R > 1.0, R < 0.1
                kernel.append({"bin": i + 1,
                               "ratio_min_0.16_0.8": ratio[mid].min(), "ratio_max_0.16_0.8": ratio[mid].max(),
                               "ratio_min_R_gt_1": ratio[far].min(), "ratio_max_R_gt_1": ratio[far].max(),
                               "ratio_min_R_lt_0.1": ratio[near].min(), "ratio_max_R_lt_0.1": ratio[near].max(),
                               "abel_only_min_R_lt_0.1": (g_abel / g_sis)[near].min(),
                               "R_of_abel_only_min": R[near][np.argmin((g_abel / g_sis)[near])]})
        rows.append(row)
        print(row)

    pooled = {"row": "pooled"}
    for name in models:
        for ker in ("SIS", "FWD"):
            pooled[f"{name}_{ker}"] = sum(chi[name][ker]) / nu_pool[name]
    rows.append(pooled)
    print(pooled)

    # Global single-parameter fits under each kernel (nu = 59).
    def total(make, x, fwd):
        return sum(chi2(project(make(m, x, i), b[:, 0], m) if fwd else make(m, x, i)(b[:, 0]), b)
                   for i, (b, m) in enumerate(zip(bins, mbars)))
    makers = {
        "HMG": (lambda m, x, i: g_hmg(m, x), (0.30, 0.70)),
        "MOND": (lambda m, x, i: g_mond(m, x * A0_STAR), (1.0, 2.5)),
        "CDM": (lambda m, x, i: g_cdm(m, mh[i], x), (5.0, 30.0)),
    }
    glob = {"row": "global fit"}
    for name, (make, bounds) in makers.items():
        for ker in ("SIS", "FWD"):
            res = minimize_scalar(lambda x: total(make, x, ker == "FWD"), bounds=bounds,
                                  method="bounded", options={"xatol": 1e-4})
            glob[f"{name}_{ker}"] = res.fun / (N - 1)
            glob[f"{name}_{ker}_param"] = res.x
            print(f"global {name} {ker}: param = {res.x:.4f}  chi2_nu = {res.fun / (N - 1):.3f}")
    rows.append(glob)

    keys = ["row"] + [f"{m}_{k}" for m in models for k in ("SIS", "FWD")] + \
           [f"{m}_{k}_param" for m in models for k in ("SIS", "FWD")]
    with open(os.path.join(OUT, "forward_lensing_table_A2.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{r[k]:.5g}" if isinstance(r.get(k), float) else r.get(k, "")) for k in keys})
    with open(os.path.join(OUT, "forward_lensing_kernel.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(kernel[0].keys()))
        w.writeheader()
        for r in kernel:
            w.writerow({k: (f"{v:.5g}" if isinstance(v, float) else v) for k, v in r.items()})
    print("ratio g_fwd/g_SIS (HMG), 0.16<R<0.8 Mpc: [%.4f, %.4f]" %
          (min(r["ratio_min_0.16_0.8"] for r in kernel), max(r["ratio_max_0.16_0.8"] for r in kernel)))


if __name__ == "__main__":
    main()
