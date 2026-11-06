"""Numerical convergence of the forward projection of Appendix A.4.

Repeats the projection of forward_lensing.py for HMG, MOND and CDM in the four mass
bins with every numerical setting refined at once (line-of-sight cut-off 200 -> 400
Mpc, radial grid of rho_eff 6000 -> 12000 points, Sigma(R) grid 100 -> 200 points,
Sigma_bar integral 1000 -> 4000 points, its inner limit 1 -> 0.5 kpc, quadrature
tolerance 1e-5 -> 1e-7, line-of-sight integral split at 10 R and 1 Mpc), and then
with one setting refined at a time. Reports the maximum change of g_fwd at the data
radii, the maximum relative change of the per-bin chi2_nu, and the number of grid
points with rho_eff <= 0 (set to zero by the projection).

Output: outputs/forward_projection_convergence.json
"""
import json
import os
import numpy as np

import hmg_model as h
import forward_lensing as F

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = dict(F.DEFAULTS)
FINE = dict(r_min=3e-4, r_max=600.0, n_r=12000, z_max=400.0, n_Rg=200,
            n_int=4000, R_lo=0.0005, epsrel=1e-7, split=True)
SINGLE = ["r_max", "n_r", "z_max", "n_Rg", "n_int", "R_lo", "epsrel"]


def main():
    s_bin = [h.fit_s_bin(b, m) for b, m in zip(h.BINS, h.MBAR)]
    zeta0 = h.fit_zeta0_global()
    out = []
    for i, (b, mb) in enumerate(zip(h.BINS, h.MBAR)):
        R = b[:, 0]
        mh = h.mhalo_from_mstar(h.MSTAR[i])
        for name, g in [("HMG", F.g_hmg(mb, s_bin[i])), ("MOND", F.g_mond(mb, F.A0_STAR)),
                        ("CDM", F.g_cdm(mb, mh, zeta0))]:
            g0, c0 = F.project(g, R, mb, BASE, return_clip=True)
            g1, c1 = F.project(g, R, mb, FINE, return_clip=True)
            x0, x1 = F.chi2(g0, b), F.chi2(g1, b)
            single = {}
            for k in SINGLE:
                p = dict(BASE, split=True)
                p[k] = FINE[k]
                if k == "z_max":
                    p["r_max"] = FINE["r_max"]
                single[k] = float(np.max(np.abs(F.project(g, R, mb, p) / g0 - 1.0)))
            row = dict(bin=i + 1, model=name,
                       max_rel_g_all=float(np.max(np.abs(g1 / g0 - 1.0))),
                       max_rel_g_016_08=float(np.max(np.abs(g1 / g0 - 1.0)[(R >= 0.16) & (R <= 0.8)])),
                       rel_change_chi2=float(abs(x1 / x0 - 1.0)),
                       clipped_points=c0, single_setting=single)
            out.append(row)
            print(row["bin"], name, f"dg/g = {row['max_rel_g_all']:.4f}  dchi2/chi2 = {row['rel_change_chi2']:.4f}  "
                  f"clipped = {c0}  dominant = {max(single, key=single.get)}", flush=True)
    os.makedirs(os.path.join(HERE, "outputs"), exist_ok=True)
    json.dump(out, open(os.path.join(HERE, "outputs", "forward_projection_convergence.json"), "w"), indent=1)
    print("max dg/g (all) = %.4f, HMG = %.4f; max dchi2/chi2 = %.4f" %
          (max(r["max_rel_g_all"] for r in out),
           max(r["max_rel_g_all"] for r in out if r["model"] == "HMG"),
           max(r["rel_change_chi2"] for r in out)))


if __name__ == "__main__":
    main()
