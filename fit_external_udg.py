"""HMG vicinity scale of the two gas-rich UDG samples of Table A.1 and Fig. A.2.

Gas-rich UDGs (Mancera Pina et al. 2022; six galaxies, one outer rotation velocity each;
data/external_reference/udg_manceraPina2022.txt, model of fit_hmg_udg.py):
  joint fit of one s on the s < 1 branch, chi2 in velocity space with asymmetric errors;
  interval from Delta chi2 = 1 (lower limit 0 when chi2 stays below chi2_min + 1).

MIGHTEE UDGs (Ponomareva et al. 2021; four galaxies with kinematics;
data/external_reference/mightee_udg_ponomareva2021.txt):
  M_bar = 1.33 M_HI + M*, r_sys = R_HI (Wang et al. 2016 HI size-mass relation),
  v_N = sqrt(G M_bar / r_sys); nominal velocity uncertainty +/- 8 km/s.
  Per galaxy, s solves v_HMG(s) = v_obs on the s < 1 branch; its range comes from scaling
  r_sys by 0.6 and 1.4 (+/- 40 per cent). The sample value is the median s, and the
  reduced chi2 is evaluated at that s with nu = N - 1.

r_eq = (G M_bar t0^2)^(1/3) as in Table A.1.

Output: outputs/udg_systems.csv (one row per sample, median and 16th-84th percentiles)
        outputs/mightee_per_galaxy.csv
"""
import csv
import math
import os
import numpy as np

import fit_hmg_udg as F

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
EXT = os.path.join(HERE, "data", "external_reference")
G_CODE = 4.30091727e-6          # (km/s)^2 kpc / Msun
T0 = 14.11                      # kpc / (km/s)
S_GRID = np.logspace(-5.0, 0.0, 4001)


def r_eq_mpc(mbar):
    return (G_CODE * mbar * T0 ** 2) ** (1.0 / 3.0) / 1000.0


def pct(a):
    a = np.asarray(a, float)
    return float(np.median(a)), float(np.percentile(a, 16)), float(np.percentile(a, 84))


def galaxy(name, log_mbar, r_kpc, v_obs, sp, sm, v_n):
    return F.Galaxy(name, log_mbar, r_kpc, v_obs, sp, sm, v_n, 0.0, 0.0)


def chi2_sample(gals, s):
    c = 0.0
    for g in gals:
        res = F.v_hmg_kms(g, s) - g.v_obs_kms
        c += (res / F.sigma_eff_kms(res, g)) ** 2
    return c


def s_match(g):
    """s < 1 at which v_HMG(s) = v_obs (closest grid value, refined by bisection)."""
    v = np.array([F.v_hmg_kms(g, float(s)) for s in S_GRID]) - g.v_obs_kms
    i = int(np.argmin(np.abs(v)))
    j = [k for k in (i - 1, i + 1) if 0 <= k < len(S_GRID) and v[k] * v[i] <= 0]
    if not j:
        return float(S_GRID[i])
    a, b = sorted((S_GRID[i], S_GRID[j[0]]))
    return F.bisect_root(lambda s: F.v_hmg_kms(g, s) - g.v_obs_kms, float(a), float(b))


def mancera_pina():
    gals = F.DATA
    c = np.array([chi2_sample(gals, float(s)) for s in S_GRID])
    i = int(np.argmin(c))
    lo, hi = S_GRID[max(i - 2, 0)], S_GRID[min(i + 2, len(S_GRID) - 1)]
    s_best, c_min = F.golden_section_minimum(lambda s: chi2_sample(gals, s), float(lo), float(hi), tol=1e-9)
    above = c > c_min + 1.0
    left = np.where(above[:i])[0]
    right = np.where(above[i:])[0]
    s_lo = float(S_GRID[left[-1]]) if left.size else 0.0
    s_hi = float(S_GRID[i + right[0]]) if right.size else 1.0
    req = [r_eq_mpc(10 ** g.log10_mbar) for g in gals]
    nu = len(gals) - 1
    return dict(sample="Gas-rich UDGs", N=len(gals), s=s_best, s_lo=s_lo, s_hi=s_hi,
                chi2=c_min, nu=nu, n_sigma=math.sqrt(c_min / nu),
                r_eq=pct(req)[0], r_eq_lo=pct(req)[1], r_eq_hi=pct(req)[2],
                m_bar=float(np.median([10 ** g.log10_mbar for g in gals])),
                r_sys_kpc=float(np.median([g.r_sys_kpc for g in gals])))


def mightee():
    rows, gals = [], []
    for ln in open(os.path.join(EXT, "mightee_udg_ponomareva2021.txt")):
        if ln.startswith("#") or not ln.strip():
            continue
        name, lmb, rk, vo, sp, sm, vn = ln.split()
        lmb, rk, vo, sp, sm = float(lmb), float(rk), float(vo), float(sp), float(sm)
        mbar = 10 ** lmb
        s_scaled = []
        for scale in (1.0, 0.6, 1.4):
            r = rk * scale
            s_scaled.append(s_match(galaxy(name, lmb, r, vo, sp, sm, math.sqrt(G_CODE * mbar / r))))
        gals.append(galaxy(name, lmb, rk, vo, sp, sm, math.sqrt(G_CODE * mbar / rk)))
        rows.append(dict(name=name, r_eq=r_eq_mpc(mbar), s=s_scaled[0],
                         s_lo=min(s_scaled), s_hi=max(s_scaled), m_bar=mbar, r_sys_kpc=rk))
    s_med, _, _ = pct([r["s"] for r in rows])
    req = pct([r["r_eq"] for r in rows])
    c = chi2_sample(gals, s_med)
    nu = len(gals) - 1
    summary = dict(sample="MIGHTEE gas-rich UDGs", N=len(gals), s=s_med,
                   s_lo=float(np.percentile([r["s"] for r in rows], 16)),
                   s_hi=float(np.percentile([r["s"] for r in rows], 84)),
                   chi2=c, nu=nu, n_sigma=math.sqrt(c / nu),
                   r_eq=req[0], r_eq_lo=req[1], r_eq_hi=req[2],
                   m_bar=float(np.median([r["m_bar"] for r in rows])),
                   r_sys_kpc=float(np.median([r["r_sys_kpc"] for r in rows])))
    return summary, rows


def main():
    os.makedirs(OUT, exist_ok=True)
    mp = mancera_pina()
    mi, per = mightee()
    with open(os.path.join(OUT, "udg_systems.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(mp.keys()))
        w.writeheader()
        for row in (mp, mi):
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in row.items()})
    with open(os.path.join(OUT, "mightee_per_galaxy.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(per[0].keys()))
        w.writeheader()
        for row in per:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in row.items()})
    for row in (mp, mi):
        print(f"{row['sample']:<24} s = {row['s']:.4f} [{row['s_lo']:.4f}, {row['s_hi']:.4f}]  "
              f"chi2_nu = {row['chi2'] / row['nu']:.2f} (nu = {row['nu']})  "
              f"n_sigma = {row['n_sigma']:.2f}  r_eq = {row['r_eq']:.4f} Mpc")
    for r in per:
        print(f"  {r['name']}  r_eq = {r['r_eq']:.4f}  s = {r['s']:.4f} [{r['s_lo']:.4f}, {r['s_hi']:.4f}]")


if __name__ == "__main__":
    main()
