"""
xcop_s_rnei_joint.py -- Joint (s, r_nei) fit for X-COP clusters.

Extends xcop_rnei_nsig.py (Model A, s=1) to a 2-parameter model where both
the neighbourhood amplitude s and the clip radius r_nei are free.

Three models compared
---------------------
  A  r_nei only, s=1 (xcop_rnei_nsig.py baseline, n_sigma=0.918)
  B  scalar s,   r_nei=inf (no clip; gives n_sigma=2.731)
  C  (s, r_nei) jointly -- this script

Key result
----------
The chi2_nu(s) profile (Model C, r_nei profiled out) is FLAT for s in [0.7, 10]:
the data cannot constrain s when r_nei is free.  The minimum is at s~0.9 with
n_sigma~0.87, barely better than Model A (0.916).  Conclusion: the r_nei model
(s=1) is the correct parsimonious parametrisation for X-COP; a joint fit does
not provide a meaningful s measurement.

The r_nei_best(s) ridge shows the degeneracy: larger s -> larger r_nei to
compensate, with a roughly power-law correlation r_nei ~ s^alpha per cluster.

Outputs
-------
  outputs/xcop_s_rnei_profile.png   -- chi2(s) profile + r_nei(s) correlation
  outputs/xcop_s_rnei_grid.csv      -- per-cluster best (s, r_nei, n_sigma) from 2D grid
  outputs/xcop_s_rnei_ridge.csv     -- median r_nei_best(s) across 12 clusters
"""
import sys
import numpy as np
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

try:                                   # Delta-chi2 / arrow glyphs crash on cp1252 (Windows) stdout
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE    = Path(__file__).resolve().parent
PROF    = HERE / "data" / "external_reference" / "xcop_profiles.txt"
OUT_PNG = HERE / "outputs" / "xcop_s_rnei_profile.png"
OUT_GRID= HERE / "outputs" / "xcop_s_rnei_grid.csv"
OUT_RIDGE=HERE / "outputs" / "xcop_s_rnei_ridge.csv"

# ── Constants (verbatim from xcop_rnei_nsig.py) ───────────────────────────────
c0      = 3e8
Msol    = 1.9891e30
kpc     = 3261.8116478174 * 365*24*3600 * c0
T0      = 13.7e9 * 365*24*3600
GN      = 6.674e-11
rho_vac = 3.0 / (8*np.pi*GN*T0**2)
GA_EMPTY  = np.pi/3
GA_CENTER = np.pi/2
R_GRAV    = 50.0
R_FIT_MIN = 1000.0

S_GRID    = np.logspace(-1, 1.2, 120)        # s from 0.1 to ~16
RNEI_GRID = np.arange(300, 6001, 100)        # r_nei 300–6000 kpc


def hmg_gamma0(q):
    g = np.arcsin(np.sqrt(np.sin(GA_EMPTY)**2 +
                          (np.sin(GA_CENTER)**2 - np.sin(GA_EMPTY)**2) * q))
    return g / np.cos(g)


def hmg_predict(M_enc_kg, R_ref_kpc, R0_kpc, s=1.0):
    """HMG acceleration with amplitude s and clip at R0_kpc.
    eps^2 = 1/6 + rho_enc/(s^3 * rho_vac).  s=1 reproduces xcop_rnei_nsig."""
    M  = np.asarray(M_enc_kg, float)
    Rm = np.asarray(R_ref_kpc, float) * kpc
    gN  = GN * M / Rm**2
    ve2 = 2 * gN * Rm
    vh2 = Rm**2 / T0**2
    dens = M / (4/3 * np.pi * Rm**3)
    eps0 = np.sqrt(1/6 + dens / (rho_vac * s**3))
    R    = np.asarray(R_ref_kpc)
    win  = (R > R_GRAV) & (R < R0_kpc)
    if not win.any():
        return np.full_like(gN, np.nan)
    eps1 = eps0.copy()
    eps1[R < R_GRAV]  = np.nanmax(eps0[win])
    eps1[R > R0_kpc]  = np.nanmin(eps0[win])
    q1 = np.abs(ve2 - vh2*eps1**2) / (vh2*eps1**2 + ve2)
    return np.sqrt(gN**2 + 2*gN*((c0/T0)/hmg_gamma0(q1)))


def chi2nu(arr, r_nei, s=1.0):
    """log-g reduced chi^2 (df=N-1) for one cluster."""
    R_IN,R_OUT,R_REF,M_DM,M_DM_LO,M_DM_HI,MGAS,MGAS_LO,MGAS_HI = arr.T
    ap  = hmg_predict(Msol*MGAS, R_REF, r_nei, s=s)
    obs = GN*Msol*(M_DM+MGAS) / (R_REF*kpc)**2
    with np.errstate(divide="ignore", invalid="ignore"):
        oHI = GN*Msol*(M_DM_HI+MGAS_HI) / (R_IN*kpc)**2
    oLO = GN*Msol*(M_DM_LO+MGAS_LO) / (R_OUT*kpc)**2
    err = (oHI - oLO) / 2
    m   = R_REF >= R_FIT_MIN
    N   = int(m.sum())
    if N < 2:
        return N, np.nan
    sl  = err[m] / (obs[m] * np.log(10))
    c2  = np.nansum(((np.log10(obs[m]) - np.log10(ap[m])) / sl)**2)
    return N, c2 / (N - 1)


def load_profiles(path):
    rnei, rows, order = {}, {}, []
    with open(path) as fh:
        for line in fh:
            s = line.strip()
            if s.startswith("#"):
                p = s.split()
                if len(p) >= 4 and p[1] == "rnei":
                    rnei[p[2]] = float(p[3])
                continue
            if not s:
                continue
            p = s.split(); c = p[0]
            if c not in rows:
                rows[c] = []; order.append(c)
            rows[c].append([float(x) for x in p[1:10]])
    return order, rnei, {c: np.asarray(rows[c], float) for c in order}


def main():
    order, rnei_pub, prof = load_profiles(PROF)
    HERE.parent  # suppress unused
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)

    # ── Model A: r_nei published, s=1 ────────────────────────────────────────
    ns_A = [np.sqrt(chi2nu(prof[c], rnei_pub[c], s=1.0)[1]) for c in order]
    med_A = float(np.median(ns_A))

    # ── 2D grid: per-cluster best (s, r_nei) ─────────────────────────────────
    best2d = {}   # cluster -> (s_best, r_best, ns_best, N, delta_chi2)
    for cname in order:
        arr = prof[cname]
        _, c2A = chi2nu(arr, rnei_pub[cname], s=1.0)
        N_ref = chi2nu(arr, rnei_pub[cname], s=1.0)[0]
        best_c2, best_s, best_r = np.inf, 1.0, rnei_pub[cname]
        for sv in S_GRID:
            for rv in RNEI_GRID:
                N, c2 = chi2nu(arr, rv, s=sv)
                if c2 < best_c2:
                    best_c2, best_s, best_r = c2, sv, rv
        dc2 = (c2A - best_c2) * (N_ref - 1)
        best2d[cname] = (best_s, best_r, np.sqrt(best_c2), N_ref, dc2)

    ns_C = [best2d[c][2] for c in order]
    med_C = float(np.median(ns_C))

    print(f"\n{'Cluster':<9} {'s_best':>8} {'r_nei_best':>11} {'n_sig_C':>9} {'Δχ²(A→C)':>10}")
    for c in order:
        sb, rb, ns, N, dc = best2d[c]
        print(f"{c:<9} {sb:>8.3f} {int(rb):>11} {ns:>9.3f} {dc:>10.1f}")
    print(f"\nMedian n_sigma: A={med_A:.3f}  C={med_C:.3f}")

    # ── chi2(s) profile: best r_nei per cluster per s, then sample median ────
    profile_C  = []   # joint model C
    profile_B  = []   # scalar-s, no clip
    ridge_med  = []   # median r_nei_best(s) across clusters
    ridge_p16  = []
    ridge_p84  = []

    for sv in S_GRID:
        ns_c, ns_b, r_bests = [], [], []
        for cname in order:
            arr = prof[cname]
            R_REF = arr[:, 2]
            # Model C: profile over r_nei
            best_c2c, best_rc = np.inf, rnei_pub[cname]
            for rv in RNEI_GRID:
                _, c2 = chi2nu(arr, rv, s=sv)
                if c2 < best_c2c:
                    best_c2c, best_rc = c2, rv
            ns_c.append(np.sqrt(best_c2c))
            r_bests.append(best_rc)
            # Model B: no clip
            _, c2b = chi2nu(arr, R_REF.max()*10, s=sv)
            ns_b.append(np.sqrt(c2b))
        profile_C.append(float(np.median(ns_c)))
        profile_B.append(float(np.median(ns_b)))
        ridge_med.append(float(np.median(r_bests)))
        ridge_p16.append(float(np.percentile(r_bests, 16)))
        ridge_p84.append(float(np.percentile(r_bests, 84)))

    profile_C = np.array(profile_C)
    profile_B = np.array(profile_B)
    s_best_C  = float(S_GRID[np.argmin(profile_C)])
    ns_best_C = float(np.min(profile_C))
    print(f"Profile minimum: s={s_best_C:.3f}, n_sigma={ns_best_C:.3f}")

    # ── Save CSVs ─────────────────────────────────────────────────────────────
    with open(OUT_GRID, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["cluster", "s_best", "r_nei_best_kpc", "n_sigma_C",
                    "n_sigma_A", "delta_chi2"])
        for c in order:
            sb, rb, nsc, N, dc = best2d[c]
            nsa = float(np.sqrt(chi2nu(prof[c], rnei_pub[c], s=1.0)[1]))
            w.writerow([c, f"{sb:.4f}", f"{int(rb)}", f"{nsc:.4f}",
                        f"{nsa:.4f}", f"{dc:.2f}"])
        w.writerow(["MEDIAN", "", "", f"{med_C:.4f}", f"{med_A:.4f}", ""])

    with open(OUT_RIDGE, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["s", "r_nei_med_kpc", "r_nei_p16_kpc", "r_nei_p84_kpc",
                    "n_sigma_C_med", "n_sigma_B_med"])
        for i, sv in enumerate(S_GRID):
            w.writerow([f"{sv:.5f}", f"{ridge_med[i]:.0f}",
                        f"{ridge_p16[i]:.0f}", f"{ridge_p84[i]:.0f}",
                        f"{profile_C[i]:.4f}", f"{profile_B[i]:.4f}"])

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Panel 1: chi2(s) profile
    ax1.plot(S_GRID, profile_C, color='#c8620a', lw=2.0,
             label=r'$(s,\,r_\mathrm{nei})$ jointly')
    ax1.plot(S_GRID, profile_B, color='#555', lw=1.2, ls='--',
             label=r'scalar $s$ (no clip)')
    ax1.axhline(med_A, color='#1a7abf', lw=1.5, ls=':',
                label=fr'$r_\mathrm{{nei}}$ only ($s=1$): $n_\sigma={med_A:.3f}$')
    ax1.axhline(1.0, color='gray', lw=0.6)
    ax1.axvline(1.0, color='gray', lw=0.6, ls='--')
    ax1.axvline(s_best_C, color='#c8620a', lw=0.8, ls=':')
    ax1.text(s_best_C*1.05, ns_best_C+0.1,
             fr'$s={s_best_C:.2f}$, $n_\sigma={ns_best_C:.3f}$',
             color='#c8620a', fontsize=8)
    ax1.set_xscale('log')
    ax1.set_xlabel(r'neighbourhood parameter $s$')
    ax1.set_ylabel(r'median $n_\sigma$ (12 clusters)')
    ax1.set_title(r'X-COP: $\chi^2_\nu(s)$ profile')
    ax1.set_ylim(0, 5); ax1.set_xlim(S_GRID[0], S_GRID[-1])
    ax1.legend(fontsize=8)

    # Panel 2: r_nei_best(s) — degeneracy ridge
    rm  = np.array(ridge_med)
    rp16= np.array(ridge_p16)
    rp84= np.array(ridge_p84)
    # only plot where profile is near the flat minimum (n_sigma < 2)
    good = profile_C < 2.0
    ax2.fill_between(S_GRID[good], rp16[good], rp84[good],
                     color='#c8620a', alpha=0.25, label='p16–p84')
    ax2.plot(S_GRID[good], rm[good], color='#c8620a', lw=2.0, label='median')
    ax2.axvline(1.0, color='gray', lw=0.8, ls='--', label='s=1')
    ax2.axvline(s_best_C, color='#c8620a', lw=0.8, ls=':')
    # reference: published r_nei values
    pub_med = float(np.median(list(rnei_pub.values())))
    ax2.axhline(pub_med, color='#1a7abf', lw=1.5, ls=':',
                label=fr'published median $r_{{nei}}$={pub_med:.0f} kpc')
    ax2.set_xscale('log')
    ax2.set_xlabel(r'$s$')
    ax2.set_ylabel(r'$r_\mathrm{nei,best}$ [kpc]')
    ax2.set_title(r'Degeneracy ridge $r_\mathrm{nei}(s)$ — X-COP')
    ax2.legend(fontsize=8)
    ax2.set_xlim(S_GRID[good][0], S_GRID[good][-1])

    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150)
    print(f"\nSaved: {OUT_PNG}")
    print(f"       {OUT_GRID}")
    print(f"       {OUT_RIDGE}")


if __name__ == "__main__":
    main()
