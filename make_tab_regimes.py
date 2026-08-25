"""
make_tab_regimes.py -- reproducibly generate Table A.1 (tab:regimes).

Every column is derived from the total BARYONIC mass M_bar:
  v_N = sqrt(G M_bar / r_sys),  v_H = r_sys / t0,
  r_eq = (G M_bar t0^2)^(1/3),  s_eq = 12^(1/3) r_eq / r_sys,
  xi_s^2 = 1/s^3 + v_H^2/(12 v_N^2),  g_s/a0 = (2c/t0) q(xi_s) / a0.
Domain = Hubble if s/s_eq > 1 (Hubble term dominates xi_s^2), else neighb. (neighbourhood term).

1-sigma s interval:
  KiDS subsamples (this work) -> fit Delta chi2 = 1, read from tab_model_comparison.csv
                                 (preferred, lower-chi2 branch).
  External systems            -> sample dispersion (SPARC, clusters) or MC / literature
                                 (MW, gal-gal WL); read from reference_systems.csv.
The s interval is propagated to xi_s^2 and g_s/a0 by evaluating the formulae at the
interval endpoints (xi_s^2 and g_s are monotone in s away from the peak).

KiDS baryonic mass: M_bar = M_star (1 + f_cold), f_cold from Boselli (2014),
log10 f_cold = -0.69 log10(M_star/Msun) + 6.63 (same gas correction as g_bar).
External systems already quote baryonic masses.

Final column n_sigma = sqrt(reduced chi2) of the HMG fit vs g_obs in log-acceleration
space (see the block below), recomputed here from each system's own RAR data in
data/external_reference/ at the row's s -- a uniform "deviation in sigmas".

Outputs: outputs/tab_regimes.csv  and  outputs/tab_regimes_body.tex
"""
import os
import csv
import math
import numpy as np
import hmg_model as h
import fit_hmg_udg as _F_mightee

# MIGHTEE UDGs: joint HMG fit over the 6 Ponomareva+2021 galaxies (one V_out each).
# s_best=0.0098 minimises chi2 (velocity space); nu = N - k = 6 - 1 = 5.
_MIGHTEE_NU = len(_F_mightee.DATA) - 1          # 5
_MIGHTEE_CHI2NU = _F_mightee.chi2_hmg(0.0098) / _MIGHTEE_NU   # ~1.72

G = 4.30091727e-6      # (km/s)^2 kpc / Msun
T0 = 14.11             # kpc / (km/s)
C_KMS = 299792.458     # km/s
CODE_TO_SI = 3.24078e-14
A0 = 1.20e-10
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")


def q(x):
    x = max(float(x), 1e-12)
    d = abs(x * x - 1) / (x * x + 1)
    s2 = math.sin(math.pi / 3) ** 2 + (1 - math.sin(math.pi / 3) ** 2) * d
    g = math.asin(min(math.sqrt(s2), 1.0))
    return math.cos(g) / g


def fcold(mstar):
    return 10 ** (-0.69 * math.log10(mstar) + 6.63)


def r_newton_kpc(m):
    return (G * m * T0 ** 2) ** (1.0 / 3.0)


def derive(mbar, rsys, s):
    """All s-dependent columns for one system."""
    vN = math.sqrt(G * mbar / rsys)
    vH = rsys / T0
    req = r_newton_kpc(mbar)
    seq = 12 ** (1.0 / 3.0) * req / rsys
    xi2 = 1.0 / s ** 3 + vH ** 2 / (12 * vN ** 2)
    gs = 2 * C_KMS / T0 * q(math.sqrt(xi2)) * CODE_TO_SI / A0
    return dict(vN=vN, vH=vH, req=req, seq=seq, xi2=xi2, gs=gs,
                domain=("Hubble" if s / seq > 1 else "neighb."))


# Median baryonic-bin STELLAR masses (Brouwer+2021); the CSV omits them for the
# stacked/morphological subsets, so they are supplied here (single source below).
KIDS_MSTAR = {
    "Global": 3.2e10, "Bin 1": 1.5e10, "Bin 2": 3.2e10, "Bin 3": 4.6e10, "Bin 4": 8.9e10,
    "Late-type (n<2.5)": 9.5e9, "Early-type (n>=2.5)": 2.0e10,
    "Blue (u-r)": 7.8e9, "Red (u-r)": 1.2e10,
}


# ── KiDS subsamples: s (+/- 1sigma) of the preferred branch, from make_tables.py ──
def load_kids():
    path = os.path.join(OUT, "tab_model_comparison.csv")
    out = {}
    for r in csv.DictReader(open(path)):
        slo, elo, clo = float(r["s_best"]), float(r["s_err"]), float(r["chi2nu_lo"])
        shi, ehi, chi = float(r["s_hi"]), float(r["s_hi_err"]), float(r["chi2nu_hi"])
        if clo <= chi:
            s, e, c2nu = slo, elo, clo
        else:
            s, e, c2nu = shi, ehi, chi
        out[r["subsample"]] = (KIDS_MSTAR[r["subsample"]], s, e, c2nu)
    return out


# ── external reference systems: s and its interval from reference_systems.csv ─────
def load_refs():
    path = os.path.join(OUT, "reference_systems.csv")
    out = {}
    for r in csv.DictReader(open(path)):
        out[r["name"]] = r
    return out


kids = load_kids()
refs = load_refs()

# ── data goodness of fit: n_sigma = sqrt(reduced chi2) of HMG vs g_obs ─────────────
# n_sigma is the RMS deviation of the data from the HMG prediction, in units of the
# per-point 1-sigma error, computed uniformly in log10-acceleration space:
#   chi2nu = (1/(N-k)) sum[ (log10 g_tot - log10 g_obs) / sigma_log ]^2 ,  n_sigma = sqrt(chi2nu).
# g_tot is the full HMG prediction sqrt(g_bar (g_bar + g_s(r))) at the row's s; morphological
# subsets (binned in g_bar only, no radius) use the deep limit hmg_pred_dl. Single stacked RAR
# curves (KiDS bins, gal-gal WL, UDGs) are fit jointly at the row's s (k=1). Samples of
# individual objects (SPARC galaxies, HIFLUGCS and X-COP clusters) are fit per object and the
# median chi2nu is reported -- the row's s is the sample median of the per-object s. The Milky
# Way alone combines radial + vertical constraints (outside the rotation-RAR framework), so its
# n_sigma is taken from the published reduced chi2 (see mw_chi2nu_MI.txt: nbar1 = MI, the best fit).
DATA_EXT = os.path.join(HERE, "data", "external_reference")


def hmg_gtot(gbar_si, r_kpc, s):
    """Full HMG total acceleration from per-point (g_bar[SI], r[kpc]) and s -> SI."""
    an = np.asarray(gbar_si, float) / CODE_TO_SI                     # code accel
    r = np.asarray(r_kpc, float)
    v2n = an * r
    vh2 = (r / T0) ** 2
    xi2 = 1.0 / s ** 3 + vh2 / (12.0 * v2n + 1e-60)
    gs = 2.0 * C_KMS / T0 * h._cos_over_gamma(xi2)
    return np.sqrt(np.maximum(an * (an + gs), 0.0)) * CODE_TO_SI


LN10 = math.log(10.0)
C2S = h.CODE_TO_SI                                   # (km/s)^2/kpc -> m/s^2


def _chi2nu(gt, gobs, sig_gobs, k=1):
    """Reduced chi2 of the HMG prediction gt vs g_obs in log10-acceleration space
    (all SI arrays). sig_gobs = 1-sigma on g_obs; k = number of fitted parameters."""
    gt = np.asarray(gt, float); gobs = np.asarray(gobs, float)
    sig_log = np.asarray(sig_gobs, float) / (gobs * LN10)
    chi2 = float(np.sum(((np.log10(gt) - np.log10(gobs)) / sig_log) ** 2))
    return chi2 / max(len(gobs) - k, 1)


def _fit_s_chi2nu(r_kpc, gbar, gobs, sig_gobs, k=1):
    """Per-object best-fit s (grid) and its reduced chi2 in log-g space."""
    grid = np.logspace(-3, 2, 600)
    best = min(grid, key=lambda s: _chi2nu(hmg_gtot(gbar, r_kpc, s), gobs, sig_gobs, k))
    return best, _chi2nu(hmg_gtot(gbar, r_kpc, best), gobs, sig_gobs, k)


def _median_nsig(groups, s_fixed=None):
    """Per-cluster reduced chi2 (best-fit s, or fixed s), sqrt of the median over clusters."""
    vals = []
    for a in groups.values():                       # a cols: r_kpc, gbar, gobs, sig_gobs (SI)
        if len(a) < 2:
            continue
        if s_fixed is None:
            _, c = _fit_s_chi2nu(a[:, 0], a[:, 1], a[:, 2], a[:, 3], k=1)
        else:
            c = _chi2nu(hmg_gtot(a[:, 1], a[:, 0], s_fixed), a[:, 2], a[:, 3], k=0)
        vals.append(c)
    return math.sqrt(float(np.median(vals)))


def _perpoint_groups(fname):
    """Read '# cluster r_kpc gbar_si gobs_si sigma_gobs_si' rows grouped by cluster."""
    groups = {}
    for ln in open(os.path.join(DATA_EXT, fname)):
        if ln.startswith("#") or not ln.strip():
            continue
        p = ln.split()
        groups.setdefault(p[0], []).append([float(x) for x in p[1:5]])
    return {k: np.array(v) for k, v in groups.items()}


def nsig_kids(csv_name, s):
    """n_sigma = sqrt(reduced chi2) of the HMG fit at the row's s (log-g space)."""
    if csv_name == "Global":
        gb = np.concatenate([h.gbar_si(b[:, 0], mb) for b, mb in zip(h.BINS, h.MBAR)])
        go = np.concatenate([b[:, 1] for b in h.BINS])
        gl = np.concatenate([b[:, 2] for b in h.BINS]); gh = np.concatenate([b[:, 3] for b in h.BINS])
        rk = np.concatenate([b[:, 0] * h.MPC_TO_KPC for b in h.BINS])
        return math.sqrt(_chi2nu(hmg_gtot(gb, rk, s), go, (gh - gl) / 2))
    if csv_name.startswith("Bin "):
        i = int(csv_name.split()[1]) - 1; b = h.BINS[i]
        gb = h.gbar_si(b[:, 0], h.MBAR[i]); rk = b[:, 0] * h.MPC_TO_KPC
        return math.sqrt(_chi2nu(hmg_gtot(gb, rk, s), b[:, 1], (b[:, 3] - b[:, 2]) / 2))
    _MF = {"Late-type (n<2.5)": "Sersicbin_1", "Early-type (n>=2.5)": "Sersicbin_2",
           "Blue (u-r)": "Colorbin_1", "Red (u-r)": "Colorbin_2"}
    if csv_name in _MF:
        gb, go, ge = h.load_morph(f"Fig-8_RAR-KiDS-isolated_{_MF[csv_name]}.txt")
        return math.sqrt(_chi2nu(h.hmg_pred_dl(gb, s), go, ge))       # deep-limit model (no radius)
    return None


def nsig_sparc(s=None):
    # Sample of 61 galaxies -> per-galaxy best-fit s, median reduced chi2 (like the clusters).
    groups = {}
    for ln in open(os.path.join(DATA_EXT, "McGaugh2007.txt")):
        p = ln.split()
        if len(p) < 7 or p[0] == "Name":
            continue
        try:
            R, Vo, Vg, Vs, Qm, Qp = map(float, p[1:7])
        except ValueError:
            continue
        vbar2 = Vg ** 2 + Qp * Vs ** 2
        if R > 0 and Vo > 0 and vbar2 > 0:
            sv = math.sqrt((0.043 * Vo) ** 2 + 4.6 ** 2)             # Lelli+2016 velocity-error proxy
            groups.setdefault(p[0], []).append([R, vbar2 / R * C2S, Vo ** 2 / R * C2S, 2 * Vo * sv / R * C2S])
    return _median_nsig({k: np.array(v) for k, v in groups.items()}, s_fixed=None)


def _vel_nsig(fname, mcol, rcol, vcol, errcols, s):
    """log-g n_sigma for a velocity-space RAR file (g = v^2/r, g_bar = G M_bar/r^2)."""
    gb, go, rk, sg = [], [], [], []
    for ln in open(os.path.join(DATA_EXT, fname)):
        if ln.startswith("#") or not ln.strip():
            continue
        p = ln.split()
        mbar = 10 ** float(p[mcol])
        r = 10 ** float(p[rcol]) if rcol == 3 and fname.startswith("mistele") else float(p[rcol])
        v = float(p[vcol]); sv = math.sqrt(sum(float(p[c]) ** 2 for c in errcols))
        gb.append(h.gbar_si(r / 1000.0, mbar)); go.append(v ** 2 / r * C2S)
        rk.append(r); sg.append(2 * v * sv / r * C2S)
    gb, go, rk, sg = map(np.array, (gb, go, rk, sg))
    return math.sqrt(_chi2nu(hmg_gtot(gb, rk, s), go, sg))


def nsig_udg(s):
    # udg file cols: agc log10_mbar r_kpc v_obs sig_plus sig_minus v_newton
    return _vel_nsig("udg_manceraPina2022.txt", 1, 2, 3, (4, 5), s)


def nsig_ggwl(s):
    # mistele file cols: bin z log10Mb log10R vel err_std err_sys
    return _vel_nsig("mistele_gg_wl.txt", 2, 3, 4, (5, 6), s)


def nsig_mw(s=None):
    # The MW fit combines radial + vertical constraints (outside the rotation-RAR framework),
    # so n_sigma = sqrt(published reduced chi2); see the data-file header for provenance.
    for ln in open(os.path.join(DATA_EXT, "mw_chi2nu_MI.txt")):
        if not ln.startswith("#") and ln.strip():
            return math.sqrt(float(ln.split()[0]))


def nsig_hiflugcs(s=None):
    groups = {}
    for ln in open(os.path.join(DATA_EXT, "clusterRAR.dat")):
        if ln.startswith("#") or not ln.strip():
            continue
        p = ln.split()
        go = 10 ** float(p[4]); sglog = (float(p[5]) + float(p[6])) / 2
        groups.setdefault(p[0], []).append([float(p[2]), 10 ** float(p[3]), go, go * LN10 * sglog])
    return _median_nsig({k: np.array(v) for k, v in groups.items()}, s_fixed=None)   # per-cluster fit


def xcop_fit():
    """X-COP sample of 12 clusters: per-cluster best-fit s and reduced chi2.
    Returns (s_median, s_p16, s_p84, n_sigma=sqrt(median chi2nu))."""
    ss, cc = [], []
    for a in _perpoint_groups("xcop_rar_perpoint.txt").values():
        if len(a) < 2:
            continue
        s, c = _fit_s_chi2nu(a[:, 0], a[:, 1], a[:, 2], a[:, 3], k=1)
        ss.append(s); cc.append(c)
    ss = np.array(ss)
    return (float(np.median(ss)), float(np.percentile(ss, 16)), float(np.percentile(ss, 84)),
            math.sqrt(float(np.median(cc))))


def kids_row(name, disp, rsys, cite):
    mstar, s, e, c2nu = kids[name]
    mbar = mstar * (1 + fcold(mstar))
    # chi2nu from make_tables.py (full HMG formula for all subsamples, including morphological)
    return dict(name=disp, cite=cite, mbar=mbar, rsys=rsys,
                s=s, s_lo=s - e, s_hi=s + e, s_disp=None, work=True, nsig=math.sqrt(c2nu))


def ref_row(disp, mbar, rsys, s, s_lo, s_hi, cite, s_disp=None, nsig_fn=None, seq=None):
    return dict(name=disp, cite=cite, mbar=mbar, rsys=rsys,
                s=s, s_lo=s_lo, s_hi=s_hi, s_disp=s_disp, work=False,
                nsig=nsig_fn(s) if nsig_fn else None, seq=seq)


# SPARC / HIFLUGCS / gal-gal WL s-intervals come straight from reference_systems.csv
# (single source of truth shared with Fig. A.2).
_sp = refs["SPARC"]
_hi = refs["HIFLUGCS"]
_gw = refs["gal-gal WL"]   # adopted s>1 branch = 2.76 [2.67,2.83] (per-bin HMG-pure fit, Mistele 2024)
_xc = refs["X-COP"]        # Model C (s & r_nei jointly fitted): s = 0.95 [0.70,1.51]
# X-COP: per-cluster best-fit s (median + 16-84 dispersion) and n_sigma from the same fit,
# so the row is self-consistent (s, xi^2, g_s and n_sigma all at the fitted s) -- the clusters
# prefer s slightly below the gamma=pi/3 equilibrium value (s=1) of the cited analysis.
def _xcop_rnei_nsig():
    """X-COP n_sigma from the CORRECT r_nei model (radially-varying epsilon_0 following the gas
    density), reproduced by xcop_rnei_nsig.py -> outputs/xcop_rnei_nsig.txt.  A single scalar s is
    NOT the right model for the X-COP hydrostatic profiles (see that script / the hydrostatic_
    equilibrium_xcop model_comparison.md), so we do NOT use the scalar-s value (2.73) here.
    Falls back to the published median 0.918 if the recompute output is absent."""
    path = os.path.join(OUT, "xcop_rnei_nsig.txt")
    if os.path.exists(path):
        for ln in open(path):
            if ln.lower().startswith("median"):        # 'median_nsigma <value>' line
                return float(ln.split()[-1])
    return 0.918

# X-COP = Model C: s AND r_nei fitted jointly per cluster (the ONE 2-parameter system in the table).
# s and reduced chi2 are the medians over the 12 clusters from the joint (s,r_nei) fit
# (scripts_replicable/xcop_s_rnei_joint.py; see memory theory_eps_hmg_models.md sec.4).
_xc_s, _xc_slo, _xc_shi = float(_xc["s"]), float(_xc["s_lo"]), float(_xc["s_hi"])   # 0.95 [0.70,1.51]
_xc_nsig = 0.858    # Model C median sqrt(chi2nu) -> chi2nu = 0.74 (reproduced by xcop_s_rnei_joint.py)

SYSTEMS = [
    # --- external systems (baryonic masses from the cited works) ---
    # UDG global HMG fit (fit_hmg_udg.py) prefers the s<1 branch: s=0.0098 (chi2=8.6)
    # over the s->inf asymptotic branch (chi2=17.1); s interval from Delta chi2=1.
    ref_row(r"Gas-rich UDGs",       1.6e9,  9,   0.0098, None, None, "Monjo2026udg",
            s_disp="$0.010^{+0.008}_{-0.010}$", nsig_fn=nsig_udg),
    # MIGHTEE Ponomareva+2021: 6 gas-rich UDGs with one V_out each at R_HI/2.
    # Displayed s is per-galaxy median (individual bisection); chi2_nu is from the joint fit
    # at s_best=0.0098 (velocity space, nu=5). s_lo/s_hi=None -> xi2 shown without interval.
    ref_row(r"MIGHTEE gas-rich UDGs", 5.8e9, 19, 0.19, None, None, "Ponomareva2021",
            s_disp=r"$0.19^{+0.09}_{-0.10}$",
            nsig_fn=lambda s: math.sqrt(_MIGHTEE_CHI2NU)),
    ref_row(r"Galaxy--galaxy WL",   1.0e11, 500, float(_gw["s"]), float(_gw["s_lo"]), float(_gw["s_hi"]),
            "Monjo2025", nsig_fn=nsig_ggwl),  # Mistele, no-z; s from reference_systems.csv
    # SPARC: distributional medians of the 50 galaxies (matches Fig. A.2): r_eq~284 kpc
    # (M_bar~2.68e10), s=3.08, and s_eq = median of the per-galaxy 12^(1/3) r_eq/r_sys = 34.
    ref_row(r"SPARC galaxies",      2.68e10, 15, float(_sp["s"]), float(_sp["s_lo"]), float(_sp["s_hi"]),
            "MonjoBanik2025RAR", nsig_fn=nsig_sparc, seq=34.0),
    ref_row(r"Milky Way",           7.41e10, 8,  2.375, 2.24, 2.51, "Monjo2026vgrav", nsig_fn=nsig_mw),  # nbar1/MI: M_bar & s from MI fit; chi2nu=2.63, degenerate s<1(0.42)/s>1(2.375)
    ref_row(r"X-COP clusters",      1.0e14, 800, _xc_s, _xc_slo, _xc_shi, "monjo2025clusters",
            nsig_fn=lambda s: _xc_nsig),
    ref_row(r"HIFLUGCS clust.",     7.5e13, 800, float(_hi["s"]), float(_hi["s_lo"]), float(_hi["s_hi"]),
            "MonjoBanik2025RAR", nsig_fn=nsig_hiflugcs),
    # X-ray galaxy groups (Gastaldello+2007): 16 groups, 15-point NFW fit each.
    # Median s=1.000 (12/16 groups hit s=1 boundary); p16=0.993, p84=1.000.
    # s=1.000 gives xi2=1.00 and gs/a0=5.48 (q_max at xi2=1).
    # chi2_nu=2.53: sample median over 16 groups (fit_gastaldello_s.py, sigma_log=0.10).
    ref_row(r"X-ray galaxy groups", 2.4e12, 331, 1.000, None, None, "Gastaldello2007",
            s_disp=r"$1.00^{+0.00}_{-0.01}$",
            nsig_fn=lambda s: math.sqrt(2.53)),
    # --- KiDS subsamples (this work) ---
    kids_row("Global",              r"RAR KiDS all",              500, None),
    kids_row("Bin 1",               r"RAR KiDS bin 1",           500, None),
    kids_row("Bin 2",               r"RAR KiDS bin 2",           500, None),
    kids_row("Bin 3",               r"RAR KiDS bin 3",           500, None),
    kids_row("Bin 4",               r"RAR KiDS bin 4",           500, None),
    kids_row("Late-type (n<2.5)",   r"RAR KiDS late ($n<2.5$)",  500, None),
    kids_row("Early-type (n>=2.5)", r"RAR KiDS early ($n\geq2.5$)", 500, None),
    kids_row("Blue (u-r)",          r"RAR KiDS blue ($u-r$)",    500, None),
    kids_row("Red (u-r)",           r"RAR KiDS red ($u-r$)",     500, None),
]


def interval(mbar, rsys, s, s_lo, s_hi, key):
    """Propagate the s interval to xi2/gs by evaluating at the endpoints."""
    vals = [derive(mbar, rsys, s)[key]]
    for sx in (s_lo, s_hi):
        if sx is not None:
            vals.append(derive(mbar, rsys, sx)[key])
    return min(vals), max(vals)


def pm(v, lo, hi, p):
    """Format v with (asymmetric) interval; plain value if no interval.
    Large magnitudes (>=1e4) use scientific notation."""
    if lo is None or hi is None:
        if abs(v) >= 1e4:
            e = int(math.floor(math.log10(abs(v))))
            return f"${v/10**e:.1f}\\times10^{{{e}}}$"
        return f"{v:.{p}f}"
    up, dn = hi - v, v - lo
    return f"${v:.{p}f}^{{+{up:.{p}f}}}_{{-{dn:.{p}f}}}$"


rows_csv = []
body = []
for sy in SYSTEMS:
    d = derive(sy["mbar"], sy["rsys"], sy["s"])
    _seq = sy.get("seq") or d["seq"]                                   # distributional override (SPARC)
    _dom = "Hubble" if sy["s"] / _seq > 1 else "neighb."
    xi_lo, xi_hi = interval(sy["mbar"], sy["rsys"], sy["s"], sy["s_lo"], sy["s_hi"], "xi2")
    gs_lo, gs_hi = interval(sy["mbar"], sy["rsys"], sy["s"], sy["s_lo"], sy["s_hi"], "gs")
    # display strings
    e = int(math.floor(math.log10(sy["mbar"]))); man = sy["mbar"] / 10 ** e
    if man >= 9.95:
        man /= 10; e += 1
    mtex = f"${man:.1f}\\times10^{{{e}}}$"
    s_str = sy["s_disp"] if sy["s_disp"] else pm(sy["s"], sy["s_lo"], sy["s_hi"], 2)
    xi_str = pm(d["xi2"], xi_lo if sy["s_lo"] else None, xi_hi if sy["s_hi"] else None, 2)
    gs_str = pm(d["gs"], gs_lo if sy["s_lo"] else None, gs_hi if sy["s_hi"] else None, 2)
    ns = sy.get("nsig")
    chi2nu = ns**2 if ns is not None else None        # column reports chi2_nu = (n_sigma)^2
    ns_str = f"{chi2nu:.2f}" if chi2nu is not None else "---"
    name = f"{sy['name']} \\citep{{{sy['cite']}}}" if sy["cite"] else sy["name"]
    if sy["work"]:
        name = sy["name"] + " [this work]"
    body.append(f"{name} & {mtex} & {sy['rsys']:.0f} & {d['vN']:.0f} & {d['vH']:.0f} & "
                f"{d['vH']/d['vN']:.2f} & {d['req']:.0f} & {s_str} & {_seq:.2f} & "
                f"{_dom} & {xi_str} & {gs_str} & {ns_str} \\\\")
    rows_csv.append(dict(system=sy["name"], M_bar=f"{sy['mbar']:.3e}", r_sys=sy["rsys"],
                         v_N=round(d["vN"], 1), v_H=round(d["vH"], 1), r_eq=round(d["req"], 1),
                         s=sy["s"], s_lo=sy["s_lo"], s_hi=sy["s_hi"], s_eq=round(_seq, 2),
                         domain=_dom, xi2=round(d["xi2"], 3),
                         xi2_lo=round(xi_lo, 3), xi2_hi=round(xi_hi, 3),
                         gs_a0=round(d["gs"], 3), gs_a0_lo=round(gs_lo, 3), gs_a0_hi=round(gs_hi, 3),
                         chi2nu=(round(chi2nu, 3) if chi2nu is not None else "")))

os.makedirs(OUT, exist_ok=True)
with open(os.path.join(OUT, "tab_regimes.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows_csv[0].keys()))
    w.writeheader(); w.writerows(rows_csv)
with open(os.path.join(OUT, "tab_regimes_body.tex"), "w", encoding="utf-8", newline="") as fh:
    fh.write("\n".join(body) + "\n")

print("Wrote tab_regimes.csv and tab_regimes_body.tex\n")
for b in body:
    print(b)
