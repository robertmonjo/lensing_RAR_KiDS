"""
make_figA2_landscape.py -- Fig. A.2 of the paper (double-column):
  Panel (a) — (r_eq, s) dynamical-regime map (y=s, x=r_eq)          [LEFT]
  Panel (b) — chi2_nu(s) landscape rotated 90 deg (y=s, x=chi2_nu)  [RIGHT]

Both panels share the y-axis (s), linking each subsample's chi2 minimum to its
location in the regime map.  KiDS data and baryonic masses are imported from
hmg_model.py; external reference systems are read from reference_systems.csv.
"""
import os, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe

# ── paths ──────────────────────────────────────────────────────────────────────
DATA_DIR    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
DST         = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "figA2_landscape.png")
os.makedirs(os.path.dirname(DST), exist_ok=True)

# ── constants ──────────────────────────────────────────────────────────────────
G_CODE      = 4.30091727e-6   # (km/s)^2 kpc^-1 Msun^-1
T0          = 14.11            # kpc (km/s)^-1
C_KMS       = 299792.458       # km/s
CODE_TO_SI  = 3.24078e-14      # (km/s)^2 kpc^-1 → m s^-2
MPC_TO_KPC  = 1000.0
ESD2G       = 4.0 * 4.52e-30 * 3.086e16   # Msun/pc^2 → m/s^2
SIN2_GU     = math.sin(math.pi / 3) ** 2

# ── KiDS stellar-mass bins (inline; Brouwer+2021) ─────────────────────────────
MSTAR = [1.5e10, 3.2e10, 4.6e10, 8.9e10]

def f_cold(m):
    """Cold-gas fraction (Boselli 2014): log10 f_cold = -0.69 log10(M*) + 6.63."""
    return 10.0**(-0.69*math.log10(m)+6.63)
MBAR = [m*(1.0+f_cold(m)) for m in MSTAR]   # baryonic point mass
# The KiDS ESD profiles (BINS), stellar masses (MSTAR) and baryonic masses (MBAR)
# come from hmg_model.py -- the single source of truth shared with Table 1 / Fig. 2.
# An inline copy of BINS used to live here but was always overwritten by the import
# below; it was removed to avoid a second, divergent copy of the data.
import hmg_model as _hm
BINS, MSTAR, MBAR = _hm.BINS, _hm.MSTAR, _hm.MBAR

# ── HMG functions (from hmg_kids_landscape.py) ────────────────────────────────
def _q(x):
    x = np.maximum(np.asarray(x, float), 1e-12)
    delta = np.abs(x*x - 1) / (x*x + 1)
    s2 = math.sin(math.pi/3)**2 + (1 - math.sin(math.pi/3)**2) * delta
    gamma = np.arcsin(np.sqrt(np.clip(s2, 0, 1)))
    return np.cos(gamma) / np.maximum(gamma, 1e-12)

def hmg_gobs(r_kpc, ms, s):
    r   = np.asarray(r_kpc, float)
    v2n = G_CODE * ms / r
    vh  = r / T0
    eps = np.sqrt(2*v2n / (s**3 * vh**2) + 1.0/6.0)
    x   = np.sqrt(2*v2n) / (eps * vh)
    q   = _q(x)
    gn  = v2n / r
    return np.sqrt(gn * (gn + 2*C_KMS/T0*q)) * CODE_TO_SI

def chi2_all(s):
    tot = 0.0; n = 0
    for b, ms in zip(BINS, MBAR):
        r_kpc  = b[:, 0] * MPC_TO_KPC
        g_d    = b[:, 1]
        g_h    = hmg_gobs(r_kpc, ms, s)
        sig_up = np.log10(b[:, 3]) - np.log10(g_d)
        sig_dn = np.log10(g_d) - np.log10(b[:, 2])
        sig    = np.where(g_h >= g_d, sig_up, sig_dn)
        tot   += np.sum(((np.log10(g_d) - np.log10(g_h)) / sig)**2)
        n     += len(b)
    return float(tot) / (n - 1)          # chi2_nu = chi2/(N-k), k=1 (fitted s)

# ── Morphology-specific r(g_bar) from MICE (breaks deep-limit s<->1/s symmetry) ─
import re as _re
def _load_morph_r(key):
    blocks = {}; cur = None
    with open(os.path.join(DATA_DIR, "morph_gbar_r_mice.txt")) as _fh:
        for ln in _fh:
            m = _re.match(r"### (\w+)", ln)
            if m:
                cur = m.group(1); blocks[cur] = []; continue
            if cur and ln[:1].isdigit():
                a = ln.split()
                if len(a) == 2:
                    blocks[cur].append((float(a[0]), float(a[1])))
    arr = np.array(blocks[key]); r = arr[:, 0] * MPC_TO_KPC; lgb = arr[:, 1]
    o = np.argsort(lgb)
    return lambda gbar_si: np.interp(np.log10(gbar_si), lgb[o], r[o])

_MKEY = {"late": "LATE_sersic", "early": "EARLY_sersic",
         "blue": "BLUE_color", "red": "RED_color"}
_r_of = {k: _load_morph_r(v) for k, v in _MKEY.items()}

def chi2_morph(s, gbar, gobs, gerr, key):
    # Full xi_s^2 = 1/s^3 + v_H^2/(12 v_N^2), r per point from the MICE morphology
    # mapping -> the Hubble term breaks the s <-> 1/s deep-limit degeneracy.
    rk   = _r_of[key](gbar)
    gc   = gbar / CODE_TO_SI
    v2n  = gc * rk
    vh2  = (rk / T0)**2
    xi2  = 1.0/s**3 + vh2/(12.0*v2n + 1e-60)
    ge   = 2.0 * C_KMS / T0 * _q(np.sqrt(xi2)) * CODE_TO_SI
    pred = np.sqrt(gbar * (gbar + ge))
    sig  = gerr / (gobs * math.log(10))
    res  = (np.log10(gobs) - np.log10(pred)) / sig
    return float(np.sum(res**2)) / (len(gbar) - 1)

# ── Load morphological data ───────────────────────────────────────────────────
def _load_morph(fname):
    d    = np.loadtxt(os.path.join(DATA_DIR, fname))
    bias = d[:, 4]   # per-point (matches hmg_model.py; constant in the current files)
    gbar = d[:, 0]
    gobs = ESD2G * d[:, 1] / bias
    gerr = ESD2G * d[:, 3] / bias
    ok   = gobs > 0
    return gbar[ok], gobs[ok], gerr[ok]

_gb_late,  _go_late,  _ge_late  = _load_morph("Fig-8_RAR-KiDS-isolated_Sersicbin_1.txt")
_gb_early, _go_early, _ge_early = _load_morph("Fig-8_RAR-KiDS-isolated_Sersicbin_2.txt")
_gb_blue,  _go_blue,  _ge_blue  = _load_morph("Fig-8_RAR-KiDS-isolated_Colorbin_1.txt")
_gb_red,   _go_red,   _ge_red   = _load_morph("Fig-8_RAR-KiDS-isolated_Colorbin_2.txt")

def chi2_bin(s, ib):
    b      = BINS[ib]; ms = MBAR[ib]
    r_kpc  = b[:, 0] * MPC_TO_KPC
    g_d    = b[:, 1]
    g_h    = hmg_gobs(r_kpc, ms, s)
    sig_up = np.log10(b[:, 3]) - np.log10(g_d)
    sig_dn = np.log10(g_d) - np.log10(b[:, 2])
    sig    = np.where(g_h >= g_d, sig_up, sig_dn)
    return float(np.sum(((np.log10(g_d) - np.log10(g_h)) / sig)**2)) / (len(b) - 1)   # N-k, k=1

# ── Cluster chi2_nu(s) profiles (median over clusters): full xi_s^2, log-g space ─
def _chi2_cluster_one(rk, gbar, gobs, gerr, s):
    gc  = gbar / CODE_TO_SI
    v2n = gc * rk
    vh2 = (rk / T0)**2
    xi2 = 1.0/s**3 + vh2/(12.0*v2n + 1e-60)
    ge  = 2.0 * C_KMS / T0 * _q(np.sqrt(xi2)) * CODE_TO_SI
    pred = np.sqrt(gbar * (gbar + ge))
    sig  = gerr / (gobs * math.log(10))
    return float(np.sum(((np.log10(gobs) - np.log10(pred)) / sig)**2)) / max(len(gbar) - 1, 1)

def _load_cluster_groups(kind):
    g = {}
    if kind == "xcop":                                     # r gbar gobs sig (SI)
        for ln in open(os.path.join(DATA_DIR, "external_reference", "xcop_rar_perpoint.txt")):
            if ln.startswith("#") or not ln.strip():
                continue
            p = ln.split()
            g.setdefault(p[0], []).append((float(p[1]), float(p[2]), float(p[3]), float(p[4])))
    else:                                                  # HIFLUGCS: log columns + dex errors
        for ln in open(os.path.join(DATA_DIR, "external_reference", "clusterRAR.dat")):
            if ln.startswith("#") or not ln.strip():
                continue
            p = ln.split(); go = 10**float(p[4])
            gerr = go * math.log(10) * (float(p[5]) + float(p[6])) / 2
            g.setdefault(p[0], []).append((float(p[2]), 10**float(p[3]), go, gerr))
    return [np.array(v) for v in g.values()]

def chi2_cluster_curves(kind):
    """One reduced-chi2(s) profile per cluster (list of arrays over s_arr)."""
    grp = _load_cluster_groups(kind)
    return [np.array([_chi2_cluster_one(a[:, 0], a[:, 1], a[:, 2], a[:, 3], s) for s in s_arr])
            for a in grp]

# ── Compute chi2 landscape over s range matching panel axes ───────────────────
s_arr = np.logspace(np.log10(0.001), np.log10(600), 1500)  # extends past [0.002,500] axis so (b) curves exit cleanly
print("Computing chi2 landscape...", flush=True)
c_all   = np.array([chi2_all(s)   for s in s_arr])
c_bins  = [np.array([chi2_bin(s, i) for s in s_arr]) for i in range(4)]
c_late  = np.array([chi2_morph(s, _gb_late,  _go_late,  _ge_late,  "late")  for s in s_arr])
c_early = np.array([chi2_morph(s, _gb_early, _go_early, _ge_early, "early") for s in s_arr])
c_blue  = np.array([chi2_morph(s, _gb_blue,  _go_blue,  _ge_blue,  "blue")  for s in s_arr])
c_red   = np.array([chi2_morph(s, _gb_red,   _go_red,   _ge_red,   "red")   for s in s_arr])
print("Done.")
# ── UDG global chi2_nu(s): 6 gas-rich UDGs, joint HMG s<1 fit ─────────────
# fit_hmg_udg.py is bundled in this directory for standalone reproducibility.
import fit_hmg_udg as _F_udg
_c_udg_nu = len(_F_udg.DATA) - 1   # N - k = 6 - 1 = 5
_c_udg = np.array([_F_udg.chi2_hmg(float(s)) for s in s_arr]) / _c_udg_nu
print(f"UDG chi2_nu at s=0.0098: {_F_udg.chi2_hmg(0.0098)/_c_udg_nu:.3f}")

# ── s_equal functions ─────────────────────────────────────────────────────────
def Cseq(M):
    return (12.0 * G_CODE * M * T0**2) ** (1.0/3.0)   # kpc

def r_newton(M):
    return (G_CODE * M * T0**2) ** (1.0/3.0) / 1000.0  # Mpc

# ── Layout constants ───────────────────────────────────────────────────────────
s_min_ax  = 0.002
s_max_ax  = 100.0
r_min_ax  = 0.050
r_max_ax  = 5.50
M_REF     = 3.2e10
r_ref_kpc = 1000.0
hub       = (r_ref_kpc / T0)**2 / (12.0 * G_CODE * M_REF / r_ref_kpc)
s_cross   = hub ** (-1.0/3.0)
M_APPROX  = 3.2e10

print(f"hub={hub:.3f}  s_cross={s_cross:.3f}")

# ── Reference data ─────────────────────────────────────────────────────────────
bins_s = [
    {'label': r'Bin 1 ($1.5\!\times\!10^{10}\,M_\odot$)', 'M': 1.5e10, 'lc': '#3498db'},
    {'label': r'Bin 2 ($3.2\!\times\!10^{10}\,M_\odot$)', 'M': 3.2e10, 'lc': '#27ae60'},
    {'label': r'Bin 3 ($4.6\!\times\!10^{10}\,M_\odot$)', 'M': 4.6e10, 'lc': '#e67e22'},
    {'label': r'Bin 4 ($8.9\!\times\!10^{10}\,M_\odot$)', 'M': 8.9e10, 'lc': '#c0392b'},
]

extra = [
    {'label': 'Global',                  'M': M_APPROX, 'mk': 'D', 'mc': '#222222', 'ms': 9},
    {'label': r'Late ($n\!<\!2.5$)',     'M': 9.5e9,    'mk': '^', 'mc': '#2ecc71', 'ms': 11},
    {'label': r'Early ($n\!\geq\!2.5$)', 'M': 2.0e10,   'mk': 's', 'mc': '#8e44ad', 'ms': 8},
    {'label': r'Blue ($u\!-\!r$)',       'M': 7.8e9,    'mk': 'o', 'mc': '#2980b9', 'ms': 7},
    {'label': r'Red ($u\!-\!r$)',        'M': 1.2e10,   'mk': 'o', 'mc': '#e91e63', 'ms': 7},
]

stroke_eff = [pe.withStroke(linewidth=3.5, foreground='white')]
C_NB  = '#c03a2b'
C_HUB = '#505050'

# ── Actual curve minima per basin → markers AND cross-panel links ─────────────
# (markers/links are placed at the true argmin of each chi2 curve so they line up
#  exactly with the minima in panel (b), instead of hardcoded s_best values.)
def _bmin(curve, lo, hi):
    m = (s_arr >= lo) & (s_arr < hi)
    if not m.any():
        return None
    i = int(np.argmin(curve[m]))
    return float(s_arr[m][i]), float(curve[m][i])   # (s_at_min, chi2_min)

rn_proxy = r_newton(M_APPROX)
ELEMENTS = []   # each: color, marker, ms, label, map_x, lo=(s,chi2), hi=(s,chi2), stroke
for i, b in enumerate(bins_s):
    ELEMENTS.append(dict(color=b['lc'], mk='o', ms=8, label=f'Bin {i+1}',
                         map_x=r_newton(b['M']*(1+f_cold(b['M']))), stroke=False,
                         lo=_bmin(c_bins[i], s_min_ax, 1.0),
                         hi=_bmin(c_bins[i], 1.0, s_max_ax)))
for e, curve in [(extra[0], c_all), (extra[1], c_late), (extra[2], c_early),
                 (extra[3], c_blue), (extra[4], c_red)]:
    ELEMENTS.append(dict(color=e['mc'], mk=e['mk'], ms=e['ms'], label=e['label'],
                         map_x=r_newton(e['M']*(1+f_cold(e['M']))), stroke=(e['mk'] == '^'),
                         lo=_bmin(curve, s_min_ax, 1.0),
                         hi=_bmin(curve, 1.0, s_max_ax)))

# ── Symmetric branch relations forced to cross at s = 1 (regime change) ─────────
#   s_lo(r_eq) = k r_eq^{+P} ,  s_hi(r_eq) = (1/k) r_eq^{-P}  =>  s_lo s_hi = 1.
#   Exponent P = 1 is the physical "constant neighbourhood density" line: from
#   1/s^3 = (v_H^2/2v_N^2)(rho_s/rho_0) at fixed r, rho_s => s ∝ r_eq.
#   Fit the single amplitude k using ONLY the filled (preferred, lower-chi2) point
#   of each subsample.  A lower-basin point gives  ln k = ln s - P ln r_eq ;
#   an upper-basin point gives  ln k = -ln s - P ln r_eq  (mirror about s=1).
FIT_SET = 'bins'   # 'bins' (paper: s_low/s_high from the 4 mass bins) | 'all' (all filled points)
SHADE_MODE = 'multifield'   # 'multifield'(paper) | 'field' | 'logs' | 'hybrid' | 'deeplimit' | 'greybands'
A0_SI = 1.20e-10        # MOND acceleration scale [m s^-2], for g_s/a_0 shading
# Fit s_low = k * r_eq^a to the four mass bins, folding each s>1 point to its s<1 mirror.
# Exponent a = 1 (fixed): the "constant neighbourhood density" line (s ~ r_eq); a free fit
# on the current bins gives a = 0.86 with the same 0.086 dex scatter, so a = 1 is retained.
SLOPE = 1.0
_lx, _ly = [], []
for el in ELEMENTS:
    if FIT_SET == 'bins' and not str(el.get('label') or '').startswith('Bin'):
        continue
    cand = [(kk, el[kk]) for kk in ('lo', 'hi')
            if el[kk] is not None and s_min_ax <= el[kk][0] <= s_max_ax]
    if not cand:
        continue
    pref = min(cand, key=lambda kv: kv[1][1])[0]
    s_p = el[pref][0]
    _lx.append(np.log(el['map_x']))
    _ly.append(np.log(s_p) if s_p <= 1.0 else -np.log(s_p))
_lnk = float(np.mean(np.array(_ly) - SLOPE * np.array(_lx)))
K = float(np.exp(_lnk))
R_CROSS = K ** (-1.0 / SLOPE)   # r_eq where s_lo = s_hi = 1
L_SYM = 0.24  # [Mpc] exponential scale length: s_sym = exp(+-L_SYM/r_eq)
_rms = float(np.sqrt(np.mean((np.array(_ly) - (SLOPE*np.array(_lx) + _lnk))**2)) / np.log(10))  # dex
print(f"exponent a={SLOPE:.3f}  k(filled)={K:.3f}  s=1 crossing r_eq={R_CROSS:.3f} Mpc  L_SYM={L_SYM:.2f}  rms={_rms:.3f} dex")

# ── Matplotlib style ──────────────────────────────────────────────────────────
mpl.rcParams.update({
    'font.family'        : 'serif',
    'font.size'          : 9,
    'axes.linewidth'     : 0.55,
    'axes.spines.top'    : True,
    'axes.spines.right'  : True,
    'xtick.direction'    : 'in',
    'ytick.direction'    : 'in',
    'xtick.top'          : False,
    'ytick.right'        : False,
    'xtick.minor.visible': False,
    'ytick.minor.visible': False,
    'legend.frameon'     : True,
    'legend.edgecolor'   : '#cccccc',
    'legend.framealpha'  : 0.92,
})

_S_CROSS = 12.0 ** (1.0 / 3.0)          # ≈ 2.289; crossover v_H = v_N
yt_base = [0.002, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 50, 100]
yt = sorted(yt_base + [_S_CROSS, 1.0/_S_CROSS])

# ── Figure layout: (a)=s_equal, (b)=chi2 landscape ───────────────────────────
fig, (ax1, ax2) = plt.subplots(
    1, 2, figsize=(13.0, 7.04),
    gridspec_kw={'width_ratios': [1.2, 1.0]},
    sharey=True
)
fig.subplots_adjust(left=0.09, right=0.98, bottom=0.12, top=0.96, wspace=0.16)

# ══════════════════════════════════════════════════════════════════════════════
# Panel (a) — s_equal regime map (x=r, y=s)   [LEFT, shows y-axis]
# ══════════════════════════════════════════════════════════════════════════════
# s_low/s_high power-law relations with k=1 [Mpc^-1], crossing at r_eq = 1/k = 1 Mpc.
# s_low = k*r_eq (neighbourhood-dominated), s_high = 1/(k*r_eq) (Hubble-dominated),
# merging to s=1 for r_eq >= 1/k.
K_REF = 1.0   # [Mpc^-1]
r_line = np.logspace(-3, np.log10(r_max_ax), 500)  # from 0.001 Mpc ≈ 0
_slo = np.where(r_line < 1.0/K_REF, K_REF * r_line, 1.0)
_shi = np.where(r_line < 1.0/K_REF, 1.0 / (K_REF * r_line), 1.0)
ax1.plot(r_line, _shi, color='#333333', lw=1.6, ls=(0, (6, 3)), zorder=3)
ax1.plot(r_line, _slo, color='#333333', lw=1.6, ls=(0, (6, 3)), zorder=3)
# Additional s_low/s_high curves for k=0.5 and k=2 [Mpc^-1] in light grey
for _k_alt in (0.5, 2.0):
    _slo_alt = np.where(r_line < 1.0/_k_alt, _k_alt * r_line, 1.0)
    _shi_alt = np.where(r_line < 1.0/_k_alt, 1.0 / (_k_alt * r_line), 1.0)
    ax1.plot(r_line, _shi_alt, color='#aaaaaa', lw=0.9, ls=(0, (6, 3)), zorder=2)
    ax1.plot(r_line, _slo_alt, color='#aaaaaa', lw=0.9, ls=(0, (6, 3)), zorder=2)
# Thick, faint grey guides at the xi^2 crossover s_eq = 12^{1/3} (where the neighbourhood term
# 1/s^3 equals the Hubble term at r = r_eq, i.e. v_H = v_N) and its s <-> 1/s mirror 12^{-1/3}.
# Drawn above the g_s/a0 field (zorder 0) but below the reference lines and data stars, each with
# a small left-edge label.
# Guide lines at xi^2 crossover; s value shown as custom y-axis tick "2.29→" / "0.44→"
for _sc, _shortlbl in [(_S_CROSS, r'$v_H\!=\!v_N$, $s\!>\!1$'),
                        (1.0/_S_CROSS, r'$v_H\!=\!v_N$, $s\!<\!1$')]:
    ax1.axhline(_sc, color='0.40', lw=7, alpha=0.25, zorder=1, solid_capstyle='round')
    ax1.text(0.08, _sc, _shortlbl, ha='center', va='center', fontsize=5.5,
             color='0.25', fontstyle='italic', zorder=5,
             bbox=dict(boxstyle='round,pad=0.08', fc='white', ec='none', alpha=0.95))
_lbl_eff = [pe.withStroke(linewidth=2.6, foreground='white')]
# s_low/s_high labels: at r_eq = 0.09 Mpc, rotated parallel to each curve (~±50° in display)
_r_lbl = 0.09
ax1.text(_r_lbl, K_REF*_r_lbl / 1.55, r'$s_\mathrm{low}$', color='#333', fontsize=9,
         ha='center', va='center', rotation=50, rotation_mode='anchor', zorder=5,
         path_effects=_lbl_eff)
ax1.text(_r_lbl, 1.0/(K_REF*_r_lbl) * 1.60, r'$s_\mathrm{high}$', color='#333', fontsize=9,
         ha='center', va='center', rotation=-50, rotation_mode='anchor', zorder=5,
         path_effects=_lbl_eff)

# External reference systems from Table A.1 (r_eq [Mpc], s), read from reference_systems.csv.
# Each may carry the s<->1/s degeneracy: FILLED star = adopted branch, OPEN star = the mirror,
# FITTED separately per system (NOT 1/s -- the Hubble term breaks the symmetry, e.g. gal-gal WL).
import csv as _csv
_refdir = os.path.dirname(DST)
# Ordered warm-yellow → dark-blue colour sequence (legend order: col3 top→bot, col4 top→bot→MW last)
# 1 X-COP         groc ataronjat (orange-yellow)
# 2 HIFLUGCS      groc (yellow)
# 3 X-ray groups  groc claret (light yellow)
# 4 gal-gal WL    groc verdós (yellowish-green)
# 5 SPARC         verd claret (light green)
# 6 MIGHTEE       verd blavós claret (light blue-green)
# 7 UDGs          blau cel (sky blue)
# 8 Milky Way     blau fosc (dark blue) — last in legend
_SYS_COL = {
    'X-COP':      ('#FFA040', '#7A3800'),  # 1 groc ataronjat
    'HIFLUGCS':   ('#FFD700', '#705800'),  # 2 groc
    'gal-gal WL': ('#BBCC00', '#4A5200'),  # 4 groc verdós
    'SPARC':      ('#80CC80', '#204820'),  # 5 verd claret
    'Milky Way':  ('#1A3BAA', '#0A1A60'),  # 8 blau fosc (last)
}
_COL_XRG = ('#FFE880', '#705800')   # 3 groc claret      (X-ray groups)
_COL_MIG = ('#40BBAA', '#005555')   # 6 verd blavós claret (MIGHTEE)
_COL_UDG = ('#6BB8E8', '#1A5080')   # 7 blau cel          (UDGs)

def _star(x, y, xe, ye, filled, fc='#ffe14d', ec='#6e5200', erc='#8a6d00'):
    if filled:   # adopted branch: crisp, dark, thick edge
        ax1.errorbar(x, y, xerr=xe, yerr=ye, fmt='*', color=fc, ms=14, mec=ec,
                     mew=1.4, ecolor=erc, elinewidth=1.1, capsize=3, zorder=9)
    else:        # mirror branch: hollow, thin, soft edge
        ax1.errorbar(x, y, xerr=xe, yerr=ye, fmt='*', mfc='none', mec=ec, mew=0.8,
                     ms=14, ecolor=erc, elinewidth=0.8, capsize=3, zorder=8)

for _r in _csv.DictReader(open(os.path.join(_refdir, "reference_systems.csv"))):
    _x, _y = float(_r['r_eq']), float(_r['s'])
    _xe = None if float(_r['r_eq_lo']) == float(_r['r_eq_hi']) else \
          [[_x - float(_r['r_eq_lo'])], [float(_r['r_eq_hi']) - _x]]
    _ye = None if float(_r['s_lo']) == float(_r['s_hi']) else \
          [[_y - float(_r['s_lo'])], [float(_r['s_hi']) - _y]]
    _sfc, _sec = _SYS_COL.get(_r['name'], ('#ffe14d', '#6e5200'))
    _star(_x, _y, _xe, _ye, filled=True, fc=_sfc, ec=_sec, erc=_sec)
    if _r.get('s_alt', '') not in ('', None):         # mirror branch
        _ya = float(_r['s_alt'])
        _yea = [[_ya - float(_r['s_alt_lo'])], [float(_r['s_alt_hi']) - _ya]]
        _deg = str(_r.get('degenerate', '0')) == '1'
        _star(_x, _ya, _xe, _yea, filled=_deg, fc=_sfc, ec=_sec, erc=_sec)
# SPARC individual galaxies: small stars on BOTH branches -- FILLED = each galaxy's lower-chi2
# branch, OPEN = its mirror (same filled/open convention as the big stars).
_pg = os.path.join(_refdir, "sparc_per_galaxy.csv")
if os.path.exists(_pg):
    _gx, _gf, _go = [], [], []
    for _g in _csv.DictReader(open(_pg)):
        _gx.append(float(_g['r_eq'])); _gf.append(float(_g['s_fill'])); _go.append(float(_g['s_open']))
    _sp_fc, _sp_ec = _SYS_COL['SPARC']
    ax1.plot(_gx, _gf, marker='*', ls='', ms=5.4, color=_sp_fc, mec=_sp_ec, mew=0.8,
             alpha=0.65, zorder=6)                                   # filled = min-chi2
    ax1.plot(_gx, _go, marker='*', ls='', ms=5.2, mfc='none', mec=_sp_ec, mew=0.5,
             alpha=0.60, zorder=6)                                   # open = mirror
# X-COP and HIFLUGCS individual clusters: FILLED = each cluster's s<1 fit, OPEN = its 1/s mirror.
for _fn in ("xcop_per_cluster.csv", "hiflugcs_per_cluster.csv"):
    _cp = os.path.join(_refdir, _fn)
    if not os.path.exists(_cp):
        continue
    _cd = list(_csv.DictReader(open(_cp)))
    _cx = [float(_c['r_eq']) for _c in _cd]
    _cf = [float(_c['s_fill']) for _c in _cd]; _co = [float(_c['s_open']) for _c in _cd]
    _cfc, _cec = _SYS_COL['X-COP'] if 'xcop' in _fn else _SYS_COL['HIFLUGCS']
    ax1.plot(_cx, _cf, marker='*', ls='', ms=6.0, color=_cfc, mec=_cec, mew=0.8,
             alpha=0.65, zorder=6)                                   # filled = fitted s
    ax1.plot(_cx, _co, marker='*', ls='', ms=5.8, mfc='none', mec=_cec, mew=0.5,
             alpha=0.60, zorder=6)                                   # open = 1/s mirror
# Gal-gal WL individual mass bins (Mistele 2024, 4 bins): all degenerate (Delta chi2 < 0.3).
# Both branches drawn FILLED (same gold star, alpha=0.6) to signal degeneracy.
_gp = os.path.join(_refdir, "galgal_per_bin.csv")
if os.path.exists(_gp):
    _gd = list(_csv.DictReader(open(_gp)))
    _gx = [float(_g['r_eq']) for _g in _gd]
    _gf = [float(_g['s_fill']) for _g in _gd]
    _go = [float(_g['s_open']) for _g in _gd]
    _gwfc, _gwec = _SYS_COL['gal-gal WL']
    ax1.plot(_gx, _gf, marker='*', ls='', ms=6.0, color=_gwfc, mec=_gwec, mew=0.8,
             alpha=0.65, zorder=6)
    ax1.plot(_gx, _go, marker='*', ls='', ms=6.0, color=_gwfc, mec=_gwec, mew=0.8,
             alpha=0.65, zorder=6)

# ── HMG group/UDG bisector (used for X-ray groups AND UDGs below) ─────────────
_GS_MAX_SI = 2.0*C_KMS/T0 * math.cos(math.pi/3)/(math.pi/3) * CODE_TO_SI  # ≈ 5.48 a_0

def _hmg_gtot_grp(g_bar_SI, r_kpc, s):
    gc  = g_bar_SI / CODE_TO_SI
    v2n = max(gc * r_kpc, 1e-60)
    xi2 = 1.0/s**3 + (r_kpc/T0)**2 / (12.0*v2n)
    ge  = 2.0*C_KMS/T0 * float(_q(math.sqrt(max(xi2, 1e-30)))) * CODE_TO_SI
    return math.sqrt(max(g_bar_SI*(g_bar_SI + ge), 0.0))

def _bisect_s_grp(g_bar_SI, g_obs_SI, r_kpc):
    """Find s<1 s.t. g_tot(s)=g_obs by bisection. Returns None if railed."""
    if _hmg_gtot_grp(g_bar_SI, r_kpc, 0.9999) < g_obs_SI:
        return None   # railed
    if _hmg_gtot_grp(g_bar_SI, r_kpc, 1e-4) > g_obs_SI:
        return None   # g_obs < g_bar (sub-Newtonian)
    _sl, _sh = 1e-4, 0.9999
    for _ in range(60):
        _sm = 0.5*(_sl + _sh)
        if _hmg_gtot_grp(g_bar_SI, r_kpc, _sm) < g_obs_SI:
            _sl = _sm
        else:
            _sh = _sm
    return 0.5*(_sl + _sh)

# Gas-rich UDGs (Mancera Pina 2022, 6 galaxies): per-galaxy HMG fits (s<1 basin).
# Data: (id, log10_Mbar[Msun], r_sys[kpc], v_obs[km/s], v_N[km/s], sig(log10 Mbar), sv+[km/s], sv-[km/s])
# Source: Mancera Pina et al. 2022, Table 1; same values as hmg_udg_revised/scripts_experiment/make_fig2_landscape.py.
# s uncertainty: vary v_obs by ±sv (not v_N, which is fixed by M_bar); xerr from sig(log10 Mbar)/3.
_UDGS_CLA = [
    ("114905", 9.21, 8.02,  23.0, 29.0, 0.19, 4.0, 6.0),  # v_obs < v_N → sub-Newton
    ("122966", 9.21, 10.80, 37.0, 25.0, 0.13, 5.0, 6.0),
    ("219533", 9.36, 9.78,  37.0, 32.0, 0.21, 6.0, 5.0),
    ("248945", 9.05, 8.55,  27.0, 24.0, 0.19, 3.0, 3.0),
    ("334315", 9.25, 8.49,  25.0, 30.0, 0.16, 5.0, 5.0),  # v_obs < v_N → sub-Newton
    ("749290", 9.17, 8.47,  26.0, 27.0, 0.15, 6.0, 6.0),  # v_obs < v_N → sub-Newton
]
_C_UDG  = _COL_UDG[0]
_C_UDGE = _COL_UDG[1]
_udg_floor = s_min_ax * 1.65
for _uid, _lmb, _rk_u, _vo, _vn, _slm, _svp, _svm in _UDGS_CLA:
    _mb_u = 10.0**_lmb
    _req_u = (G_CODE * _mb_u * T0**2)**(1.0/3.0) / 1000.0
    _dlr   = _slm / 3.0  # sigma(log10 r_eq)
    _xerr_u = [[_req_u * (1.0 - 10.0**(-_dlr))], [_req_u * (10.0**_dlr - 1.0)]]
    _gb_u  = _vn**2 / _rk_u * CODE_TO_SI    # g_bar from v_N at r_sys
    _go_u  = _vo**2 / _rk_u * CODE_TO_SI    # g_obs from v_obs at r_sys
    if _vo <= _vn:   # sub-Newtonian: v_obs < v_N → no positive s solution
        ax1.errorbar(_req_u, _udg_floor, xerr=_xerr_u, fmt='*', color=_C_UDG, ms=6.0,
                     mec=_C_UDGE, mew=0.7, ecolor=_C_UDGE, elinewidth=0.9,
                     capsize=2.5, zorder=9)
        ax1.annotate('', xy=(_req_u, s_min_ax*1.04), xytext=(_req_u, _udg_floor*0.82),
                     arrowprops=dict(arrowstyle='-|>', color=_C_UDGE, lw=1.1), zorder=9)
    else:
        _sf_u = _bisect_s_grp(_gb_u, _go_u, _rk_u)
        if _sf_u is None:
            continue
        _go_u_hi = (_vo + _svp)**2 / _rk_u * CODE_TO_SI
        _vo_lo   = max(_vo - _svm, _vn)
        _go_u_lo = _vo_lo**2 / _rk_u * CODE_TO_SI
        _sf_u_hi = _bisect_s_grp(_gb_u, _go_u_hi, _rk_u)
        _sf_u_lo = _bisect_s_grp(_gb_u, _go_u_lo, _rk_u) if _go_u_lo > _gb_u else None
        _yerr_u = [[max(_sf_u - (_sf_u_lo if _sf_u_lo else s_min_ax), 0)],
                   [max((_sf_u_hi if _sf_u_hi else _sf_u) - _sf_u, 0)]]
        ax1.errorbar(_req_u, _sf_u, xerr=_xerr_u, yerr=_yerr_u, fmt='*', color=_C_UDG, ms=6.0,
                     mec=_C_UDGE, mew=0.7, ecolor=_C_UDGE, elinewidth=0.9,
                     capsize=2.5, zorder=9)
# Global-fit star (6 UDGs; joint HMG s<1 best fit s=0.0098, mean r_eq=0.112 Mpc)
_s_med_udg = 0.0098
_med_xerr = [[0.1117 * 0.109], [0.1117 * 0.122]]
_med_yerr = [[_s_med_udg - 0.002], [0.023 - _s_med_udg]]
_s_med_show = max(_s_med_udg, _udg_floor)
ax1.errorbar(0.1117, _s_med_show,
             xerr=_med_xerr,
             yerr=None if _s_med_udg < s_min_ax else _med_yerr,
             fmt='*', color=_C_UDG, ms=14, mec=_C_UDGE,
             mew=1.4, ecolor=_C_UDGE, elinewidth=1.2, capsize=3, zorder=10, label='_nolegend_')
if _s_med_udg < s_min_ax:
    ax1.annotate('', xy=(0.1117, s_min_ax*1.04), xytext=(0.1117, _s_med_show*0.82),
                 arrowprops=dict(arrowstyle='-|>', color=_C_UDGE, lw=1.3), zorder=10)
# Asymptotic s>1 branch (published UDG paper): alternative solution → open star, arrow up.
# Star at 40% of s_max to leave enough room for a clearly visible arrow up to 92%.
_udg_y2 = s_max_ax * 0.40
ax1.annotate('', xy=(0.111, s_max_ax * 0.92), xytext=(0.111, _udg_y2),
             arrowprops=dict(arrowstyle='-|>', color=_C_UDGE, lw=1.5), zorder=8)
ax1.plot(0.111, _udg_y2, marker='*', mfc='none', mec=_C_UDGE, mew=1.0, ms=12, ls='', zorder=8,
         path_effects=[pe.withStroke(linewidth=2.6, foreground='white')])

# MIGHTEE UDGs (Ponomareva 2021): 4 gas-rich UDGs with HMG kinematics, s<1 fits.
# r_eq [Mpc] and s values pre-computed; yerr from ±40% RHI variation, no xerr.
_MIG = [
    (0.1781, 0.1269, 0.1193, 0.1305, '#6BAED6'),  # J021759 marginal (face-on)
    (0.2484, 0.0458, 0.0333, 0.0511, '#B2182B'),  # J022350 outlier
    (0.1575, 0.2536, 0.2487, 0.2559, '#2166AC'),  # J022429 on
    (0.1635, 0.2950, 0.2908, 0.2971, '#2166AC'),  # J022522 on
]
# r_eq ∝ M_HI^(1/3): ±40% M_HI → r_eq uncertainty fractions
_RHI_LO = 1.0 - 0.60**(1.0/3.0)   # ≈ 0.157
_RHI_HI = 1.40**(1.0/3.0) - 1.0   # ≈ 0.119
_mig_fc, _mig_ec = _COL_MIG
for _mreq, _ms0, _msl, _msh, _mc in _MIG:
    _myerr = [[max(_ms0 - _msl, 0)], [max(_msh - _ms0, 0)]]
    _mxerr = [[_mreq * _RHI_LO], [_mreq * _RHI_HI]]
    ax1.errorbar(_mreq, _ms0, xerr=_mxerr, yerr=_myerr, fmt='*', color=_mig_fc, ms=6.0,
                 mec=_mig_ec, mew=0.8, ecolor=_mig_ec, elinewidth=1.1,
                 capsize=2.5, zorder=9, label='_nolegend_')
# Median MIGHTEE star (4 galaxies): r_eq=0.1708 Mpc, s=0.1903
_mig_med_xerr = [[0.1708 - 0.1575], [0.2484 - 0.1708]]
_mig_med_yerr = [[0.1903 - 0.0458], [0.2950 - 0.1903]]
ax1.errorbar(0.1708, 0.1903, xerr=_mig_med_xerr, yerr=_mig_med_yerr, fmt='*',
             color=_mig_fc, ms=14, mec=_mig_ec, mew=1.4,
             ecolor=_mig_ec, elinewidth=1.1, capsize=3, zorder=10, label='_nolegend_')

# MIGHTEE s>1 mirror branch: open stars at s_mirror = 1/s_best
_MIG_MIR = [
    (0.1781, 0.1269, 0.1193, 0.1305, '#6BAED6'),  # J021759
    (0.2484, 0.0458, 0.0333, 0.0511, '#B2182B'),  # J022350
    (0.1575, 0.2536, 0.2487, 0.2559, '#2166AC'),  # J022429
    (0.1635, 0.2950, 0.2908, 0.2971, '#2166AC'),  # J022522
]
for _mreq, _ms0, _msl, _msh, _mc in _MIG_MIR:
    _sm = 1.0/_ms0
    _ye_m = [[max(_sm - 1.0/_msh, 0)], [max(1.0/_msl - _sm, 0)]]
    _mxerr_m = [[_mreq * _RHI_LO], [_mreq * _RHI_HI]]
    ax1.errorbar(_mreq, _sm, xerr=_mxerr_m, yerr=_ye_m, fmt='*', mfc='none', mec=_mig_ec, mew=0.8,
                 ms=6.0, ecolor=_mig_ec, elinewidth=1.0, capsize=2.5, zorder=8, label='_nolegend_')
# Median mirror star (s_med_mir = 1/0.1903 ≈ 5.26)
_sm_med = 1.0/0.1903
ax1.errorbar(0.1708, _sm_med, xerr=[[0.1708-0.1575],[0.2484-0.1708]],
             yerr=[[_sm_med-1.0/0.2950],[1.0/0.0458-_sm_med]], fmt='*',
             mfc='none', mec=_mig_ec, mew=1.4, ms=14, ecolor=_mig_ec,
             elinewidth=1.1, capsize=3, zorder=8, label='_nolegend_')

# ── X-ray galaxy groups (Gastaldello et al. 2007) ─────────────────────────────
# 16 X-ray groups (T~1-3 keV), Chandra/XMM; one RAR point per group at r_Δ.
# Branch degeneracy (Hubble term << neighbourhood at group scale): BOTH branches filled.
# Columns: (r_kpc, Mtot_1e13, sMtot, Mgas_1e12, sMgas, Mstar_1e10, sMstar)
_GRPS = [
    (295, 1.85, 0.04,  1.21, 0.02,  0.0,   0.0),   # NGC 5044
    (215, 1.42, 0.03,  1.02, 0.02,  11.2,  4.1),   # NGC 1550
    (185, 0.92, 0.08,  0.31, 0.03,  0.0,   0.0),   # NGC 2563   (railed)
    (292, 3.59, 0.14,  2.60, 0.08,  22.1,  4.5),   # Abell 262  (railed)
    (262, 1.30, 0.04,  0.87, 0.02,  22.4,  2.2),   # NGC 533
    (353, 3.21, 0.10,  2.84, 0.06,  61.8,  7.2),   # MKW 4
    (319, 2.36, 0.13,  1.56, 0.05,  26.4,  6.3),   # IC 1860
    (226, 0.84, 0.07,  0.58, 0.06,  2.8,   6.7),   # NGC 5129
    (208, 1.32, 0.16,  0.66, 0.03,  0.0,   0.0),   # NGC 4325   (railed)
    (422, 5.51, 0.51,  3.35, 0.18,  0.0,   0.0),   # ESO 5520200 (railed)
    (465, 7.38, 0.61,  4.79, 0.29,  22.5,  24.7),  # AWM 4      (railed)
    (343, 5.97, 1.14,  3.45, 0.17,  0.0,   0.0),   # ESO 3060170 (railed)
    (397, 1.85, 0.07,  2.85, 0.11,  0.0,   0.0),   # RGH 80
    (405, 4.92, 1.64,  1.97, 0.19,  0.0,   0.0),   # MS 0116    (railed)
    (710, 10.68, 0.51, 11.36, 0.29,  0.0,   0.0),  # Abell 2717
    (584, 6.13, 3.30,  5.10, 0.41,  56.9,  10.5),  # RXJ 1159
]
_grp_pts = []  # list of dicts with nominal + 1σ uncertainty per group
for _rk, _mt, _smt, _mg, _smg, _ms, _sms in _GRPS:
    _mb     = _mg*1e12 + _ms*1e10
    _smb    = math.sqrt((_smg*1e12)**2 + (_sms*1e10)**2)
    _mt_sun = _mt*1e13;  _smt_sun = _smt*1e13
    _gb  = G_CODE*_mb    / _rk**2 * CODE_TO_SI
    _go  = G_CODE*_mt_sun/ _rk**2 * CODE_TO_SI
    _sgb = G_CODE*_smb   / _rk**2 * CODE_TO_SI
    _sgo = G_CODE*_smt_sun/_rk**2 * CODE_TO_SI
    _req = (G_CODE*_mb*T0**2)**(1.0/3.0) / 1000.0
    _sreq = _req * (1.0/3.0) * _smb / max(_mb, 1e-60)   # σ_r_eq from σ_M_b
    if _go <= _gb:
        continue
    _gs_need = (_go**2 - _gb**2)/_gb
    _fitted = _gs_need < _GS_MAX_SI
    _sf = None
    if _fitted:
        _sf = _bisect_s_grp(_gb, _go, _rk)
        if _sf is None:
            continue
        # 1σ propagation (fitted groups only):
        #   s_lo: g_bar+σ, g_obs−σ → more g_bar, less g_obs → less g_s needed → smaller s
        #   s_hi: g_bar−σ, g_obs+σ → less g_bar, more g_obs → more g_s needed → larger s
        _go_lo = max(_go - _sgo, max(_gb + _sgb, 1e-60) * 1.001)
        _sf_slo = _bisect_s_grp(_gb + _sgb, _go_lo, _rk)        # smaller s bound
        _sf_shi = _bisect_s_grp(max(_gb - _sgb, 1e-60), _go + _sgo, _rk)  # larger s bound
        _slo_yerr = (max(_sf - (_sf_slo if _sf_slo else s_min_ax), 0),
                     max((_sf_shi if _sf_shi else _sf) - _sf, 0))
        _shi_yerr = (max(1.0/_sf - (1.0/_sf_shi if _sf_shi else 1.0/_sf), 0),
                     max((1.0/_sf_slo if _sf_slo else s_max_ax) - 1.0/_sf, 0))
    else:
        # Railated: find min nσ at (g_bar+nσ, g_obs−nσ) that becomes reachable.
        # _sf is the edge s; the interval [_sf, 1/_sf] spans both branches.
        # No additional error bars — the edge itself IS the uncertainty bound.
        for _nsig in (1.0, 1.5, 2.0):
            _gb_e = _gb + _nsig*_sgb;  _go_e = _go - _nsig*_sgo
            if _go_e <= _gb_e: continue
            if (_go_e**2 - _gb_e**2)/_gb_e < _GS_MAX_SI:
                _sf = _bisect_s_grp(_gb_e, _go_e, _rk)
                if _sf is not None:
                    break
        if _sf is None:
            continue
        _slo_yerr = (0, 0)
        _shi_yerr = (0, 0)
    _grp_pts.append({'req': _req, 'slo': _sf, 'shi': 1.0/_sf,
                     'xerr': _sreq,
                     'slo_yerr': _slo_yerr,
                     'shi_yerr': _shi_yerr})

_grp_lo  = [p['slo'] for p in _grp_pts]
_grp_hi  = [p['shi'] for p in _grp_pts]
_grp_req = [p['req'] for p in _grp_pts]
print(f"X-ray groups: {len(_grp_pts)} points, s<1 range [{min(_grp_lo):.3f},{max(_grp_lo):.3f}], "
      f"r_eq range [{min(_grp_req):.3f},{max(_grp_req):.3f}] Mpc")
if _grp_pts:
    _xrg_fc = _COL_XRG[0];  _xrg_ec = _COL_XRG[1]
    for _p in _grp_pts:
        _xe = _p['xerr']
        ax1.errorbar(_p['req'], _p['slo'],
                     xerr=_xe, yerr=[[_p['slo_yerr'][0]], [_p['slo_yerr'][1]]],
                     fmt='*', color=_xrg_fc, ms=4.5, mec=_xrg_ec, mew=0.6,
                     ecolor=_xrg_ec, elinewidth=0.9, capsize=2.2, alpha=0.80, zorder=6)
        ax1.errorbar(_p['req'], _p['shi'],
                     xerr=_xe, yerr=[[_p['shi_yerr'][0]], [_p['shi_yerr'][1]]],
                     fmt='*', color=_xrg_fc, ms=4.5, mec=_xrg_ec, mew=0.6,
                     ecolor=_xrg_ec, elinewidth=0.9, capsize=2.2, alpha=0.80, zorder=6)
    _med_lo  = float(np.median(_grp_lo));  _med_hi = float(np.median(_grp_hi))
    _med_req = float(np.median(_grp_req))
    _xe = [[_med_req - min(_grp_req)], [max(_grp_req) - _med_req]]
    ax1.errorbar(_med_req, _med_lo, xerr=_xe,
                 yerr=[[_med_lo - min(_grp_lo)], [max(_grp_lo) - _med_lo]],
                 fmt='*', color=_xrg_fc, ms=14, mec=_xrg_ec, mew=1.4,
                 ecolor=_xrg_ec, elinewidth=1.1, capsize=3, zorder=9)
    ax1.errorbar(_med_req, _med_hi, xerr=_xe,
                 yerr=[[_med_hi - min(_grp_hi)], [max(_grp_hi) - _med_hi]],
                 fmt='*', color=_xrg_fc, ms=14, mec=_xrg_ec, mew=1.4,
                 ecolor=_xrg_ec, elinewidth=1.1, capsize=3, zorder=9)

# ── Background shading: three candidate schemes (SHADE_MODE) ───────────────────
S_REG = 1.0
R_PROBE = 0.5                                       # KiDS lensing radius [Mpc]
_re_g = np.logspace(-3, np.log10(r_max_ax), 300)  # start 0.001 Mpc for full shading
_s_g  = np.logspace(np.log10(s_min_ax), np.log10(s_max_ax), 260)
_RE, _SG = np.meshgrid(_re_g, _s_g)
_xi2 = 1.0/_SG**3 + (1.0/12.0)*(R_PROBE/_RE)**3     # xi_s^2(s, r_eq) at r = R_PROBE
_seq_line = 12.0**(1.0/3.0) * _re_g / R_PROBE       # regime boundary s_eq(r_eq) at r=R_PROBE

if SHADE_MODE == 'field':                           # actual g_s/a_0 field via q(xi^2)
    _gs = (2.0*C_KMS/T0 * _q(np.sqrt(_xi2)) * CODE_TO_SI) / A0_SI
    ax1.pcolormesh(_RE, _SG, _gs, cmap='YlOrBr', alpha=0.55, zorder=0,
                   shading='gouraud', vmin=0.0, vmax=5.6)
    ax1.plot(_re_g, _seq_line, color='#555', lw=1.1, ls=(0, (5, 2)), zorder=2)
    ax1.text(0.60, 0.47, 'high $g_s/a_0$\n(dynamical peak, $\\xi_s^2\\!\\approx\\!1$)',
             transform=ax1.transAxes, color='#7a3b00', fontsize=7.3, ha='center',
             va='center', fontstyle='italic')
    ax1.text(0.60, 0.11, 'neighbourhood-dominated', transform=ax1.transAxes,
             color='#555', fontsize=7.2, ha='center', fontstyle='italic')
    ax1.text(0.20, 0.90, 'Hubble-dominated', transform=ax1.transAxes,
             color='#555', fontsize=7.2, ha='center', fontstyle='italic')
elif SHADE_MODE == 'logs':                          # diffuse band at s=1, fades with |log s|
    _w = np.exp(-(np.log10(_SG))**2 / (2*0.32**2))
    ax1.pcolormesh(_RE, _SG, _w, cmap='Greys', alpha=0.42, zorder=0,
                   shading='gouraud', vmin=0.0, vmax=1.25)
    ax1.text(0.66, 0.50, 'high $g_s/a_0$\nnear dynamical peak',
             transform=ax1.transAxes, color='#444', fontsize=7.6, ha='center',
             va='center', fontstyle='italic')
elif SHADE_MODE == 'hybrid':                         # band along the xi^2=1 ridge + s_eq line
    _w = np.exp(-(np.log10(_xi2))**2 / (2*0.55**2))
    ax1.pcolormesh(_RE, _SG, _w, cmap='Greys', alpha=0.42, zorder=0,
                   shading='gouraud', vmin=0.0, vmax=1.25)
    ax1.plot(_re_g, _seq_line, color='#555', lw=1.1, ls=(0, (5, 2)), zorder=2)
    ax1.text(0.60, 0.47, 'high $g_s/a_0$\n($\\xi_s^2\\!\\approx\\!1$ ridge)',
             transform=ax1.transAxes, color='#444', fontsize=7.3, ha='center',
             va='center', fontstyle='italic')
    ax1.text(0.60, 0.11, 'neighbourhood-dominated', transform=ax1.transAxes,
             color='#555', fontsize=7.2, ha='center', fontstyle='italic')
    ax1.text(0.20, 0.90, 'Hubble-dominated', transform=ax1.transAxes,
             color='#555', fontsize=7.2, ha='center', fontstyle='italic')
elif SHADE_MODE == 'deeplimit':          # deep-limit g_s(s)=q(1/s^3): only s, symmetric s<->1/s
    _gs_dl = (2.0*C_KMS/T0 * _q(_SG**-1.5) * CODE_TO_SI) / A0_SI
    ax1.pcolormesh(_RE, _SG, _gs_dl, cmap='YlOrBr', alpha=0.50, zorder=0,
                   shading='gouraud', vmin=0.0, vmax=5.6)
    ax1.text(0.70, 0.50, 'high $g_s/a_0$ (deep limit:\nsymmetric $s\\!\\leftrightarrow\\!1/s$, peak $s\\!=\\!1$)',
             transform=ax1.transAxes, color='#7a3b00', fontsize=7.0, ha='center',
             va='center', fontstyle='italic')
elif SHADE_MODE == 'greybands':          # grey g_s field (r=0.5) + s_eq boundaries for several r_sys
    _gs = (2.0*C_KMS/T0 * _q(np.sqrt(_xi2)) * CODE_TO_SI) / A0_SI
    ax1.pcolormesh(_RE, _SG, _gs, cmap='Greys', alpha=0.38, zorder=0,
                   shading='gouraud', vmin=0.0, vmax=5.6)
    for _rs, _c in [(0.1, '#111111'), (0.5, '#555555'), (1.0, '#999999')]:
        _seq = 12.0**(1.0/3.0) * _re_g / _rs
        ax1.plot(_re_g, _seq, color=_c, lw=1.3, ls='--', zorder=2)
        _rlab = 5.0 * _rs / 12.0**(1.0/3.0)           # r_eq where s_eq = 5
        _lb = ('%g' % _rs)
        ax1.text(_rlab, 5.0, rf'$s_\mathrm{{eq}}\,(r_\mathrm{{sys}}={_lb}\,\mathrm{{Mpc}})$',
                 color=_c, fontsize=6.6, rotation=44, rotation_mode='anchor',
                 ha='center', va='bottom', zorder=3)
    ax1.text(0.62, 0.10, 'neighbourhood-dominated\n(below $s_\\mathrm{eq}$)', transform=ax1.transAxes,
             color='#555', fontsize=7.0, ha='center', fontstyle='italic')
    ax1.text(0.17, 0.90, 'Hubble-dominated\n(above $s_\\mathrm{eq}$)', transform=ax1.transAxes,
             color='#555', fontsize=7.0, ha='center', fontstyle='italic')
    ax1.text(0.63, 0.55, 'high $g_s/a_0$\n($\\xi_s^2\\!\\approx\\!1$, shown for $r_\\mathrm{sys}\\!=\\!0.5$ Mpc)',
             transform=ax1.transAxes, color='#333', fontsize=6.8, ha='center', va='center', fontstyle='italic')
elif SHADE_MODE == 'multifield':          # union (max) of 3 g_s fields: equal bands, no overlap darkening
    _gmax = np.zeros_like(_SG)
    for _rs in [0.1, 0.5, 1.0]:
        _xi = 1.0/_SG**3 + (1.0/12.0)*(_rs/_RE)**3
        _gmax = np.maximum(_gmax, (2.0*C_KMS/T0*_q(np.sqrt(_xi))*CODE_TO_SI)/A0_SI)
        # vertical label at the top, where this band is vertical (r_eq = r_sys/12^{1/3})
        _rlab = _rs / 12.0**(1.0/3.0)
        ax1.text(_rlab, s_max_ax*0.92, rf'$r_\mathrm{{sys}}={_rs:g}$ Mpc', rotation=90,
                 color='#333', fontsize=7.4, ha='center', va='top', fontstyle='italic', zorder=4,
                 path_effects=[pe.withStroke(linewidth=2.4, foreground='white')])
    from matplotlib.colors import LinearSegmentedColormap as _LSC
    import matplotlib as _mpl
    _softmap = _LSC.from_list('YlOrBr_soft', _mpl.colormaps['YlOrBr'](np.linspace(0.0, 0.72, 256)))
    _GS_MAX = float(_gmax.max())                       # theoretical g_s^max/a0 (~5.5): cap the scale
    _pcm = ax1.pcolormesh(_RE, _SG, _gmax, cmap=_softmap, alpha=0.55, zorder=0,
                          shading='gouraud', vmin=0.0, vmax=_GS_MAX)
    ax1.text(0.55, 0.82, 'Hubble-dominated', transform=ax1.transAxes, color='#555',
             fontsize=8.2, ha='center', va='center', fontstyle='italic', zorder=4, rotation=30)
    ax1.text(1.2, 0.08, 'neighbourhood-dominated', color='#555',
             fontsize=8.2, ha='center', va='center', fontstyle='italic', zorder=4)

# axhline(s=1) and axvline(R_CROSS) removed — no finite crossing with s_sym formula

# Markers at (map_x, s_min) on ax1 — filled = s<1 basin, open = s>1 basin.
# s_min is the TRUE argmin of each chi2 curve (so they align with panel b).
# Both branches per subsample; FILL the preferred (lower-chi2) one, leave the other open.
for el in ELEMENTS:
    eff = stroke_eff if el['stroke'] else []
    cand = [(k, el[k]) for k in ('lo', 'hi') if el[k] is not None and s_min_ax <= el[k][0] <= s_max_ax]
    if not cand:
        continue
    pref = min(cand, key=lambda kv: kv[1][1])[0]
    for k, pt in cand:
        if k == pref:
            ax1.plot(el['map_x'], pt[0], el['mk'], color=el['color'], ms=el['ms']*1.1, zorder=7,
                     mec='white', mew=0.5, path_effects=eff, label=(el['label'] or None))
        else:
            ax1.plot(el['map_x'], pt[0], el['mk'], color=el['color'], ms=el['ms']*1.1, zorder=7,
                     mfc='none', mec=el['color'], mew=0.8, path_effects=eff)

# Y-error bars (Δχ²=1, 68% CI) for KiDS mass bins in panel (a)
def _s_ci1(curve, s_best, lo, hi):
    m = (s_arr >= lo) & (s_arr < hi)
    c, s = curve[m], s_arr[m]
    if len(c) == 0:
        return s_best, s_best
    chi2_min = c.min()
    thresh = chi2_min + 1.0
    i0 = int(np.argmin(np.abs(s - s_best)))
    sl = s[0]
    for i in range(i0 - 1, -1, -1):
        if c[i] > thresh:
            sl = 0.5 * (s[i] + s[i + 1]) if i + 1 < len(s) else s[i]
            break
    sh = s[-1]
    for i in range(i0 + 1, len(s)):
        if c[i] > thresh:
            sh = 0.5 * (s[i - 1] + s[i]) if i - 1 >= 0 else s[i]
            break
    return sl, sh

_all_curves = list(c_bins) + [c_all, c_late, c_early, c_blue, c_red]
for _el, _cv in zip(ELEMENTS, _all_curves):
    _mx = _el['map_x']
    for _branch, _basin_lo, _basin_hi in [('lo', s_min_ax, 1.0), ('hi', 1.0, s_max_ax)]:
        if _el[_branch] is None:
            continue
        _sbest = _el[_branch][0]
        _sl, _sh = _s_ci1(_cv, _sbest, _basin_lo, _basin_hi)
        ax1.errorbar(_mx, _sbest,
                     yerr=[[_sbest - _sl], [_sh - _sbest]],
                     fmt='none', ecolor=_el['color'], elinewidth=1.1, capsize=2.5, zorder=6)

# ax1 axes formatting
_fwd_x = lambda x: np.log1p(np.log1p(np.maximum(x, 0.0)))
_inv_x = lambda y: np.expm1(np.expm1(y))
ax1.set_xscale('function', functions=(_fwd_x, _inv_x))
ax1.set_yscale('log')
ax1.set_xlim(0.0, r_max_ax)
ax1.set_ylim(s_min_ax, s_max_ax)
ax1.set_xlabel(r'dynamical radius $r_\mathrm{eq}\;[\mathrm{Mpc}]$')
ax1.set_ylabel(r'neighbourhood parameter $s$')

xt = [0, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
ax1.set_xticks(xt)
ax1.set_xticklabels([f'{x:g}' for x in xt])
ax1.xaxis.set_minor_locator(mticker.NullLocator())
ax1.set_yticks(yt)
ax1.yaxis.set_minor_locator(mticker.NullLocator())
# NOTE: ax1 yticklabels are applied AFTER ax2 to avoid being reset by sharey behaviour.
# See the block labelled "Re-apply ax1 y-tick labels" below.

# ── Legend 1: small top-right — filling criteria only ─────────────────────────
_ms_lc = 9   # marker size for legend circles (both same)
proxy_filled = Line2D([0], [0], marker='o', color='#666', ms=_ms_lc,
                      mec='white', mew=0.9, ls='', label=r'preferred (min $\chi^2_\nu$)')
proxy_open   = Line2D([0], [0], marker='o', color='none', ms=_ms_lc,
                      mec='#666', mew=1.5, ls='', label='alternative basin')
leg_fill = ax1.legend(handles=[proxy_filled, proxy_open],
                      title='Filling criteria', title_fontsize=7.2,
                      fontsize=7.0, loc='upper right', ncol=1,
                      handlelength=1.2, labelspacing=0.5, borderaxespad=0.8, framealpha=0.92)
leg_fill.get_title().set_fontweight('bold')
ax1.add_artist(leg_fill)

# ── Legend 2: main bottom-right — 4 columns (col-major order: each col top→bottom) ──
_SP = _SYS_COL['SPARC'];   _MW = _SYS_COL['Milky Way']
_XC = _SYS_COL['X-COP'];   _HF = _SYS_COL['HIFLUGCS'];  _GW = _SYS_COL['gal-gal WL']
_blank = Line2D([], [], color='none', linewidth=0, label='')
def _mk(marker, color, ec, label, ms=9):
    return Line2D([], [], marker=marker, color=color, ms=ms, ls='',
                  mec=ec, mew=0.8, label=label)
# Arranged in column-major order (matplotlib fills each column top→bottom with ncol=4)
_l2 = [
    # Column 1: RAR KiDS bins
    Line2D([], [], color='none', label=r'$\mathbf{RAR\ KiDS\ bins}$'),
    _mk('o', '#3498db', 'white', 'Bin 1'),
    _mk('o', '#27ae60', 'white', 'Bin 2'),
    _mk('o', '#e67e22', 'white', 'Bin 3'),
    _mk('D', '#222222', 'white', 'Global'),
    # Column 2: RAR KiDS morfo
    Line2D([], [], color='none', label=r'$\mathbf{RAR\ KiDS\ morfo.}$'),
    _mk('^', '#2ecc71', 'white', r'Late ($n\!<\!2.5$)'),
    _mk('s', '#8e44ad', 'white', r'Early ($n\!\geq\!2.5$)'),
    _mk('o', '#2980b9', 'white', r'Blue ($u\!-\!r$)'),
    _mk('o', '#e91e63', 'white', r'Red ($u\!-\!r$)'),
    # Column 3: clusters + gal-gal WL
    Line2D([], [], color='none', label=r'$\mathbf{Other\ systems}$'),
    _mk('*', _XC[0], _XC[1], 'X-COP clusters', ms=11),
    _mk('*', _HF[0], _HF[1], 'HIFLUGCS clusters', ms=11),
    _mk('*', _COL_XRG[0], _COL_XRG[1], 'X-ray groups', ms=11),
    _mk('*', _GW[0], _GW[1], r'gal--gal WL', ms=11),
    # Column 4: galaxies (MW last = darkest blue)
    _blank,
    _mk('*', _SP[0], _SP[1], 'SPARC galaxies', ms=11),
    _mk('*', _COL_MIG[0], _COL_MIG[1], 'MIGHTEE', ms=11),
    _mk('*', _COL_UDG[0], _COL_UDG[1], 'UDGs', ms=11),
    _mk('*', _MW[0], _MW[1], 'Milky Way', ms=11),
]
ax1.legend(_l2, [h.get_label() for h in _l2],
           fontsize=6.8, loc='lower right', ncol=4,
           handlelength=1.0, labelspacing=0.35, columnspacing=0.8,
           handletextpad=0.4, borderaxespad=0.6, framealpha=0.92)
# g_s/a_0 colour-scale legend for the panel-(a) shading, in the gap between the two panels
if SHADE_MODE == 'multifield':
    _cax = fig.add_axes([0.548, 0.14, 0.013, 0.29])
    _cb = fig.colorbar(_pcm, cax=_cax, ticks=[1, 2, 3, 4, 5])
    _cb.solids.set(alpha=1.0)
    _cb.set_label(r'$g_s/a_0$', fontsize=8.5, labelpad=2)
    _cb.ax.tick_params(labelsize=7)
ax1.text(-0.02, 1.04, '(a)', transform=ax1.transAxes,
         fontsize=11, fontweight='bold', va='bottom', ha='right')

# ══════════════════════════════════════════════════════════════════════════════
# Panel (b) — chi2_nu landscape ROTATED 90° (x=chi2, y=s)   [RIGHT]
# ══════════════════════════════════════════════════════════════════════════════

# Reference lines
ax2.axvline(1.0, color='#888888', lw=0.8, ls='--', zorder=1)
ax2.text(1.13, s_max_ax * 0.88, r'$\chi^2_\nu=1$', color='#888888',
         fontsize=7.5, va='top', ha='left')

# Per-bin chi2 curves (dashed, colored) — rotated
bin_colors = ['#3498db', '#27ae60', '#e67e22', '#c0392b']
bin_labels_short = [r'Bin 1', r'Bin 2', r'Bin 3', r'Bin 4']
for i, (col, lbl) in enumerate(zip(bin_colors, bin_labels_short)):
    ax2.loglog(c_bins[i], s_arr, color=col, lw=1.3, ls='--',
               label=lbl, zorder=3)

# Global + morphological curves (thick, solid) — rotated
ax2.loglog(c_all,   s_arr, color='#222222', lw=2.4,
           label='Global (all bins)', zorder=5)
ax2.loglog(c_late,  s_arr, color='#2ecc71', lw=2.2,
           label=r'Late ($n<2.5$)', zorder=5)
ax2.loglog(c_early, s_arr, color='#8e44ad', lw=2.2,
           label=r'Early ($n\geq2.5$)', zorder=5)
ax2.loglog(c_blue,  s_arr, color='#2980b9', lw=1.8, ls=':',
           label=r'Blue ($u-r$)', zorder=4)
ax2.loglog(c_red,   s_arr, color='#c0392b', lw=1.8, ls=':',
           label=r'Red ($u-r$)', zorder=4)
ax2.loglog(_c_udg, s_arr, color='0.45', lw=3.5, alpha=0.14, ls='-',
           label=r'UDGs (6, $s\!<\!1$)', zorder=3)
# NOTE: the per-cluster chi2_nu(s) haze (both HIFLUGCS and X-COP) is NOT drawn.
# For X-COP a single scalar s is not the right model (hydrostatic profiles -- see xcop_rnei_nsig.py);
# for HIFLUGCS the reduced-chi2 haze added no information and cluttered the panel.

# Mark both minima (small dots); fill the preferred (lower-chi2) one, other open.
for el in ELEMENTS:
    cand = [(k, el[k]) for k in ('lo', 'hi') if el[k] is not None and s_min_ax <= el[k][0] <= s_max_ax]
    if not cand:
        continue
    pref = min(cand, key=lambda kv: kv[1][1])[0]
    for k, pt in cand:
        if k == pref:
            ax2.plot(pt[1], pt[0], el['mk'], color=el['color'], ms=el['ms']*0.7,
                     zorder=8, mec='white', mew=0.7)
        else:
            ax2.plot(pt[1], pt[0], el['mk'], color=el['color'], ms=el['ms']*0.7,
                     zorder=8, mfc='none', mec=el['color'], mew=1.2)

# UDG minimum marker in panel (b)
_chi2_udg_min = float(_c_udg.min())
_s_udg_min = float(s_arr[_c_udg.argmin()])
ax2.plot(_chi2_udg_min, _s_udg_min, '*', color='#ffe14d', ms=9, mec='#6e5200', mew=0.8, zorder=9)

# ax2 axes formatting
ax2.set_xscale('log')
ax2.set_yscale('log')
chi2_lo = min(c_all.min(), c_late.min(), c_early.min(), c_blue.min(), c_red.min(),
              min(cb.min() for cb in c_bins), _c_udg.min()) * 0.55
chi2_hi = 200.0
ax2.set_xlim(max(chi2_lo, 0.18), chi2_hi)
ax2.set_ylim(s_min_ax, s_max_ax)
ax2.set_xlabel(r'$\chi^2_\nu$ (log-space residuals)')
# y-axis ticks on right side of panel (b)
ax2.set_yticks(yt)
ax2.tick_params(labelleft=False, left=True, labelright=True, right=True)
def _y2lbl(y):
    if abs(y - _S_CROSS)/(1e-12+_S_CROSS) < 0.005 or abs(y - 1.0/_S_CROSS)/(1e-12+1.0/_S_CROSS) < 0.005:
        return ''   # guide ticks: no label on right axis
    return f'{y:g}'
ax2.set_yticklabels([_y2lbl(y) for y in yt])

# ── Re-apply ax1 y-tick labels (must come AFTER ax2 block to survive sharey reset) ──
def _y1lbl(y):
    if abs(y - _S_CROSS) / _S_CROSS < 0.005:           return r'$2.29\!\to$'
    if abs(y - 1.0/_S_CROSS) / (1.0/_S_CROSS) < 0.005: return r'$0.44\!\to$'
    return f'{y:g}'
ax1.set_yticklabels([_y1lbl(y) for y in yt])
for _tlbl in ax1.get_yticklabels():
    if r'\to' in _tlbl.get_text():
        _tlbl.set_color('0.30'); _tlbl.set_fontstyle('italic'); _tlbl.set_fontsize(8.5)

xt2 = [1, 2, 5, 10, 20, 50, 100]
ax2.set_xticks(xt2)
ax2.set_xticklabels([str(x) for x in xt2])
ax2.xaxis.set_minor_locator(mticker.NullLocator())
ax2.yaxis.set_minor_locator(mticker.NullLocator())

ax2.legend(fontsize=7.2, loc='lower right', ncol=1,
           handlelength=1.4, labelspacing=0.25, borderaxespad=0.8)
ax2.text(-0.02, 1.04, '(b)', transform=ax2.transAxes,
         fontsize=11, fontweight='bold', va='bottom', ha='right')

# ── Cross-panel links: segment from map point (a) to chi2 minimum (b) ─────────
# Same colour as the element linked; starts at the (r,s) marker in (a) and ends
# exactly at the chi2 minimum in (b), both at constant s (horizontal in figure).
from matplotlib.patches import ConnectionPatch
for el in ELEMENTS:
    cand = [(k, el[k]) for k in ('lo', 'hi') if el[k] is not None and s_min_ax <= el[k][0] <= s_max_ax]
    if not cand:
        continue
    pref = min(cand, key=lambda kv: kv[1][1])[0]
    for k, pt in cand:
        s_m, chi2_m = pt
        con = ConnectionPatch(
            xyA=(el['map_x'], s_m), coordsA=ax1.transData,
            xyB=(chi2_m,      s_m), coordsB=ax2.transData,
            color=el['color'], ls=':', lw=0.8,
            alpha=0.55 if k == pref else 0.35, zorder=1)
        fig.add_artist(con)

# UDG cross-panel dashed line at s_udg_min
_con_udg = ConnectionPatch(
    xyA=(0.1117, _s_udg_min), coordsA=ax1.transData,
    xyB=(_chi2_udg_min, _s_udg_min), coordsB=ax2.transData,
    color='0.55', ls='--', lw=0.9, alpha=0.70, zorder=1)
fig.add_artist(_con_udg)

# ── Save ──────────────────────────────────────────────────────────────────────
fig.savefig(DST, dpi=200, bbox_inches='tight')
print("Saved:", DST, "(SHADE_MODE=%s, FIT_SET=%s)" % (SHADE_MODE, FIT_SET))
fig.savefig(DST.replace('.png', '.pdf'), bbox_inches='tight')
print(f"Saved: {DST}")
plt.close(fig)
