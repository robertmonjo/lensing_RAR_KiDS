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

def _log_sigma(b):
    return (np.log10(b[:, 3]) - np.log10(b[:, 2])) / 2.0

def chi2_all(s):
    tot = 0.0; n = 0
    for b, ms in zip(BINS, MBAR):
        r_kpc = b[:, 0] * MPC_TO_KPC
        g_d   = b[:, 1]
        g_h   = hmg_gobs(r_kpc, ms, s)
        sig   = _log_sigma(b)
        tot  += np.sum(((np.log10(g_d) - np.log10(g_h)) / sig)**2)
        n    += len(b)
    return float(tot) / n

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
    b = BINS[ib]; ms = MBAR[ib]
    r_kpc = b[:, 0] * MPC_TO_KPC
    g_d   = b[:, 1]
    g_h   = hmg_gobs(r_kpc, ms, s)
    sig   = _log_sigma(b)
    return float(np.sum(((np.log10(g_d) - np.log10(g_h)) / sig)**2)) / len(b)

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
s_arr = np.logspace(np.log10(0.08), np.log10(13.0), 700)   # extends past the [0.1,10] axis so (b) curves exit cleanly
print("Computing chi2 landscape...", flush=True)
c_all   = np.array([chi2_all(s)   for s in s_arr])
c_bins  = [np.array([chi2_bin(s, i) for s in s_arr]) for i in range(4)]
c_late  = np.array([chi2_morph(s, _gb_late,  _go_late,  _ge_late,  "late")  for s in s_arr])
c_early = np.array([chi2_morph(s, _gb_early, _go_early, _ge_early, "early") for s in s_arr])
c_blue  = np.array([chi2_morph(s, _gb_blue,  _go_blue,  _ge_blue,  "blue")  for s in s_arr])
c_red   = np.array([chi2_morph(s, _gb_red,   _go_red,   _ge_red,   "red")   for s in s_arr])
print("Done.")

# ── s_equal functions ─────────────────────────────────────────────────────────
def Cseq(M):
    return (12.0 * G_CODE * M * T0**2) ** (1.0/3.0)   # kpc

def r_newton(M):
    return (G_CODE * M * T0**2) ** (1.0/3.0) / 1000.0  # Mpc

# ── Layout constants ───────────────────────────────────────────────────────────
s_min_ax  = 0.1
s_max_ax  = 10.0
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
_rms = float(np.sqrt(np.mean((np.array(_ly) - (SLOPE*np.array(_lx) + _lnk))**2)) / np.log(10))  # dex
print(f"exponent a={SLOPE:.3f}  k(filled)={K:.3f}  s=1 crossing r_eq={R_CROSS:.3f} Mpc  rms={_rms:.3f} dex")

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

yt = [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0]

# ── Figure layout: (a)=s_equal, (b)=chi2 landscape ───────────────────────────
fig, (ax1, ax2) = plt.subplots(
    1, 2, figsize=(13.0, 6.4),
    gridspec_kw={'width_ratios': [1.2, 1.0]},
    sharey=True
)
fig.subplots_adjust(left=0.09, right=0.98, bottom=0.12, top=0.96, wspace=0.16)

# ══════════════════════════════════════════════════════════════════════════════
# Panel (a) — s_equal regime map (x=r, y=s)   [LEFT, shows y-axis]
# ══════════════════════════════════════════════════════════════════════════════
# Symmetric relations forced to cross at s=1, fitted amplitude k (filled points):
#   s_lo = k r_eq^{+1/2} ;  s_hi = (1/k) r_eq^{-1/2}   (mirror about s=1)
r_line = np.logspace(np.log10(r_min_ax), np.log10(r_max_ax), 400)
# Below the crossing the two basins are distinct; above it they merge onto s=1.
r_lo = np.concatenate([r_line[r_line < R_CROSS], [R_CROSS]])
r_hi = np.concatenate([[R_CROSS], r_line[r_line > R_CROSS]])
ax1.plot(r_lo, K * r_lo**SLOPE,        color='#333333', lw=1.6, ls=(0, (6, 3)), zorder=3)
ax1.plot(r_lo, (1.0/K) * r_lo**-SLOPE, color='#333333', lw=1.6, ls=(0, (6, 3)), zorder=3)
ax1.plot(r_hi, np.ones_like(r_hi),     color='#333333', lw=1.8, ls=(0, (6, 3)), zorder=3)
_lbl_eff = [pe.withStroke(linewidth=2.6, foreground='white')]
_rlbl = 0.56                                        # label anchor at r_eq ~ 0.56 Mpc
ax1.text(_rlbl, K*_rlbl**SLOPE, r'$s_\mathrm{low}$', color='#222', fontsize=10, rotation=43,
         rotation_mode='anchor', ha='center', va='bottom', zorder=5, path_effects=_lbl_eff)
ax1.text(_rlbl, 1.0/(K*_rlbl**SLOPE), r'$s_\mathrm{high}$', color='#222', fontsize=10, rotation=-43,
         rotation_mode='anchor', ha='center', va='bottom', zorder=5, path_effects=_lbl_eff)

# External reference systems from Table A.1 (r_eq [Mpc], s), read from reference_systems.csv.
# Each may carry the s<->1/s degeneracy: FILLED star = adopted branch, OPEN star = the mirror,
# FITTED separately per system (NOT 1/s -- the Hubble term breaks the symmetry, e.g. gal-gal WL).
import csv as _csv
_refdir = os.path.dirname(DST)
_lab = {'SPARC': ('SPARC', (9, 3), 'left'), 'X-COP': ('X-COP clusters', (-10, -13), 'right'),
        'HIFLUGCS': ('HIFLUGCS', (-8, 11), 'right'), 'Milky Way': ('Milky Way', (9, -3), 'left'),
        'gal-gal WL': ('gal--gal WL', (9, 2), 'left')}
_abs = {'HIFLUGCS': (2.5, 0.85), 'X-COP': (3.2, 0.65)}      # cluster labels at data positions
# labels for the OPEN (mirror) stars of the degenerate systems
_lab_alt = {'SPARC': ('SPARC', (-9, -4), 'right'), 'Milky Way': ('Milky Way', (9, 2), 'left'),
            'gal-gal WL': ('gal--gal WL', (9, -3), 'left')}
def _star(x, y, xe, ye, filled, fc='#ffe14d', ec='#6e5200', erc='#8a6d00'):
    if filled:   # adopted branch: crisp, dark, thick edge
        ax1.errorbar(x, y, xerr=xe, yerr=ye, fmt='*', color=fc, ms=18, mec=ec,
                     mew=1.6, ecolor=erc, elinewidth=1.1, capsize=3, zorder=9)
    else:        # mirror branch: hollow, thin, soft edge
        ax1.errorbar(x, y, xerr=xe, yerr=ye, fmt='*', mfc='none', mec=ec, mew=0.8,
                     ms=18, ecolor=erc, elinewidth=0.8, capsize=3, zorder=8)
def _reflabel(nm, x, y, off, ha, alt=False):
    ax1.annotate(nm, (x, y), textcoords='offset points', xytext=off, fontsize=(7.0 if alt else 7.4),
                 color=('#a98f52' if alt else '#8a6d00'), style='italic', ha=ha, zorder=9,
                 path_effects=[pe.withStroke(linewidth=2.0, foreground='white')])
for _r in _csv.DictReader(open(os.path.join(_refdir, "reference_systems.csv"))):
    _x, _y = float(_r['r_eq']), float(_r['s'])
    _xe = None if float(_r['r_eq_lo']) == float(_r['r_eq_hi']) else \
          [[_x - float(_r['r_eq_lo'])], [float(_r['r_eq_hi']) - _x]]
    _ye = None if float(_r['s_lo']) == float(_r['s_hi']) else \
          [[_y - float(_r['s_lo'])], [float(_r['s_hi']) - _y]]
    _star(_x, _y, _xe, _ye, filled=True)
    if _r['name'] in _abs:                            # filled-star label
        ax1.text(*_abs[_r['name']], _lab[_r['name']][0], fontsize=7.4, color='#8a6d00', style='italic',
                 ha='center', va='center', zorder=9, path_effects=[pe.withStroke(linewidth=2.2, foreground='white')])
    else:
        _reflabel(_lab[_r['name']][0], _x, _y, _lab[_r['name']][1], _lab[_r['name']][2])
    if _r.get('s_alt', '') not in ('', None):         # mirror branch: filled if statistically
        _ya = float(_r['s_alt'])                      # indistinguishable from the other, else open
        _yea = [[_ya - float(_r['s_alt_lo'])], [float(_r['s_alt_hi']) - _ya]]
        _deg = str(_r.get('degenerate', '0')) == '1'
        _star(_x, _ya, _xe, _yea, filled=_deg)
        if _r['name'] in _lab_alt:
            _reflabel(_lab_alt[_r['name']][0], _x, _ya, _lab_alt[_r['name']][1], _lab_alt[_r['name']][2], alt=not _deg)
# SPARC individual galaxies: small stars on BOTH branches -- FILLED = each galaxy's lower-chi2
# branch, OPEN = its mirror (same filled/open convention as the big stars).
_pg = os.path.join(_refdir, "sparc_per_galaxy.csv")
if os.path.exists(_pg):
    _gx, _gf, _go = [], [], []
    for _g in _csv.DictReader(open(_pg)):
        _gx.append(float(_g['r_eq'])); _gf.append(float(_g['s_fill'])); _go.append(float(_g['s_open']))
    ax1.plot(_gx, _gf, marker='*', ls='', ms=5.4, color='#ffe14d', mec='#6e5200', mew=0.8,
             alpha=0.6, zorder=6)                                    # filled = min-chi2 (crisp edge)
    ax1.plot(_gx, _go, marker='*', ls='', ms=5.2, mfc='none', mec='#c9a24a', mew=0.5,
             alpha=0.6, zorder=6)                                    # open = mirror (soft edge)
# X-COP and HIFLUGCS individual clusters: FILLED = each cluster's s<1 fit, OPEN = its 1/s mirror.
for _fn in ("xcop_per_cluster.csv", "hiflugcs_per_cluster.csv"):
    _cp = os.path.join(_refdir, _fn)
    if not os.path.exists(_cp):
        continue
    _cd = list(_csv.DictReader(open(_cp)))
    _cx = [float(_c['r_eq']) for _c in _cd]
    _cf = [float(_c['s_fill']) for _c in _cd]; _co = [float(_c['s_open']) for _c in _cd]
    ax1.plot(_cx, _cf, marker='*', ls='', ms=6.0, color='#ffe14d', mec='#6e5200', mew=0.8,
             alpha=0.6, zorder=6)                                    # filled = fitted s (crisp edge)
    ax1.plot(_cx, _co, marker='*', ls='', ms=5.8, mfc='none', mec='#c9a24a', mew=0.5,
             alpha=0.6, zorder=6)                                    # open = 1/s mirror (soft edge)
# Gal-gal WL individual mass bins (Mistele 2024, 4 bins): all degenerate (Delta chi2 < 0.3).
# Both branches drawn FILLED (same gold star, alpha=0.6) to signal degeneracy.
_gp = os.path.join(_refdir, "galgal_per_bin.csv")
if os.path.exists(_gp):
    _gd = list(_csv.DictReader(open(_gp)))
    _gx = [float(_g['r_eq']) for _g in _gd]
    _gf = [float(_g['s_fill']) for _g in _gd]
    _go = [float(_g['s_open']) for _g in _gd]
    ax1.plot(_gx, _gf, marker='*', ls='', ms=6.0, color='#ffe14d', mec='#6e5200', mew=0.8,
             alpha=0.65, zorder=6)
    ax1.plot(_gx, _go, marker='*', ls='', ms=6.0, color='#ffe14d', mec='#6e5200', mew=0.8,
             alpha=0.65, zorder=6)

# Gas-rich UDGs (Monjo 2026udg): global HMG fit prefers s<1 (s=0.01, chi2=8.6) over the
# s->inf branch (chi2=17.1) -> bottom with a downward arrow (s < axis floor).
_udg_y = s_min_ax*1.28
ax1.annotate('', xy=(0.111, s_min_ax*1.01), xytext=(0.111, _udg_y),
             arrowprops=dict(arrowstyle='-|>', color='#b8860b', lw=1.5), zorder=9)
ax1.plot(0.111, _udg_y, marker='*', color='#ffe14d', ms=16, mec='#6e5200', mew=1.6, ls='',
         zorder=9, path_effects=[pe.withStroke(linewidth=3.0, foreground='white')])
ax1.annotate(r'UDGs ($s\!\simeq\!0.01$)', (0.111, _udg_y), textcoords='offset points',
             xytext=(9, 1), fontsize=7.4, color='#8a6d00', style='italic', ha='left', zorder=9,
             path_effects=[pe.withStroke(linewidth=2.2, foreground='white')])
# Asymptotic s>1 branch (published UDG paper): alternative solution -> open star, top, arrow up.
_udg_y2 = s_max_ax*0.82
ax1.annotate('', xy=(0.111, s_max_ax*0.985), xytext=(0.111, _udg_y2),
             arrowprops=dict(arrowstyle='-|>', color='#b8860b', lw=1.3), zorder=8)
ax1.plot(0.111, _udg_y2, marker='*', mfc='none', mec='#c9a24a', mew=1.0, ms=15, ls='', zorder=8,
         path_effects=[pe.withStroke(linewidth=2.6, foreground='white')])
ax1.text(0.11, 9.5, r'UDGs ($s\!>\!40$)', fontsize=7.4, color='#8a6d00', style='italic',
         ha='left', va='top', zorder=8, path_effects=[pe.withStroke(linewidth=2.2, foreground='white')])

# ── Background shading: three candidate schemes (SHADE_MODE) ───────────────────
S_REG = 1.0
R_PROBE = 0.5                                       # KiDS lensing radius [Mpc]
_re_g = np.logspace(np.log10(r_min_ax), np.log10(r_max_ax), 260)
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
        _rlab = max(_rs / 12.0**(1.0/3.0), r_min_ax*1.18)
        ax1.text(_rlab, s_max_ax*0.92, rf'$r_\mathrm{{sys}}={_rs:g}$ Mpc', rotation=90,
                 color='#333', fontsize=7.4, ha='center', va='top', fontstyle='italic', zorder=4,
                 path_effects=[pe.withStroke(linewidth=2.4, foreground='white')])
    from matplotlib.colors import LinearSegmentedColormap as _LSC
    import matplotlib.cm as _mcm
    _softmap = _LSC.from_list('YlOrBr_soft', _mcm.get_cmap('YlOrBr')(np.linspace(0.0, 0.72, 256)))
    _GS_MAX = float(_gmax.max())                       # theoretical g_s^max/a0 (~5.5): cap the scale
    _pcm = ax1.pcolormesh(_RE, _SG, _gmax, cmap=_softmap, alpha=0.55, zorder=0,
                          shading='gouraud', vmin=0.0, vmax=_GS_MAX)
    ax1.text(0.71, 0.82, 'Hubble-dominated', transform=ax1.transAxes, color='#555',
             fontsize=8.2, ha='center', va='center', fontstyle='italic', zorder=4, rotation=30)
    ax1.text(0.80, 0.045, 'neighbourhood-dominated', transform=ax1.transAxes, color='#555',
             fontsize=8.2, ha='center', va='center', fontstyle='italic', zorder=4)

ax1.axhline(S_REG,   color='#888888', lw=1.0, ls='--', zorder=2)
ax1.axvline(R_CROSS, color='#888888', lw=1.0, ls='--', zorder=2)
ax1.text(r_min_ax * 1.15, S_REG * 1.03, r'$s=1$ (regime change)',
         color='#555', fontsize=8, va='bottom', ha='left')
ax1.text(R_CROSS * 1.04, s_min_ax * 1.2, rf'$r_\mathrm{{eq}}={R_CROSS:.2f}$ Mpc',
         color='#555', fontsize=8, va='bottom', ha='left', rotation=90)

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
            ax1.plot(el['map_x'], pt[0], el['mk'], color=el['color'], ms=el['ms']*1.5, zorder=7,
                     mec='white', mew=1.0, path_effects=eff, label=(el['label'] or None))
        else:
            ax1.plot(el['map_x'], pt[0], el['mk'], color=el['color'], ms=el['ms']*1.5, zorder=7,
                     mfc='none', mec=el['color'], mew=1.5, path_effects=eff)

# ax1 axes formatting
ax1.set_xscale('log'); ax1.set_yscale('log')
ax1.set_xlim(r_min_ax, r_max_ax)
ax1.set_ylim(s_min_ax, s_max_ax)
ax1.set_xlabel(r'dynamical radius $r_\mathrm{eq}\;[\mathrm{Mpc}]$')
ax1.set_ylabel(r'neighbourhood parameter $s$')

xt = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0]
ax1.set_xticks(xt)
ax1.set_xticklabels([f'{x:g}' for x in xt])
ax1.xaxis.set_minor_locator(mticker.NullLocator())
ax1.set_yticks(yt)
ax1.set_yticklabels([f'{y:g}' for y in yt])
ax1.yaxis.set_minor_locator(mticker.NullLocator())

# Legend on ax1: branch lines + markers + filled/open (preferred/alternative) proxies
proxy_filled = Line2D([0], [0], marker='o', color='#888', ms=7,
                      mec='white', mew=0.8, ls='', label=r'preferred (min $\chi^2_\nu$)')
proxy_open   = Line2D([0], [0], marker='o', color='#888', ms=7,
                      mfc='none', mec='#888', mew=1.4, ls='', label='alternative basin')
h1, l1 = ax1.get_legend_handles_labels()
h1 += [proxy_filled, proxy_open]
l1 += [proxy_filled.get_label(), proxy_open.get_label()]
ax1.legend(h1, l1, fontsize=7.2, loc='upper right', ncol=1,
           handlelength=1.4, labelspacing=0.62, markerscale=0.78, borderaxespad=0.8)
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

# ax2 axes formatting
ax2.set_xscale('log'); ax2.set_yscale('log')
chi2_lo = min(c_all.min(), c_late.min(), c_early.min(), c_blue.min(), c_red.min(),
              min(cb.min() for cb in c_bins)) * 0.55
chi2_hi = 200.0
ax2.set_xlim(max(chi2_lo, 0.18), chi2_hi)
ax2.set_ylim(s_min_ax, s_max_ax)
ax2.set_xlabel(r'$\chi^2_\nu$ (log-space residuals)')
# y-axis ticks on right side of panel (b)
ax2.set_yticks(yt)
ax2.tick_params(labelleft=False, left=True, labelright=True, right=True)
ax2.set_yticklabels([f'{y:g}' for y in yt])
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

# ── Save ──────────────────────────────────────────────────────────────────────
fig.savefig(DST, dpi=200, bbox_inches='tight')
print("Saved:", DST, "(SHADE_MODE=%s, FIT_SET=%s)" % (SHADE_MODE, FIT_SET))
fig.savefig(DST.replace('.png', '.pdf'), bbox_inches='tight')
print(f"Saved: {DST}")
plt.close(fig)
