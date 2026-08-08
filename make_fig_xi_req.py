"""
Fig: xi_sys vs dynamical radius r_eq for every HMG-tested system.

For each system we take its own probed radius r, stellar/baryonic mass M (hence
r_eq = (G M t_0^2)^{1/3}) and best-fit s, and compute the *actual* neighbourhood
ratio  xi_sys^2 = 1/s^3 + v_H^2/(12 v_N^2) = 1/s^3 + (1/12)(r/r_eq)^3.

Plotting (r_eq, xi_sys) is unambiguous: the extra acceleration g_s = (2c/t_0) q(xi^2)
is an exact function of the y-axis (xi_sys), a bell peaking at xi_sys = 1 and
symmetric under xi^2 -> 1/xi^2.  No fixed r or mass is assumed for the background.
"""
import os
import math
import csv as _csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import hmg_model as _h

# ── constants (same as make_figA2_landscape.py) ───────────────────────────────
G_CODE     = 4.30091727e-6      # (km/s)^2 kpc Msun^-1
T0         = 14.11              # kpc (km/s)^-1
C_KMS      = 299792.458         # km/s
CODE_TO_SI = 3.24078e-14        # (km/s)^2 kpc^-1 -> m s^-2
A0_SI      = 1.20e-10           # m s^-2

def _q(x):
    x = np.maximum(np.asarray(x, float), 1e-12)
    delta = np.abs(x*x - 1) / (x*x + 1)
    s2 = math.sin(math.pi/3)**2 + (1 - math.sin(math.pi/3)**2) * delta
    gamma = np.arcsin(np.sqrt(np.clip(s2, 0, 1)))
    return np.cos(gamma) / np.maximum(gamma, 1e-12)

def r_newton_kpc(M):
    return (G_CODE * M * T0**2) ** (1.0/3.0)      # kpc

def xi2_of(r_kpc, req_kpc, s):
    nb = 0.0 if not np.isfinite(s) else 1.0/s**3
    return nb + (1.0/12.0) * (r_kpc/req_kpc)**3

def gs_a0_of(xi2):
    return (2.0*C_KMS/T0 * _q(np.sqrt(xi2)) * CODE_TO_SI) / A0_SI

# ── baryonic masses and best-fit s from the pipeline outputs ─────────────────
_MBAR_BINS   = _h.MBAR          # M_star*(1+f_cold), Boselli 2014; same as make_tables.py
_MBAR_GLOBAL = _h.MBAR[1]       # Global: median M_star = 3.2e10 (same as KIDS_MSTAR['Global'])

_MSTAR_MORPH = {'late': 10**9.745, 'early': 10**10.154,
                'blue': 10**9.610, 'red':   10**9.888}
_MBAR_MORPH  = {k: m * (1 + _h.f_cold(m)) for k, m in _MSTAR_MORPH.items()}

def _load_s(path):
    """Return preferred-branch s for each subsample from tab_model_comparison.csv."""
    out = {}
    for r in _csv.DictReader(open(path)):
        clo, chi = float(r['chi2nu_lo']), float(r['chi2nu_hi'])
        out[r['subsample']] = float(r['s_best']) if clo <= chi else float(r['s_hi'])
    return out

_s = _load_s(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'outputs', 'tab_model_comparison.csv'))

# ── systems: (label, r_kpc, M_bar[Msun] or None, r_eq_kpc or None, s, marker, color, ms, ann) ──
#   KiDS systems: give M_bar (r_eq computed), r = 500 kpc.  References: give r_eq directly.
_BIN_COLORS = ['#3498db', '#27ae60', '#e67e22', '#c0392b']
SYS = (
    [(f"Bin {i+1}", 500, _MBAR_BINS[i], None, _s[f"Bin {i+1}"],
      'o', _BIN_COLORS[i], 6, None) for i in range(4)]
    + [
    ("Global",              500, _MBAR_GLOBAL,         None, _s['Global'],              'D', '#222222', 8,  None),
    (r"Late ($n<2.5$)",     500, _MBAR_MORPH['late'],  None, _s['Late-type (n<2.5)'],   '^', '#2ecc71', 10, None),
    (r"Early ($n\geq2.5$)", 500, _MBAR_MORPH['early'], None, _s['Early-type (n>=2.5)'], 's', '#8e44ad', 7,  None),
    (r"Blue ($u-r$)",       500, _MBAR_MORPH['blue'],  None, _s['Blue (u-r)'],           'o', '#2980b9', 7,  None),
    (r"Red ($u-r$)",        500, _MBAR_MORPH['red'],   None, _s['Red (u-r)'],            'o', '#c0392b', 7,  None),
    # external reference systems (gold stars), each with its OWN r
    ("Gas-rich UDGs", 9,   None, 111,  np.inf, '*', '#f1c40f', 15, 'UDGs'),
    ("SPARC",         15,  None, 295,  4.0,    '*', '#f1c40f', 15, 'SPARC'),
    ("gal--gal WL",   500, None, 441,  2.5,    '*', '#f1c40f', 15, 'gal--gal WL'),
    ("X-COP clusters",800, None, 4408, 1.0,    '*', '#f1c40f', 17, 'X-COP clusters'),
    ]
)

X_KIND = 'r_eq'    # 'r_eq' (dyn. radius, mass proxy) | 'r_sys' (probed radius r) | 'ratio' (r_sys/r_eq)

rows = []
for lab, r_kpc, M, req, s, mk, col, ms, ann in SYS:
    req_kpc = r_newton_kpc(M) if req is None else float(req)
    xi2 = xi2_of(r_kpc, req_kpc, s)
    if X_KIND == 'r_eq':
        xval = req_kpc / 1000.0
    elif X_KIND == 'r_sys':
        xval = r_kpc / 1000.0
    else:                       # 'ratio' = r_sys/r_eq  (sets the Hubble term)
        xval = r_kpc / req_kpc
    rows.append(dict(lab=lab, req=xval, xi=math.sqrt(xi2), xi2=xi2,
                     gs=gs_a0_of(xi2), mk=mk, col=col, ms=ms, ann=ann, s=s))

for r in rows:
    print(f"{r['lab']:16s} r_eq={r['req']:.3f} Mpc  s={r['s']:.3g}  xi_sys^2={r['xi2']:.3g}  g_s/a0={r['gs']:.2f}")

# ── figure ────────────────────────────────────────────────────────────────────
mpl_rc = {'font.family': 'serif', 'font.size': 10, 'axes.linewidth': 0.6,
          'xtick.direction': 'in', 'ytick.direction': 'in'}
plt.rcParams.update(mpl_rc)

fig, ax = plt.subplots(figsize=(7.4, 5.6))
fig.subplots_adjust(left=0.11, right=0.985, bottom=0.11, top=0.96)

if X_KIND == 'r_eq':
    x_min, x_max = 0.05, 6.0
    x_ticks = [0.05, 0.1, 0.2, 0.5, 1, 2, 5]
    x_label = r'dynamical radius $r_\mathrm{eq}$ [Mpc]'
elif X_KIND == 'r_sys':
    x_min, x_max = 0.005, 1.5
    x_ticks = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1]
    x_label = r'probed radius $r_\mathrm{sys}$ [Mpc]'
else:  # ratio
    x_min, x_max = 0.03, 4.0
    x_ticks = [0.05, 0.1, 0.2, 0.5, 1, 2]
    x_label = r'$r_\mathrm{sys}/r_\mathrm{eq}$  (sets the Hubble term)'
y_min, y_max = 3e-3, 45.0

# Background: g_s/a0 as an exact function of xi_sys (the y-axis) -> horizontal band
_yg = np.logspace(np.log10(y_min), np.log10(y_max), 400)
_xg = np.array([x_min, x_max])
_YY = np.tile(_yg[:, None], (1, 2))
_GS = np.tile(gs_a0_of(_yg**2)[:, None], (1, 2))
_XX = np.tile(_xg[None, :], (len(_yg), 1))
ax.pcolormesh(_XX, _YY, _GS, cmap='YlOrBr', alpha=0.55, zorder=0,
              shading='gouraud', vmin=0.0, vmax=5.6)

# anomaly peak line xi_sys = 1 (xi^2 = 1, g_s maximal)
ax.axhline(1.0, color='#7a3b00', lw=1.2, ls='--', zorder=2)
ax.text(x_min*1.15, 1.16, r'anomaly peak  $\xi_s^2=1$  ($g_s^{\max}\simeq5.5\,a_0$)',
        color='#7a3b00', fontsize=8.5, va='bottom', ha='left', zorder=3)

if X_KIND == 'ratio':
    _rr = np.logspace(np.log10(x_min), np.log10(x_max), 200)
    _floor = np.sqrt((1.0/12.0) * _rr**3)          # xi_sys with 1/s^3 -> 0 (pure Hubble term)
    ax.plot(_rr, _floor, color='#2c6ea3', lw=1.4, ls=':', zorder=2)
    ax.text(0.85, np.sqrt((1.0/12.0)*0.85**3)*0.78, r'Hubble floor  $\xi_\mathrm{sys}^2=v_H^2/(12v_N^2)$  ($s\to\infty$)',
            color='#2c6ea3', fontsize=7.4, rotation=40, rotation_mode='anchor',
            ha='center', va='top', fontstyle='italic', zorder=3)
    ax.axvline(1.0, color='#888888', lw=0.9, ls='--', zorder=1)
    ax.text(1.02, y_min*1.5, r'$r_\mathrm{sys}=r_\mathrm{eq}$ ($v_N\!=\!v_H$)', color='#666',
            fontsize=7.2, rotation=90, va='bottom', ha='left', zorder=3)

stroke = [pe.withStroke(linewidth=3.0, foreground='white')]
for r in rows:
    eff = stroke if r['mk'] == '^' else []
    ax.plot(r['req'], r['xi'], r['mk'], color=r['col'], ms=r['ms'], mec='white',
            mew=0.9, zorder=6, path_effects=eff,
            label=r['lab'] if r['ann'] is None and r['mk'] != 'o' or r['mk'] in ('D', '^', 's') else None)
    if r['ann']:
        _left = r['req'] > 2.5
        ax.annotate(r['ann'], (r['req'], r['xi']), textcoords='offset points',
                    xytext=((-12, 7) if _left else (10, -3)), fontsize=7.6,
                    color='#7a5c00', style='italic', zorder=6,
                    ha='right' if _left else 'left')

# secondary y on the right: g_s/a0 at that xi (bell, so shown as tick guide)
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max)
ax.set_xlabel(x_label)
ax.set_ylabel(r'$\xi_\mathrm{sys}=\sqrt{1/s^3 + v_H^2/(12v_N^2)}$')
ax.set_xticks(x_ticks)
ax.get_xaxis().set_major_formatter(plt.matplotlib.ticker.FuncFormatter(lambda v, _: f'{v:g}'))
ax.set_yticks([0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30])
ax.get_yaxis().set_major_formatter(plt.matplotlib.ticker.FuncFormatter(lambda v, _: f'{v:g}'))

# annotate "saturated wing" (xi>>1, low g_s) and "deep wing" (xi<<1, low g_s)
ax.text(0.045, 22, 'saturated wing\n($\\xi_s^2\\gg1$, low $g_s$)', color='#555',
        fontsize=7.6, va='center', ha='left', fontstyle='italic',
        transform=ax.transData)
ax.text(3.2, 0.02, 'deep wing\n($\\xi_s^2\\ll1$, low $g_s$)', color='#555',
        fontsize=7.6, va='center', ha='center', fontstyle='italic')

ax.legend(fontsize=7.6, loc='lower right', ncol=1, handlelength=1.3,
          labelspacing=0.3, framealpha=0.92, edgecolor='#cccccc')

DST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", f"fig_xi_{X_KIND}.png")
fig.savefig(DST, dpi=200, bbox_inches='tight')
fig.savefig(DST.replace('.png', '.pdf'), bbox_inches='tight')
print("Saved:", DST)
