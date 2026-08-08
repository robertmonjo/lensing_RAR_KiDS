"""
Publication-quality two-panel figure for sec:two_regimes — vertical layout.

Panel (a): xi2_s decomposition vs s   [log-log, r=1 Mpc, M*=3.2e10 Msun]
Panel (b): g_cosm/a0 vs r [Mpc]       [log-log, 8 s values, thin->thick lw]
"""
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

import os as _os
_OUT = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "outputs")
_os.makedirs(_OUT, exist_ok=True)
DST = _os.path.join(_OUT, "figA1_regimes.png")

# ── constants ──────────────────────────────────────────────────────────────────
G_CODE     = 4.30091727e-6
T0         = 14.11
C_KMS      = 299792.458
CODE_TO_SI = 3.24078e-14
A0         = 1.20e-10
SIN2_GU    = math.sin(math.pi / 3) ** 2

# ── HMG functions ─────────────────────────────────────────────────────────────
def q_func(xi2):
    xi2   = np.asarray(xi2, float)
    delta = np.abs(1.0 - xi2) / (xi2 + 1.0)
    s2    = SIN2_GU + (1.0 - SIN2_GU) * np.clip(delta, 0.0, 1.0)
    gs    = np.arcsin(np.sqrt(np.clip(s2, 0.0, 1.0)))
    return np.cos(gs) / np.maximum(gs, 1e-10)

def xi2_func(s, r_kpc, M_star):
    v_H = r_kpc / T0
    v_N = np.sqrt(G_CODE * M_star / r_kpc)
    return 1.0 / s**3 + v_H**2 / (12.0 * v_N**2)

def g_cosm_a0(s, r_kpc, M_star):
    return (2.0 * C_KMS / T0) * q_func(xi2_func(s, r_kpc, M_star)) * CODE_TO_SI / A0

# ── Panel A: reference values at r=1 Mpc ─────────────────────────────────────
r_ref, M_ref = 1000.0, 3.2e10
hub     = (r_ref / T0)**2 / (12.0 * G_CODE * M_ref / r_ref)
s_cross = hub ** (-1.0 / 3.0)
q_max   = math.cos(math.pi / 3) / (math.pi / 3)
g_max   = (2.0 * C_KMS / T0) * q_max * CODE_TO_SI / A0

# Transition radii for panel (b)
C_seq = (12.0 * G_CODE * M_ref * T0**2) ** (1.0 / 3.0)  # kpc
s_min_B = 0.3   # smallest s in panel (b)
s_max_B = 3.0   # largest s in panel (b)
r_all_hub = C_seq / (s_min_B * 1000)   # Mpc: beyond this ALL 8 curves are HUB-dominated
r_all_nh  = C_seq / (s_max_B * 1000)   # Mpc: below this ALL 8 curves are NH-dominated

print(f"hub={hub:.3f}  s_cross={s_cross:.3f}  g_max={g_max:.2f} a0")
print(f"C_seq={C_seq:.1f} kpc  r_all_hub(s=0.3)={r_all_hub:.3f} Mpc  r_all_nh(s=3.0)={r_all_nh:.3f} Mpc")

s_A  = np.logspace(np.log10(0.28), np.log10(3.8), 600)
nb_A = 1.0 / s_A**3
xi_A = nb_A + hub

# ── Panel B: 8 curves, thin->thick with increasing s ─────────────────────────
r_Mpc_B = np.logspace(np.log10(0.030), np.log10(5.0), 500)
r_kpc_B = r_Mpc_B * 1000.0
s_vals   = [0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 2.5, 3.0]
s_labels = [r'$s=0.3$', r'$s=0.5$', r'$s=0.7$', r'$s=1.0$',
            r'$s=1.5$', r'$s=2.0$', r'$s=2.5$', r'$s=3.0$']
# Rainbow: s=0.3 → red (h=0), s=3.0 → violet (h=0.80)
colors_8  = [mpl.colors.hsv_to_rgb([h, 0.88, 0.82]) for h in np.linspace(0.0, 0.80, 8)]
# Line widths: thin (s=0.3) → thick (s=3.0); thick lines drawn first (behind thin lines)
lw_vals   = np.linspace(0.7, 3.2, 8)

print(f"\nPanel (b) curves:")
for sv, lw in zip(s_vals, lw_vals):
    rt = C_seq / (sv * 1000)
    print(f"  s={sv}  lw={lw:.2f}  r_trans={rt:.3f} Mpc")

# ── Matplotlib style ──────────────────────────────────────────────────────────
mpl.rcParams.update({
    'font.family'        : 'serif',
    'font.size'          : 9,
    'axes.linewidth'     : 0.55,
    'axes.spines.top'    : True,
    'axes.spines.right'  : True,
    'xtick.direction'    : 'in',
    'ytick.direction'    : 'in',
    'xtick.minor.visible': False,
    'ytick.minor.visible': False,
    'xtick.top'          : False,
    'ytick.right'        : False,
    'legend.frameon'     : True,
    'legend.edgecolor'   : '#cccccc',
    'legend.framealpha'  : 0.92,
})

C_NB  = '#c03a2b'
C_HUB = '#2471a3'
C_TOT = '#222222'

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5.5, 7.0))
fig.subplots_adjust(left=0.12, right=0.97, bottom=0.07, top=0.97, hspace=0.12)

# ══════════════════════════════════════════════════════════════════════════════
# Panel (a)  — xi2_s decomposition
# ══════════════════════════════════════════════════════════════════════════════
ax1.axvspan(0.28,  s_cross, alpha=0.09, color=C_NB,  zorder=0, lw=0)
ax1.axvspan(s_cross, 3.8,   alpha=0.09, color=C_HUB, zorder=0, lw=0)
ax1.axvline(s_cross, color='#888888', lw=0.9, ls=':', zorder=1)

ax1.loglog(s_A, nb_A, color=C_NB,  lw=2.0, label=r'$1/s^3$')
ax1.axhline(hub, color=C_HUB, lw=2.0, ls='--', label=r'$v_H^2\!/12v_N^2$')
ax1.loglog(s_A, xi_A, color=C_TOT, lw=2.4, label=r'$\xi_s^2$ (total)')

ax1.annotate(
    rf'$s_\mathrm{{eq}}\approx{s_cross:.2f}$',
    xy=(s_cross, hub * 1.35),
    xytext=(s_cross * 1.35, hub * 3.5),
    fontsize=8, color='#666666',
    arrowprops=dict(arrowstyle='->', color='#888888', lw=0.8),
)

# Regime labels in axes-fraction coordinates:
#   NH: axes (0.10, 0.32) → data (s≈0.38, xi2≈0.46) — in red zone, well below hub
#   HUB: axes (0.80, 0.45) → data (s≈1.75, xi2≈1.58) — in blue zone, below legend area
ax1.text(0.19, 0.32, 'neighbourhood-\ndominated',
         transform=ax1.transAxes, color=C_NB, fontsize=7.5,
         va='center', ha='center', fontstyle='italic')
ax1.text(0.80, 0.45, 'Hubble-\ndominated',
         transform=ax1.transAxes, color=C_HUB, fontsize=7.5,
         va='center', ha='center', fontstyle='italic')

ax1.set_xlabel(r'neighbourhood parameter $s$')
ax1.set_ylabel(r'$\xi_s^2$ components')
ax1.set_xlim(0.28, 3.8)
ax1.set_ylim(0.05, 50)

xt_A = [0.3, 0.4, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0]
ax1.set_xticks(xt_A)
ax1.set_xticklabels([f'{x:g}' for x in xt_A])
ax1.xaxis.set_minor_locator(mticker.NullLocator())

ax1.legend(fontsize=8.5, framealpha=0.9, loc='upper right')
ax1.set_title(r'$r=1\,\mathrm{Mpc},\;M_\star=3.2\times10^{10}\,M_\odot$',
              fontsize=8, color='#555555', pad=3, loc='center')
ax1.text(-0.09, 1.04, '(a)', transform=ax1.transAxes,
         fontsize=11, fontweight='bold', va='bottom', ha='left')

# ══════════════════════════════════════════════════════════════════════════════
# Panel (b)  — g_cosm/a0 vs r, 8 s values, thin→thick line widths
# ══════════════════════════════════════════════════════════════════════════════

# KiDS weak-lensing band
KIDS_RMIN, KIDS_RMAX = 0.04, 2.0
ax2.axvspan(KIDS_RMIN, KIDS_RMAX, alpha=0.10, color='#e8a838', zorder=0, lw=0)
ax2.text(0.40, 0.40, 'KiDS\nweak-lensing', color='#a07020',
         transform=ax2.transAxes,
         fontsize=7.5, va='center', ha='center', fontstyle='italic')

# Compute all curves, then plot thick→thin (thick behind, thin on top)
curves_B = []
for sv, lbl, clr, lw in zip(s_vals, s_labels, colors_8, lw_vals):
    gc = np.array([g_cosm_a0(sv, r, M_ref) for r in r_kpc_B])
    curves_B.append((gc, clr, lw, lbl))

for gc, clr, lw, lbl in reversed(curves_B):   # thick first → behind thin lines
    ax2.loglog(r_Mpc_B, gc, color=clr, lw=lw, label=lbl)

# Reference lines
ax2.axhline(1.0, color='#888888', lw=0.8, ls='--', zorder=1)
ax2.text(4.7, 1.10, r'$a_0$', color='#888888', fontsize=8, va='bottom', ha='right')
ax2.axhline(g_max, color='#888888', lw=0.6, ls=':', zorder=1)
ax2.text(4.7, g_max * 0.88, r'$g_\mathrm{cosm}^\mathrm{max}$',
         color='#888888', fontsize=7.5, va='top', ha='right')

ax2.set_xlabel(r'projected radius $r\;[\mathrm{Mpc}]$')
ax2.set_ylabel(r'$g_\mathrm{cosm}\;/\,a_0$')
ax2.set_xlim(0.030, 5.0)
ax2.set_ylim(0.10, 8.0)

xt_B = [0.05, 0.1, 0.2, 0.5, 1, 2, 5]
ax2.set_xticks(xt_B)
ax2.set_xticklabels([f'{x:g}' for x in xt_B])
ax2.xaxis.set_minor_locator(mticker.NullLocator())

yt_B = [0.2, 0.5, 1, 2, 5]
ax2.set_yticks(yt_B)
ax2.set_yticklabels([f'{y:g}' for y in yt_B])
ax2.yaxis.set_minor_locator(mticker.NullLocator())

# Legend in forward order (thin=s=0.3 first at top)
handles, labels_leg = ax2.get_legend_handles_labels()
ax2.legend(handles[::-1], labels_leg[::-1], fontsize=7.5, loc='lower left',
           ncol=2, handlelength=1.4, columnspacing=0.8, labelspacing=0.25)
ax2.text(-0.09, 1.04, '(b)', transform=ax2.transAxes,
         fontsize=11, fontweight='bold', va='bottom', ha='left')

# ── save ──────────────────────────────────────────────────────────────────────
fig.savefig(DST, dpi=200, bbox_inches='tight')
fig.savefig(DST.replace('.png', '.pdf'), bbox_inches='tight')
print(f"Saved: {DST}")
plt.close(fig)
