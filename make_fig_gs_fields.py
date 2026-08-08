"""
Three g_s/a_0 grey fields over the (r_eq, s) plane, one per probed radius r_sys.

The extra acceleration g_s = (2c/t_0) q(xi_s^2) with
    xi_s^2(s, r_eq; r_sys) = 1/s^3 + (1/12)(r_sys/r_eq)^3
depends on the probed radius r_sys through the Hubble term, so the high-g_s ridge
(xi_s^2 = 1) sits at a different place for each r_sys.  Panels: r_sys = 0.1, 0.5, 1 Mpc.
The empirical basins s_lo = k r_eq, s_hi = (k r_eq)^{-1} (k = 1.57 Mpc^-1) and the
system markers are the same in every panel (observed positions); only the field changes.
"""
import os
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

G_CODE     = 4.30091727e-6
T0         = 14.11
C_KMS      = 299792.458
CODE_TO_SI = 3.24078e-14
A0_SI      = 1.20e-10
K_AMP      = 1.57          # s_lo = K r_eq [Mpc^-1]
R_CROSS    = 1.0 / K_AMP   # basins merge onto s=1

def _q(x):
    x = np.maximum(np.asarray(x, float), 1e-12)
    delta = np.abs(x*x - 1) / (x*x + 1)
    s2 = math.sin(math.pi/3)**2 + (1 - math.sin(math.pi/3)**2) * delta
    gamma = np.arcsin(np.sqrt(np.clip(s2, 0, 1)))
    return np.cos(gamma) / np.maximum(gamma, 1e-12)

def gs_a0(xi2):
    return (2.0*C_KMS/T0 * _q(np.sqrt(xi2)) * CODE_TO_SI) / A0_SI

# systems: (label, r_eq[Mpc], s_best, marker, color, ms, is_star)
SYS = [
    (None,      0.234, 0.344, 'o', '#3498db', 6, False),
    (None,      0.301, 0.575, 'o', '#27ae60', 6, False),
    (None,      0.340, 0.585, 'o', '#e67e22', 6, False),
    (None,      0.424, 0.457, 'o', '#c0392b', 6, False),
    ('Global',  0.301, 0.495, 'D', '#222222', 8, False),
    ('Late',    0.201, 0.326, '^', '#2ecc71', 10, False),
    ('Early',   0.257, 2.076, 's', '#8e44ad', 7, False),
    ('Blue',    0.188, 0.314, 'o', '#2980b9', 7, False),
    ('Red',     0.217, 2.076, 'o', '#c0392b', 7, False),
    ('SPARC',       0.295, 4.0, '*', '#f1c40f', 15, True),
    (r'gal--gal',   0.441, 2.5, '*', '#f1c40f', 15, True),
    ('X-COP',       4.408, 1.0, '*', '#f1c40f', 16, True),
]

r_min, r_max = 0.05, 5.5
s_min, s_max = 0.25, 8.0
R_SYS = [0.1, 0.5, 1.0]

_re = np.logspace(np.log10(r_min), np.log10(r_max), 300)
_sg = np.logspace(np.log10(s_min), np.log10(s_max), 300)
RE, SG = np.meshgrid(_re, _sg)

plt.rcParams.update({'font.family': 'serif', 'font.size': 9, 'axes.linewidth': 0.55,
                     'xtick.direction': 'in', 'ytick.direction': 'in'})
fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.9), sharey=True)
fig.subplots_adjust(left=0.05, right=0.915, bottom=0.13, top=0.90, wspace=0.06)

stroke = [pe.withStroke(linewidth=3.0, foreground='white')]
r_line = np.logspace(np.log10(r_min), np.log10(r_max), 400)
r_lo = np.concatenate([r_line[r_line < R_CROSS], [R_CROSS]])
r_hi = np.concatenate([[R_CROSS], r_line[r_line > R_CROSS]])

pcm = None
for ax, rsys in zip(axes, R_SYS):
    xi2 = 1.0/SG**3 + (1.0/12.0)*(rsys/RE)**3
    pcm = ax.pcolormesh(RE, SG, gs_a0(xi2), cmap='Greys', alpha=0.85, zorder=0,
                        shading='gouraud', vmin=0.0, vmax=5.6)
    # xi^2 = 1 ridge (high-g_s locus) for this r_sys
    _inner = 1.0 - (1.0/12.0)*(rsys/_re)**3
    _ridge = np.where(_inner > 0, np.maximum(_inner, 1e-9)**(-1.0/3.0), np.nan)
    ax.plot(_re, _ridge, color='#c0392b', lw=1.4, zorder=3)
    # empirical basins + merge
    ax.plot(r_lo, K_AMP*r_lo,        color='#c03a2b', lw=1.6, zorder=2)
    ax.plot(r_lo, 1.0/(K_AMP*r_lo),  color='#505050', lw=1.6, zorder=2)
    ax.plot(r_hi, np.ones_like(r_hi), color='#505050', lw=1.8, zorder=2)
    ax.axhline(1.0, color='#888', lw=0.9, ls='--', zorder=1)
    # system markers (same in every panel)
    for lab, req, s, mk, col, ms, star in SYS:
        eff = stroke if mk == '^' else []
        mec = '#7a5c00' if star else 'white'
        ax.plot(req, s, mk, color=col, ms=ms, mec=mec, mew=0.9, zorder=6, path_effects=eff)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlim(r_min, r_max); ax.set_ylim(s_min, s_max)
    ax.set_xticks([0.05, 0.1, 0.2, 0.5, 1, 2, 5])
    ax.get_xaxis().set_major_formatter(plt.matplotlib.ticker.FuncFormatter(lambda v, _: f'{v:g}'))
    ax.set_xlabel(r'dynamical radius $r_\mathrm{eq}$ [Mpc]')
    ax.set_title(rf'$r_\mathrm{{sys}}={rsys:g}$ Mpc', fontsize=11)
    ax.text(0.04, 0.94, r'ridge $\xi_s^2=1$', transform=ax.transAxes, color='#c0392b',
            fontsize=7.5, va='top', ha='left', fontstyle='italic')

axes[0].set_yticks([0.3, 0.4, 0.5, 0.7, 1, 1.5, 2, 3, 4, 6, 8])
axes[0].get_yaxis().set_major_formatter(plt.matplotlib.ticker.FuncFormatter(lambda v, _: f'{v:g}'))
axes[0].set_ylabel(r'neighbourhood parameter $s$')

cax = fig.add_axes([0.925, 0.13, 0.014, 0.77])
cb = fig.colorbar(pcm, cax=cax)
cb.set_label(r'$g_s/a_0$', fontsize=10)

DST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "fig_gs_fields.png")
fig.savefig(DST, dpi=200, bbox_inches='tight')
fig.savefig(DST.replace('.png', '.pdf'), bbox_inches='tight')
print("Saved:", DST)
