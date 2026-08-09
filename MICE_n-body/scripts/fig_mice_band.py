"""fig_mice_band.py
Diagnostic figure: MICE2 RAR band comparison.

3 × 2 layout:
  Rows 1–2: four stellar-mass bins (B21 Fig. 9 style)
  Row  3  : morphological/colour bins (Sérsic index, observed colour)

Each mass-bin panel shows:
  - 1:1 Newton line  (g_obs = g_bar)
  - Wide P16–P84 band  (individual NFW profiles; our first approach)
  - Narrow B21 band  (ztrue lower vs zphotoz upper isolation; B21 Sect. 5.3)
  - z_true stacked mean line
  - KiDS-1000 data  (Brouwer+2021 Fig. 9)

Each morphology panel shows:
  - 1:1 Newton line
  - P16–P84 band per morphological sub-sample (our NFW stacking)
  - KiDS morphology data  (Brouwer+2021 Fig. 8)

Output: ../fig_mice_band.pdf

Usage:
    cd MICE_n-body/scripts
    python fig_mice_band.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
HERE           = Path(__file__).parent           # MICE_n-body/scripts/
DATA_DIR       = HERE.parent / "data"            # MICE_n-body/data/
KIDS_MORPH_DIR = HERE.parent.parent / "data"     # KiDS Fig-8 files
OUT_PDF        = HERE.parent / "fig_mice_band.pdf"

# ── Constants ──────────────────────────────────────────────────────────────────
# B21 Sect. 5.3: MICE isolation unreliable at r < 0.3 Mpc; hook/noise cut at 1.50 Mpc
R_MIN_VALID_MPC = 0.30
R_MAX_VALID_MPC = 1.50

# gbar equivalents of R_MIN/R_MAX for morphological panels (M*~4e10 Msun average)
GBAR_MORPH_MAX = 6.2e-14   # ≈ R_MIN = 0.30 Mpc
GBAR_MORPH_MIN = 2.5e-15   # ≈ R_MAX = 1.50 Mpc

G_PC   = 4.52e-30   # pc^3 M_sun^-1 s^-2
PC_M   = 3.086e16   # m/pc
ESD2G  = 4.0 * G_PC * PC_M   # M_sun/pc^2 -> m/s^2

# ── KiDS-1000 mass-bin data (Brouwer+2021 Fig. 9) — cols: r_Mpc, gobs, lo, hi ─
# Units: r_Mpc [Mpc], gobs/lo/hi [m s⁻²].  Loaded from hardcoded arrays (same
# source as hmg_kids_v2.py BINS) to avoid dependency on external data files.
KIDS_BINS = [
  np.array([
    [3.5390e-02,8.9678e-12,6.1374e-12,1.9384e-11],[4.8108e-02,8.7469e-12,7.3406e-12,1.3834e-11],
    [6.5396e-02,8.0130e-12,6.7410e-12,1.0456e-11],[8.8896e-02,6.2018e-12,5.1260e-12,7.8323e-12],
    [1.2084e-01,4.3587e-12,3.6570e-12,5.3527e-12],[1.6427e-01,2.9690e-12,2.5314e-12,3.5749e-12],
    [2.2330e-01,2.0745e-12,1.7913e-12,2.4488e-12],[3.0354e-01,1.5605e-12,1.3583e-12,1.8130e-12],
    [4.1262e-01,1.2783e-12,1.1054e-12,1.4809e-12],[5.6090e-01,1.0607e-12,9.1520e-13,1.2117e-12],
    [7.6247e-01,7.8897e-13,6.7724e-13,9.0410e-13],[1.0365e+00,5.2832e-13,4.5092e-13,6.0996e-13],
    [1.4089e+00,3.3855e-13,2.8545e-13,3.9559e-13],[1.9152e+00,2.1428e-13,1.7641e-13,2.5873e-13],
    [2.6035e+00,1.3393e-13,1.0657e-13,1.7144e-13],
  ]),
  np.array([
    [3.5390e-02,1.5240e-11,9.8535e-12,3.3211e-11],[4.8108e-02,1.4124e-11,1.1616e-11,2.3146e-11],
    [6.5396e-02,1.2871e-11,1.1021e-11,1.7090e-11],[8.8896e-02,1.0679e-11,9.1144e-12,1.3259e-11],
    [1.2084e-01,8.6016e-12,7.5652e-12,1.0177e-11],[1.6427e-01,7.1304e-12,6.3996e-12,8.1249e-12],
    [2.2330e-01,5.7500e-12,5.1995e-12,6.4133e-12],[3.0354e-01,4.3201e-12,3.9354e-12,4.7609e-12],
    [4.1262e-01,3.0574e-12,2.7931e-12,3.3592e-12],[5.6090e-01,2.0942e-12,1.9101e-12,2.2974e-12],
    [7.6247e-01,1.3954e-12,1.2591e-12,1.5433e-12],[1.0365e+00,8.8727e-13,7.8691e-13,9.9376e-13],
    [1.4089e+00,5.4361e-13,4.7481e-13,6.1821e-13],[1.9152e+00,3.2944e-13,2.8243e-13,3.8415e-13],
    [2.6035e+00,1.9842e-13,1.6559e-13,2.4103e-13],
  ]),
  np.array([
    [3.5390e-02,1.6935e-11,1.1562e-11,3.5642e-11],[4.8108e-02,1.6440e-11,1.3986e-11,2.5662e-11],
    [6.5396e-02,1.5090e-11,1.3193e-11,1.9514e-11],[8.8896e-02,1.2426e-11,1.0795e-11,1.5236e-11],
    [1.2084e-01,9.8112e-12,8.6715e-12,1.1521e-11],[1.6427e-01,7.5314e-12,6.7595e-12,8.5855e-12],
    [2.2330e-01,5.9140e-12,5.3595e-12,6.6054e-12],[3.0354e-01,4.6728e-12,4.2629e-12,5.1444e-12],
    [4.1262e-01,3.5961e-12,3.3011e-12,3.9225e-12],[5.6090e-01,2.7037e-12,2.4906e-12,2.9365e-12],
    [7.6247e-01,1.9644e-12,1.8099e-12,2.1292e-12],[1.0365e+00,1.3641e-12,1.2494e-12,1.4839e-12],
    [1.4089e+00,9.0829e-13,8.1959e-13,1.0011e-12],[1.9152e+00,5.8463e-13,5.1449e-13,6.6209e-13],
    [2.6035e+00,3.6568e-13,3.1151e-13,4.3290e-13],
  ]),
  np.array([
    [3.5390e-02,1.9524e-11,1.4866e-11,3.2192e-11],[4.8108e-02,2.0661e-11,1.8503e-11,2.6481e-11],
    [6.5396e-02,2.0456e-11,1.8438e-11,2.3269e-11],[8.8896e-02,1.6525e-11,1.4850e-11,1.8635e-11],
    [1.2084e-01,1.2646e-11,1.1571e-11,1.3953e-11],[1.6427e-01,1.0283e-11,9.5248e-12,1.1147e-11],
    [2.2330e-01,7.8173e-12,7.2633e-12,8.4194e-12],[3.0354e-01,5.6494e-12,5.2664e-12,6.0550e-12],
    [4.1262e-01,4.2016e-12,3.9093e-12,4.5022e-12],[5.6090e-01,3.0784e-12,2.8632e-12,3.3035e-12],
    [7.6247e-01,2.1934e-12,2.0376e-12,2.3523e-12],[1.0365e+00,1.5082e-12,1.3923e-12,1.6282e-12],
    [1.4089e+00,9.8307e-13,8.9601e-13,1.0749e-12],[1.9152e+00,6.2200e-13,5.5517e-13,6.9798e-13],
    [2.6035e+00,3.8775e-13,3.3420e-13,4.5779e-13],
  ]),
]

MSTAR = [1.5e10, 3.2e10, 4.6e10, 8.9e10]
BIN_LABELS = [
    r"Bin 1 — $M_\star=1.5\times10^{10}\,M_\odot$",
    r"Bin 2 — $M_\star=3.2\times10^{10}\,M_\odot$",
    r"Bin 3 — $M_\star=4.6\times10^{10}\,M_\odot$",
    r"Bin 4 — $M_\star=8.9\times10^{10}\,M_\odot$",
]
BIN_COLORS = ["#3498db", "#27ae60", "#e67e22", "#c0392b"]

# ── P16-P84 NFW band (hardcoded from the MICE run: approach A) ───────────────────
# Cols: log10_gbar, log10_gobs_mean, log10_gobs_p16, log10_gobs_p84
_MICE_BAND = [
  np.array([  # Bin 1  (M* = 1.5e10 Msun,  13 656 isolated galaxies)
    [-11.642547,-10.684008,-10.818100,-10.582857],[-11.928261,-10.769792,-10.922257,-10.658843],
    [-12.213976,-10.874314,-11.046218,-10.753594],[-12.499690,-10.997674,-11.189073,-10.868370],
    [-12.785404,-11.139202,-11.349390,-11.002593],[-13.071118,-11.297634,-11.525111,-11.155084],
    [-13.356833,-11.471342,-11.714476,-11.323804],[-13.642547,-11.658540,-11.915585,-11.506916],
    [-13.928261,-11.857453,-12.126564,-11.702615],[-14.213976,-12.066419,-12.345926,-11.909090],
    [-14.499690,-12.283950,-12.572372,-12.124679],[-14.785404,-12.508756,-12.804952,-12.348075],
    [-15.071118,-12.739736,-13.042454,-12.577909],[-15.356833,-12.975964,-13.284295,-12.813279],
    [-15.642547,-13.216671,-13.529824,-13.053296],
  ]),
  np.array([  # Bin 2  (M* = 3.2e10 Msun,  2 320 isolated galaxies)
    [-11.154567,-10.486310,-10.597948,-10.397603],[-11.440282,-10.547612,-10.675452,-10.449142],
    [-11.725996,-10.626017,-10.772079,-10.517601],[-12.011710,-10.722805,-10.889544,-10.605086],
    [-12.297425,-10.838420,-11.024929,-10.710948],[-12.583139,-10.972491,-11.178919,-10.837316],
    [-12.868853,-11.123980,-11.348796,-10.982880],[-13.154567,-11.291391,-11.533223,-11.145141],
    [-13.440282,-11.472997,-11.730437,-11.322573],[-13.725996,-11.667016,-11.937929,-11.513463],
    [-14.011710,-11.871746,-12.154656,-11.715797],[-14.297425,-12.085636,-12.378924,-11.928133],
    [-14.583139,-12.307322,-12.609230,-12.148546],[-14.868853,-12.535633,-12.844816,-12.375882],
    [-15.154567,-12.769577,-13.085041,-12.609066],
  ]),
  np.array([  # Bin 3  (M* = 4.6e10 Msun,  1 019 isolated galaxies)
    [-10.932735,-10.376085,-10.490587,-10.279201],[-11.218450,-10.423514,-10.553973,-10.317196],
    [-11.504164,-10.486012,-10.635244,-10.368698],[-11.789878,-10.565432,-10.736272,-10.437006],
    [-12.075592,-10.662915,-10.857191,-10.523936],[-12.361307,-10.778799,-10.997332,-10.630179],
    [-12.647021,-10.912647,-11.155654,-10.756395],[-12.932735,-11.063409,-11.329090,-10.901609],
    [-13.218450,-11.229630,-11.516768,-11.063872],[-13.504164,-11.409659,-11.716385,-11.241525],
    [-13.789878,-11.601824,-11.926114,-11.432436],[-14.075592,-11.804540,-12.144383,-11.634747],
    [-14.361307,-12.016366,-12.370047,-11.846816],[-14.647021,-12.236029,-12.601820,-12.067112],
    [-14.932735,-12.462418,-12.838671,-12.294349],
  ]),
  np.array([  # Bin 4  (M* = 8.9e10 Msun,   883 isolated galaxies)
    [-10.681732,-10.253198,-10.393875,-10.150319],[-10.967446,-10.286357,-10.444575,-10.173344],
    [-11.253160,-10.331577,-10.511939,-10.206511],[-11.538874,-10.391072,-10.596910,-10.253174],
    [-11.824589,-10.466658,-10.701249,-10.315686],[-12.110303,-10.559569,-10.825438,-10.395390],
    [-12.396017,-10.670351,-10.968333,-10.496192],[-12.681732,-10.798874,-11.129492,-10.617358],
    [-12.967446,-10.944415,-11.306551,-10.756758],[-13.253160,-11.105810,-11.496986,-10.913911],
    [-13.538874,-11.281609,-11.698350,-11.086984],[-13.824589,-11.470228,-11.909856,-11.273893],
    [-14.110303,-11.670076,-12.129832,-11.472852],[-14.396017,-11.879641,-12.356847,-11.682066],
    [-14.681732,-12.097549,-12.589765,-11.899952],
  ]),
]
NFW_PERBIN = []
for _d in _MICE_BAND:
    gb   = 10.0 ** _d[:, 0]
    go   = 10.0 ** _d[:, 1]
    p16  = 10.0 ** _d[:, 2]
    p84  = 10.0 ** _d[:, 3]
    NFW_PERBIN.append((gb, go, p16, p84))


# ── Load B21 band ──────────────────────────────────────────────────────────────
def load_b21_band(b):
    """Return (gb, go_ztrue, go_zphotoz) in SI, clipped to valid r range, or None."""
    f = DATA_DIR / f"rar_band_b21_bin{b}.txt"
    if not f.exists():
        return None
    d = np.loadtxt(f, comments="#")
    # col 0 = r_Mpc; keep only valid range (B21: isolation unreliable at r<0.3 Mpc)
    mask = (d[:, 0] >= R_MIN_VALID_MPC) & (d[:, 0] <= R_MAX_VALID_MPC)
    d = d[mask]
    if len(d) == 0:
        return None
    return 10.0**d[:, 1], 10.0**d[:, 2], 10.0**d[:, 3]


B21_PERBIN = [load_b21_band(b) for b in range(1, 5)]


# ── KiDS mass-bin g_bar: computed from r_Mpc and M* (Keplerian) ───────────────
G_SI  = 6.674e-11
MPC_M = 3.0857e22
MSUN  = 1.989e30

def kids_gbar(r_mpc, ms_msun):
    return G_SI * ms_msun * MSUN / (r_mpc * MPC_M)**2

KIDS_PERBIN = []
for _bdata, _ms in zip(KIDS_BINS, MSTAR):
    _gb = kids_gbar(_bdata[:, 0], _ms)
    KIDS_PERBIN.append((_gb, _bdata[:, 1], _bdata[:, 2], _bdata[:, 3]))


# ── Load KiDS morphology data ──────────────────────────────────────────────────
def load_kids_morph(fname):
    f = KIDS_MORPH_DIR / fname
    if not f.exists():
        return None
    d = np.loadtxt(f, comments="#")
    gb   = d[:, 0]
    bias = d[:, 4]
    go   = ESD2G * d[:, 1] / bias
    gerr = ESD2G * d[:, 3] / bias
    ok   = go > 0
    return gb[ok], go[ok], gerr[ok]


MORPH_FILES = [
    ("Fig-8_RAR-KiDS-isolated_Sersicbin_1.txt", "Late ($n<2.5$)",  "^", "#2ecc71"),
    ("Fig-8_RAR-KiDS-isolated_Sersicbin_2.txt", "Early ($n\\geq2.5$)", "s", "#8e44ad"),
    ("Fig-8_RAR-KiDS-isolated_Colorbin_1.txt",  "Blue",            "o", "#3498db"),
    ("Fig-8_RAR-KiDS-isolated_Colorbin_2.txt",  "Red",             "o", "#e91e63"),
]


# ── Load MICE morphology band ──────────────────────────────────────────────────
def load_morph_mice(fname, nbins=18):
    """Load allbins file -> (gb, go, p16, p84) in SI (log10 input)."""
    f = DATA_DIR / fname
    if not f.exists():
        return None
    d = np.loadtxt(f, comments="#")
    lgb = d[:, 0]; lgo = d[:, 1]; lp16 = d[:, 2]; lp84 = d[:, 3]
    edges = np.linspace(lgb.min(), lgb.max(), nbins + 1)
    bc, go_b, p16_b, p84_b = [], [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (lgb >= lo) & (lgb < hi)
        if m.sum() == 0:
            continue
        bc.append(0.5 * (lo + hi))
        go_b.append(np.median(lgo[m]))
        p16_b.append(np.median(lp16[m]))
        p84_b.append(np.median(lp84[m]))
    bc = np.array(bc)
    return 10**bc, 10**np.array(go_b), 10**np.array(p16_b), 10**np.array(p84_b)


def _clip_morph(data):
    """Clip morphological MICE band to valid gbar range (same as hmg_kids_v2.py)."""
    if data is None:
        return None
    gb, go, p16, p84 = data
    m = (gb >= GBAR_MORPH_MIN) & (gb <= GBAR_MORPH_MAX)
    return gb[m], go[m], p16[m], p84[m]


MORPH_MICE = [
    _clip_morph(load_morph_mice("morph_sersic_late_allbins.txt")),
    _clip_morph(load_morph_mice("morph_sersic_early_allbins.txt")),
    _clip_morph(load_morph_mice("morph_color_blue_allbins.txt")),
    _clip_morph(load_morph_mice("morph_color_red_allbins.txt")),
]
MORPH_MICE_COLORS = ["#f1c40f", "#d4ac0d", "#f1c40f", "#d4ac0d"]
MORPH_MICE_LS     = ["-", "--", "-", "--"]
MORPH_MICE_LABELS = [
    r"MICE disc ($B/T<0.5$)", r"MICE spheroid ($B/T\geq0.5$)",
    r"MICE blue ($g{-}r<0.6$)", r"MICE red ($g{-}r\geq0.6$)",
]


# ── Plot limits ────────────────────────────────────────────────────────────────
GBAR_MIN, GBAR_MAX = 1e-16, 3e-11
YLIM = (1e-14, 1e-10)


def _add_newton(ax):
    ax.plot([GBAR_MIN, GBAR_MAX], [GBAR_MIN, GBAR_MAX],
            color="#bbbbbb", lw=1.0, ls="-", zorder=0, label="Newton 1:1")


def _style_ax(ax, title):
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(GBAR_MIN, GBAR_MAX)
    ax.set_ylim(*YLIM)
    ax.grid(True, which="major", color="#e8e8e8", lw=0.5, zorder=0)
    ax.text(0.04, 0.97, title, transform=ax.transAxes,
            ha="left", va="top", fontsize=8,
            bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none", alpha=0.85))


# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(3, 2, figsize=(10, 12), sharex=True, sharey=True)
fig.subplots_adjust(hspace=0.08, wspace=0.06)

# ── Rows 1–2: four stellar-mass bins ─────────────────────────────────────────
for i in range(4):
    row, col = divmod(i, 2)
    ax = axes[row, col]
    col_kids = BIN_COLORS[i]

    _add_newton(ax)

    # Wide NFW P16–P84 band (very faint)
    gb_n, go_n, p16_n, p84_n = NFW_PERBIN[i]
    ax.fill_between(gb_n, p16_n, p84_n,
                    color="#f1c40f", alpha=0.12, zorder=1,
                    label=r"P$_{16}$–P$_{84}$ NFW (approach A)")

    # B21 narrow band (yellow, same hue as wide band)
    if B21_PERBIN[i] is not None:
        gb_b, go_lo, go_hi = B21_PERBIN[i]
        ax.fill_between(gb_b, go_lo, go_hi,
                        color="#f1c40f", alpha=0.28, zorder=2,
                        label=r"B21 band ($z_\mathrm{true}$ vs $z_\mathrm{photo}$)")
        # z_true mean line
        ax.plot(gb_b, go_lo,
                color="#f1c40f", lw=2.0, ls="-", zorder=3,
                label=r"$z_\mathrm{true}$ stacked mean")

    # KiDS data
    gb_k, go_k, go_klo, go_khi = KIDS_PERBIN[i]
    ax.errorbar(gb_k, go_k,
                yerr=[go_k - go_klo, go_khi - go_k],
                fmt="o", color=col_kids, ms=4.5, lw=0.9, capsize=2, zorder=5,
                label=r"KiDS-1000 (B21 Fig.\,9)")

    _style_ax(ax, BIN_LABELS[i])
    ax.legend(fontsize=6.5, loc="lower right", framealpha=0.90,
              handlelength=2.0, labelspacing=0.25)
    if row == 1:
        ax.set_xlabel(r"$g_\mathrm{bar} = GM_\star/r^2$ [m\,s$^{-2}$]", fontsize=9)
    if col == 0:
        ax.set_ylabel(r"$g_\mathrm{obs}$ [m\,s$^{-2}$]", fontsize=9)

# ── Row 3: morphological / colour bins ────────────────────────────────────────
MORPH_PAIRS = [
    (0, 1, "Sérsic index bins",
     "Late ($n<2.5$)", "Early ($n\\geq2.5$)", "^", "s", "#2ecc71", "#8e44ad"),
    (2, 3, "Colour bins",
     "Blue (late)", "Red (early)", "o", "o", "#3498db", "#e91e63"),
]

for col, (i_l, i_e, title, lab_l, lab_e, mk_l, mk_e, c_l, c_e) in enumerate(MORPH_PAIRS):
    ax = axes[2, col]

    _add_newton(ax)

    # MICE morphology — pure band (edge lines p16/p84, no centre line), same as hmg_kids_v2.py
    mm_l = MORPH_MICE[i_l]
    mm_e = MORPH_MICE[i_e]
    if mm_l is not None and mm_e is not None:
        gb_ml, go_ml, p16_ml, p84_ml = mm_l
        gb_me, go_me, p16_me, p84_me = mm_e
        # light-blue shading: region R > R_MIN (low gbar, left side)
        ax.axvspan(GBAR_MIN, float(max(gb_ml.max(), gb_me.max())),
                   color="#aed6f1", alpha=0.12, zorder=0)
        # late/disc: solid edge lines (#f1c40f)
        ax.fill_between(gb_ml, p16_ml, p84_ml,
                        color=MORPH_MICE_COLORS[i_l], alpha=0.14, zorder=1,
                        label=MORPH_MICE_LABELS[i_l])
        ax.plot(gb_ml, p16_ml, color=MORPH_MICE_COLORS[i_l], lw=0.8, ls="-",  alpha=0.55, zorder=1)
        ax.plot(gb_ml, p84_ml, color=MORPH_MICE_COLORS[i_l], lw=0.8, ls="-",  alpha=0.55, zorder=1)
        # early/spheroid: dashed edge lines (#d4ac0d)
        ax.fill_between(gb_me, p16_me, p84_me,
                        color=MORPH_MICE_COLORS[i_e], alpha=0.14, zorder=1,
                        label=MORPH_MICE_LABELS[i_e])
        ax.plot(gb_me, p16_me, color=MORPH_MICE_COLORS[i_e], lw=0.8, ls="--", alpha=0.55, zorder=1)
        ax.plot(gb_me, p84_me, color=MORPH_MICE_COLORS[i_e], lw=0.8, ls="--", alpha=0.55, zorder=1)

    # KiDS morphology data
    for fname, lab, mk, c in [
        (MORPH_FILES[i_l][0], lab_l, mk_l, c_l),
        (MORPH_FILES[i_e][0], lab_e, mk_e, c_e),
    ]:
        kd = load_kids_morph(fname)
        if kd is not None:
            gb_k, go_k, ge_k = kd
            ax.errorbar(gb_k, go_k, yerr=ge_k,
                        fmt=mk, color=c, ms=5, lw=0.9, capsize=3,
                        zorder=5, label=lab, alpha=0.9)

    _style_ax(ax, title)
    ax.set_xlabel(r"$g_\mathrm{bar} = GM_\star/r^2$ [m\,s$^{-2}$]", fontsize=9)
    if col == 0:
        ax.set_ylabel(r"$g_\mathrm{obs}$ [m\,s$^{-2}$]", fontsize=9)
    ax.legend(fontsize=6.5, loc="lower right", framealpha=0.90,
              handlelength=2.0, labelspacing=0.25)

# ── Shared axis labels ─────────────────────────────────────────────────────────
fig.suptitle(
    "MICE2 $\\Lambda$CDM band comparison — KiDS-1000 lensing RAR\n"
    r"Faint yellow: P$_{16}$–P$_{84}$ NFW band (this work, approach B).  "
    r"Yellow: B21 band ($z_\mathrm{true}$ vs $z_\mathrm{photo}$, clipped $r\in[0.30,1.50]$ Mpc, "
    r"Brouwer+2021 Sect.~5.3).",
    fontsize=9, y=0.995)

fig.savefig(OUT_PDF, dpi=150, bbox_inches="tight")
print(f"Saved: {OUT_PDF}")
_out_png = OUT_PDF.with_suffix(".png")
fig.savefig(_out_png, dpi=150, bbox_inches="tight")
print(f"Saved: {_out_png}")
plt.close(fig)
