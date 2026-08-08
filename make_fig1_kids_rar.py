"""
hmg_kids_v1.py
--------------
HMG (s free, global) + MOND + CDM (halo model) vs Brouwer+2021 KiDS-1000.

CDM halo model (one free param xi0 = two-halo amplitude):
  - One-halo : NFW truncated at r200          [0 extra params, M_h from Moster+13]
  - Two-halo : rho_2h(r) = b_g * rho_m * xi0 * (r/1Mpc)^{-1.8} for r > r200
               xi0 ~ 7 from LCDM (fit as 1 free param for fair comparison)
  Galaxy bias b_g(M_h) from Tinker+2010 at z~0.

SHMR prior : Moster+2013.  Conc. : Dutton & Maccio 2014.

Notation throughout: chi2_nu = chi^2 / dof
"""
from __future__ import annotations
import math, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import brentq, minimize_scalar
from pathlib import Path
import os

# ── Constants ──────────────────────────────────────────────────────────────────
G_SI       = 6.674e-11
G_CODE     = 4.30091727e-6
T0         = 14.11
C_KMS      = 299792.458
A0_CODE    = 3703.0
CODE_TO_SI = 3.24078e-14
KPC_M      = 3.0857e19
MPC_M      = 3.0857e22
MPC_TO_KPC = 1000.0
MSUN_KG    = 1.989e30
H0_SI      = 70e3 / MPC_M
RHO_CRIT   = 3*H0_SI**2/(8*math.pi*G_SI) * KPC_M**3/MSUN_KG   # Msun kpc^-3
OMEGA_M    = 0.3
RHO_M_MPC  = OMEGA_M * (RHO_CRIT * (MPC_TO_KPC)**3)            # Msun Mpc^-3
SIN2_GU    = math.sin(math.pi/3)**2
SIN2_GCEN  = 1.0

# ── KiDS-1000 data (Brouwer+2021) ─────────────────────────────────────────────
MSTAR = [1.5e10, 3.2e10, 4.6e10, 8.9e10]

def f_cold(m):
    """Cold-gas fraction (Boselli 2014): log10 f_cold = -0.69 log10(M*) + 6.63."""
    return 10.0**(-0.69*math.log10(m)+6.63)
MBAR = [m*(1.0+f_cold(m)) for m in MSTAR]   # baryonic point mass (g_bar uses this)
BIN_LABELS = [
    r"$M_\star=1.5\times10^{10}\,M_\odot$",
    r"$M_\star=3.2\times10^{10}\,M_\odot$",
    r"$M_\star=4.6\times10^{10}\,M_\odot$",
    r"$M_\star=8.9\times10^{10}\,M_\odot$",
]
# B21 Sect. 5.3: MICE limited to large r (low g_bar) by resolution.
# Inner limit: R < 0.30 Mpc → KiDS reliable, MICE resolution insufficient → exclude
# Outer limit: R > 1.50 Mpc → hook/noise in GGL signal → exclude
R_MIN_VALID_MPC = 0.30
R_MAX_VALID_MPC = 1.50

# gbar equivalents for morphological panels (all-bins, M*~4e10 Msun average)
# R_MIN=0.30 Mpc → gbar_max = G*4e10*Msun/(0.30 Mpc)^2 ≈ 6.2e-14 ms^-2
# R_MAX=1.50 Mpc → gbar_min = gbar_max*(0.30/1.50)^2       ≈ 2.5e-15 ms^-2
GBAR_MORPH_MAX = 6.2e-14
GBAR_MORPH_MIN = 2.5e-15

BINS = [
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

# ── MICE isolated per-bin data ────────────────────────────────────────────────
# B21-style band (preferred): loaded from rar_band_b21_bin{b}.txt
#   Columns: r_Mpc, log10_gbar, log10_gobs_ztrue, log10_gobs_zphotoz
#   Band = range between z_true isolation (lower) and photo-z isolation (upper)
#   Run MICE_n-body/scripts/run_photoz_pipeline.sh on Nieve to produce these files.
#
# Fallback (if B21 files not present): hardcoded NFW P16-P84 band from Nieve run
#   Columns: log10_gbar, log10_gobs_mean, log10_gobs_p16, log10_gobs_p84
_MICE_B21_DIR = Path(__file__).resolve().parent / "data"

def _load_mice_b21():
    """Try to load B21-style MICE band. Return list of (gb, go, go_lo, go_hi) in SI.
    Returns None if any file is missing."""
    result = []
    for b in range(1, 5):
        fname = _MICE_B21_DIR / f"rar_band_b21_bin{b}.txt"
        if not fname.exists():
            return None
        d = np.loadtxt(fname, comments="#")
        # keep only R_MIN ≤ r ≤ R_MAX: MICE resolution limit (inner) + hook cut (outer)
        mask = (d[:, 0] >= R_MIN_VALID_MPC) & (d[:, 0] <= R_MAX_VALID_MPC)
        d = d[mask]
        if len(d) == 0:
            return None
        # cols: r_Mpc, log10_gbar, log10_gobs_ztrue, log10_gobs_zphotoz
        gb    = 10.0 ** d[:, 1]
        go_lo = 10.0 ** d[:, 2]   # z_true (lower)
        go_hi = 10.0 ** d[:, 3]   # photo-z (upper)
        result.append((gb, go_lo, go_lo, go_hi))  # go_mean = go_lo (z_true clean signal)
    return result

_MICE_B21 = _load_mice_b21()
if _MICE_B21 is not None:
    print(f"[MICE] Loaded B21-style band from {_MICE_B21_DIR}")
else:
    print(f"[MICE] B21 band files not found in {_MICE_B21_DIR}; using fallback NFW P16-P84")

# Fallback: hardcoded NFW P16-P84 band
# Columns: log10_gbar, log10_gobs_mean, log10_gobs_p16, log10_gobs_p84
_MICE_NIEVE = [
  np.array([  # Bin 1  (M* = 1.5e10 Msun,  13656 isolated galaxies)
    [-11.642547,-10.684008,-10.818100,-10.582857],[-11.928261,-10.769792,-10.922257,-10.658843],
    [-12.213976,-10.874314,-11.046218,-10.753594],[-12.499690,-10.997674,-11.189073,-10.868370],
    [-12.785404,-11.139202,-11.349390,-11.002593],[-13.071118,-11.297634,-11.525111,-11.155084],
    [-13.356833,-11.471342,-11.714476,-11.323804],[-13.642547,-11.658540,-11.915585,-11.506916],
    [-13.928261,-11.857453,-12.126564,-11.702615],[-14.213976,-12.066419,-12.345926,-11.909090],
    [-14.499690,-12.283950,-12.572372,-12.124679],[-14.785404,-12.508756,-12.804952,-12.348075],
    [-15.071118,-12.739736,-13.042454,-12.577909],[-15.356833,-12.975964,-13.284295,-12.813279],
    [-15.642547,-13.216671,-13.529824,-13.053296],
  ]),
  np.array([  # Bin 2  (M* = 3.2e10 Msun,  2320 isolated galaxies)
    [-11.154567,-10.486310,-10.597948,-10.397603],[-11.440282,-10.547612,-10.675452,-10.449142],
    [-11.725996,-10.626017,-10.772079,-10.517601],[-12.011710,-10.722805,-10.889544,-10.605086],
    [-12.297425,-10.838420,-11.024929,-10.710948],[-12.583139,-10.972491,-11.178919,-10.837316],
    [-12.868853,-11.123980,-11.348796,-10.982880],[-13.154567,-11.291391,-11.533223,-11.145141],
    [-13.440282,-11.472997,-11.730437,-11.322573],[-13.725996,-11.667016,-11.937929,-11.513463],
    [-14.011710,-11.871746,-12.154656,-11.715797],[-14.297425,-12.085636,-12.378924,-11.928133],
    [-14.583139,-12.307322,-12.609230,-12.148546],[-14.868853,-12.535633,-12.844816,-12.375882],
    [-15.154567,-12.769577,-13.085041,-12.609066],
  ]),
  np.array([  # Bin 3  (M* = 4.6e10 Msun,  1019 isolated galaxies)
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
_MICE_NIEVE_PERBIN = []
for _d in _MICE_NIEVE:
    _gb  = 10.0**_d[:, 0]
    _go  = 10.0**_d[:, 1]
    _p16 = 10.0**_d[:, 2]
    _p84 = 10.0**_d[:, 3]
    _MICE_NIEVE_PERBIN.append((_gb, _go, _p16, _p84))

MICE_PERBIN = _MICE_B21 if _MICE_B21 is not None else _MICE_NIEVE_PERBIN

# ── MOND (simple interpolation function, a0 fixed) ────────────────────────────
def mond_gobs(gbar):
    a0 = A0_CODE * CODE_TO_SI   # 1.200e-10 m/s^2 (no free params)
    return gbar / (1.0 - np.exp(-np.sqrt(gbar / a0)))

# ── s-HMG (B18+B19, Monjo 2025) ───────────────────────────────────────────────
def _cos_over_gamma(xi2_s):
    xi2   = np.asarray(xi2_s, float)
    delta = np.abs(1.0 - xi2) / (1.0 + xi2 + 1e-30)
    s2    = SIN2_GU + (SIN2_GCEN - SIN2_GU) * np.clip(delta, 0.0, 1.0)
    gs    = np.arcsin(np.sqrt(np.clip(s2, 0.0, 1.0)))
    return np.cos(gs) / np.maximum(gs, 1e-10)

def gobs_hmg(r_kpc, ms, s):
    r   = np.asarray(r_kpc, float)
    v2n = G_CODE * ms / r
    vh2 = (r / T0)**2
    xi2 = 1.0/s**3 + vh2/(12.0*v2n + 1e-60)
    a_n = v2n / r
    return np.sqrt(np.maximum(a_n*(a_n + 2.0*C_KMS/T0*_cos_over_gamma(xi2)), 0.0)) * CODE_TO_SI

# ── Morphological subsets (deep-limit, Brouwer+2021 Fig-8 ESD files) ──────────
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
_G_PC   = 4.52e-30        # pc^3 Msun^-1 s^-2
_PC_M   = 3.086e16        # m/pc
ESD2G   = 4.0 * _G_PC * _PC_M

def _q_dl(x):
    delta = abs(x**2 - 1) / (x**2 + 1)
    s2 = SIN2_GU + (1.0 - SIN2_GU) * delta
    gam = math.asin(math.sqrt(max(min(s2, 1.0), 0.0)))
    return math.cos(gam) / max(gam, 1e-12)

def g_s_dl(s):
    # Morphological deep limit: xi_s^2 = 1/s^3 (Hubble term dropped).
    # The q argument is x = 1/xi_s = s^{3/2}. Since delta(x) = delta(1/x), _q_dl(s**1.5)
    # equals q(xi^2 = 1/s^3).
    return 2.0 * C_KMS / T0 * _q_dl(s**1.5) * CODE_TO_SI

def hmg_pred_dl(gbar, s):
    return np.sqrt(gbar * (gbar + g_s_dl(s)))

def chi2_dl(s, gbar, gobs, gerr):
    sig = gerr / (gobs * np.log(10))
    return float(np.sum(((np.log10(gobs) - np.log10(hmg_pred_dl(gbar, s))) / sig)**2)) / (len(gbar) - 1)

def load_morph(fname):
    d    = np.loadtxt(os.path.join(DATA_DIR, fname), comments='#')
    gbar = d[:, 0]; bias = d[:, 4]
    gobs = ESD2G * d[:, 1] / bias
    gerr = ESD2G * d[:, 3] / bias
    ok   = gobs > 0
    return gbar[ok], gobs[ok], gerr[ok]

# ── CDM: Moster+2013 SHMR ─────────────────────────────────────────────────────
_M1, _N, _BETA, _GAM = 10**11.590, 0.0351, 1.376, 0.608

def _ms_from_mh(mh):
    x = mh / _M1
    return mh * 2*_N / (x**(-_BETA) + x**_GAM)

def mhalo_from_mstar(ms):
    f = lambda lmh: math.log10(_ms_from_mh(10**lmh)) - math.log10(ms)
    return 10**brentq(f, 10.0, 15.0)

# ── CDM: NFW profile ──────────────────────────────────────────────────────────
def _r200(mh):
    return (mh / (4*math.pi/3 * 200 * RHO_CRIT))**(1/3)   # kpc

def _conc(mh):
    return 10**(0.905 - 0.101*math.log10(mh / (1e12/0.7)))

def nfw_enclosed(r_kpc, mh):
    r200 = _r200(mh); c = _conc(mh); rs = r200/c
    x    = np.asarray(r_kpc, float) / rs
    fc   = math.log(1+c) - c/(1+c)
    return mh * (np.log(1+x) - x/(1+x)) / fc

# ── CDM: galaxy bias (Tinker+2010, z~0 approximation) ────────────────────────
def _b_g(mh):
    lm = math.log10(mh)
    if   lm < 11:  return 0.65
    elif lm < 12:  return 0.65 + 0.35*(lm - 11)
    elif lm < 13:  return 1.00 + 0.70*(lm - 12)
    else:          return 1.70

# ── CDM halo model: truncated NFW + two-halo ──────────────────────────────────
# Two-halo density beyond r200:
#   rho_2h(r) = b_g * rho_m * xi0 * (r / 1 Mpc)^{-1.8}
# Enclosed two-halo mass from r200 to r:
#   M_2h(<r) = 4pi * b_g * rho_m * xi0 * (r^1.2 - r200^1.2) / 1.2   [Mpc units]
# xi0 ~ 7 from LCDM at r=1 Mpc (fitted as 1 free param).
def gobs_cdm(r_kpc_arr, ms, mh, xi0):
    r_kpc  = np.asarray(r_kpc_arr, float)
    r200   = _r200(mh)                          # kpc
    r200_m = r200 / MPC_TO_KPC                  # Mpc

    # One-halo: NFW, truncated hard at r200
    m_1h = np.minimum(nfw_enclosed(r_kpc, mh), mh)

    # Two-halo: power law, starts at r200
    r_mpc  = r_kpc / MPC_TO_KPC
    bg     = _b_g(mh)
    m_2h   = np.zeros_like(r_kpc)
    mask   = r_kpc > r200
    if np.any(mask):
        rm = r_mpc[mask]
        m_2h[mask] = (4*math.pi * bg * RHO_M_MPC * xi0 / 1.2
                      * (rm**1.2 - r200_m**1.2))

    m_tot = ms + m_1h + m_2h
    return G_SI * m_tot * MSUN_KG / (r_kpc * KPC_M)**2

# ── Helpers ───────────────────────────────────────────────────────────────────
def gbar_si(r_mpc, ms):
    return G_SI * ms * MSUN_KG / (r_mpc * MPC_M)**2

def _logsig(b):
    return (np.log10(b[:,3]) - np.log10(b[:,2])) / 2.0

def _chi2_arr(pred, bdata):
    sig = _logsig(bdata)
    return float(np.sum(((np.log10(bdata[:,1]) - np.log10(pred)) / sig)**2))

def _r_kpc(bdata):
    return bdata[:,0] * MPC_TO_KPC

# ── Fits ──────────────────────────────────────────────────────────────────────

# MOND: no free params
def _chi2nu_mond():
    tot  = sum(_chi2_arr(mond_gobs(gbar_si(b[:,0], m)), b)
               for b, m in zip(BINS, MBAR))
    return tot / sum(len(b) for b in BINS)

# HMG: 1 free param s (two basins)
def _fit_hmg():
    def c2(s):
        return sum(_chi2_arr(gobs_hmg(_r_kpc(b), m, s), b)
                   for b, m in zip(BINS, MBAR))
    ndof = sum(len(b) for b in BINS)
    rA = minimize_scalar(c2, bounds=(0.10, 0.99), method="bounded")
    rB = minimize_scalar(c2, bounds=(1.0, 12.0),  method="bounded")
    r  = rA if rA.fun <= rB.fun else rB
    return r.x, r.fun/ndof

# CDM halo model: 1 free param xi0 (two-halo amplitude), M_h from Moster+13
def _fit_cdm_global(mh_list, lo=0.1, hi=30.0):
    def c2(xi0):
        return sum(_chi2_arr(gobs_cdm(_r_kpc(b), m, mh, xi0), b)
                   for b, m, mh in zip(BINS, MBAR, mh_list))
    ndof = sum(len(b) for b in BINS)
    r    = minimize_scalar(c2, bounds=(lo, hi), method="bounded")
    return r.x, r.fun/ndof

# CDM per-bin: M_h free per bin (xi0 from global), one-halo only (xi0=0) as option
def _fit_cdm_perbin(xi0_global, mh_list):
    results = []
    for b, m, mh0 in zip(BINS, MBAR, mh_list):
        rk = _r_kpc(b)
        # fit M_h in [1e10, 1e14], xi0 fixed at global best
        res = minimize_scalar(
            lambda lmh, bd=b, ms=m, rk_=rk: _chi2_arr(
                gobs_cdm(rk_, ms, 10**lmh, xi0_global), bd),
            bounds=(10.0, 14.0), method="bounded")
        results.append((10**res.x, res.fun/len(b)))
    return results

# HMG per-bin
def _fit_hmg_perbin():
    res = []
    for b, m in zip(BINS, MBAR):
        rk = _r_kpc(b)
        rA = minimize_scalar(lambda s,bd=b,ms=m,r=rk: _chi2_arr(gobs_hmg(r,ms,s),bd),
                             bounds=(0.10, 0.99), method="bounded")
        rB = minimize_scalar(lambda s,bd=b,ms=m,r=rk: _chi2_arr(gobs_hmg(r,ms,s),bd),
                             bounds=(1.0, 12.0),  method="bounded")
        rv = rA if rA.fun <= rB.fun else rB
        res.append((rv.x, rv.fun/len(b)))
    return res

# ══════════════════════════════════════════════════════════════════════════════
print("=" * 62)
print("Moster+2013 halo masses:")
MH_MOSTER = [mhalo_from_mstar(ms) for ms in MSTAR]
for ms, mh in zip(MSTAR, MH_MOSTER):
    print(f"  M*={ms:.1e}  Mh={mh:.2e}  r200={_r200(mh):.0f} kpc "
          f" c={_conc(mh):.2f}  b_g={_b_g(mh):.2f}")

print("\nMOND chi2_nu (0 free params) ...")
C2_MOND = _chi2nu_mond()
print(f"  chi2_nu = {C2_MOND:.4f}")

print("\nFitting s-HMG (1 free param, two basins) ...")
S_BEST, C2_HMG = _fit_hmg()
print(f"  s = {S_BEST:.4f},  chi2_nu = {C2_HMG:.4f}")

print("\nFitting CDM halo model (1 free param: xi0) ...")
XI0_BEST, C2_CDM = _fit_cdm_global(MH_MOSTER)
print(f"  xi0 = {XI0_BEST:.3f},  chi2_nu = {C2_CDM:.4f}")
print(f"  (xi0=7 from LCDM theory; fitted xi0={XI0_BEST:.2f})")

print("\nFitting CDM per-bin (M_h free, xi0 fixed) ...")
PB_CDM = _fit_cdm_perbin(XI0_BEST, MH_MOSTER)
for i,(mh,c2i) in enumerate(PB_CDM):
    print(f"  bin{i} M*={MSTAR[i]:.1e}: Mh_fit={mh:.2e}  chi2_nu={c2i:.3f}")

print("\nFitting HMG per-bin ...")
PB_HMG = _fit_hmg_perbin()
for i,(s,c2i) in enumerate(PB_HMG):
    print(f"  bin{i} M*={MSTAR[i]:.1e}: s={s:.3f}  chi2_nu={c2i:.3f}")

ndof = sum(len(b) for b in BINS)
print("\n" + "=" * 62)
print("Summary — global fits (1 param each):")
print(f"  MOND         (0 params):  chi2_nu = {C2_MOND:.3f}")
print(f"  HMG  s={S_BEST:.3f}  (1 param):  chi2_nu = {C2_HMG:.3f}")
print(f"  CDM  xi0={XI0_BEST:.2f} (1 param):  chi2_nu = {C2_CDM:.3f}")
print("=" * 62)

# ── Morphological fits ────────────────────────────────────────────────────────
# ── Morphology-specific r(g_bar) from MICE isolated lenses (breaks s<->1/s) ────
# Restores the Hubble term h = v_H^2/(12 v_N^2) that the deep-limit fit dropped:
#   xi_s^2 = 1/s^3 + h,  with r per RAR point supplied by the morphology-resolved
#   MICE g_bar(r) mapping (data/morph_gbar_r_mice.txt, from Nieve 09_morph_gbar_r.py).
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

def hmg_pred_full(gbar_si, s, r_kpc):
    """Full HMG for morphology: xi_s^2 = 1/s^3 + v_H^2/(12 v_N^2), r from MICE."""
    gc  = gbar_si / CODE_TO_SI
    v2n = gc * r_kpc
    vh2 = (r_kpc / T0)**2
    xi2 = 1.0/s**3 + vh2/(12.0*v2n + 1e-60)
    d   = np.abs(1.0 - xi2) / (1.0 + xi2 + 1e-30)
    s2  = 0.75 + 0.25*np.clip(d, 0.0, 1.0)          # sin^2(pi/3)=0.75, gamma_cen=pi/2
    g   = np.arcsin(np.sqrt(np.clip(s2, 0.0, 1.0)))
    q   = np.cos(g) / np.maximum(g, 1e-12)
    return np.sqrt(np.maximum(gc*(gc + 2.0*C_KMS/T0*q), 0.0)) * CODE_TO_SI

print("Fitting morphological subsets ...")
gb_late,  go_late,  ge_late  = load_morph("Fig-8_RAR-KiDS-isolated_Sersicbin_1.txt")
gb_early, go_early, ge_early = load_morph("Fig-8_RAR-KiDS-isolated_Sersicbin_2.txt")
gb_blue,  go_blue,  ge_blue  = load_morph("Fig-8_RAR-KiDS-isolated_Colorbin_1.txt")
gb_red,   go_red,   ge_red   = load_morph("Fig-8_RAR-KiDS-isolated_Colorbin_2.txt")

def _fit_morph(gbar, gobs, gerr, key):
    # Full xi_s^2 = 1/s^3 + h with r from MICE: the Hubble term breaks the deep-limit
    # s <-> 1/s degeneracy.  Scan both basins and report the preferred (lower-chi2) one.
    rk  = _r_of[key](gbar)
    sig = gerr / (gobs * np.log(10))
    def c(s):
        return float(np.sum(((np.log10(gobs) - np.log10(hmg_pred_full(gbar, s, rk))) / sig)**2)) / (len(gbar) - 1)
    lo = minimize_scalar(c, bounds=(0.15, 0.999), method='bounded')
    hi = minimize_scalar(c, bounds=(1.001, 8.0),  method='bounded')
    return (lo.x, lo.fun) if lo.fun <= hi.fun else (hi.x, hi.fun)

s_late,  c2_late  = _fit_morph(gb_late,  go_late,  ge_late,  "late")
s_early, c2_early = _fit_morph(gb_early, go_early, ge_early, "early")
s_blue,  c2_blue  = _fit_morph(gb_blue,  go_blue,  ge_blue,  "blue")
s_red,   c2_red   = _fit_morph(gb_red,   go_red,   ge_red,   "red")
print(f"  Sersic: late s={s_late:.3f} chi2={c2_late:.2f}; early s={s_early:.3f} chi2={c2_early:.2f}")
print(f"  Color:  blue s={s_blue:.3f} chi2={c2_blue:.2f};  red  s={s_red:.3f}  chi2={c2_red:.2f}")

# ── CDM for morphological subsets: r from MICE mapping, M_star = MICE median,
#    halo mass fitted per subset (xi0 fixed at the global best, as for mass bins) ──
MSTAR_MORPH = {"late": 10**9.745, "early": 10**10.154, "blue": 10**9.610, "red": 10**9.888}
MBAR_MORPH = {k: m * (1 + f_cold(m)) for k, m in MSTAR_MORPH.items()}   # baryonic (M_star + cold gas)
def _fit_mh_morph(key, gb, go, ge):
    ms = MBAR_MORPH[key]; rk = _r_of[key](gb); sig = ge / (go * np.log(10))
    def c2(lmh):
        return float(np.sum(((np.log10(go) - np.log10(gobs_cdm(rk, ms, 10**lmh, XI0_BEST))) / sig)**2))
    r = minimize_scalar(c2, bounds=(10.5, 13.5), method="bounded")
    return ms, 10**r.x, c2(r.x) / (len(gb) - 1)
MH_MORPH = {}
for _k, _gb, _go, _ge in [("late", gb_late, go_late, ge_late), ("early", gb_early, go_early, ge_early),
                          ("blue", gb_blue, go_blue, ge_blue), ("red", gb_red, go_red, ge_red)]:
    MH_MORPH[_k] = _fit_mh_morph(_k, _gb, _go, _ge)
    print(f"  CDM {_k:5s}: M_h={MH_MORPH[_k][1]:.2e}  chi2nu={MH_MORPH[_k][2]:.2f}")

# ── Custom legend handler: shaded band below the line ────────────────────────
from matplotlib.legend_handler import HandlerBase
import matplotlib.patches as _mp, matplotlib.lines as _ml

class _BandLineHandle:
    """Dummy handle carrying colour/style info for _HandlerBandLine."""
    def __init__(self, bc, lc, ls="-", lw=1.8, alpha=0.30):
        self.bc = bc; self.lc = lc; self.ls = ls; self.lw = lw; self.alpha = alpha

class _HandlerBandLine(HandlerBase):
    """Draws a horizontal line (top ~70%) with a filled band below (bottom ~42%)."""
    def create_artists(self, legend, h, x0, y0, width, height, fontsize, trans):
        rect = _mp.Rectangle(
            (x0, y0), width, height * 0.42,
            facecolor=h.bc, edgecolor='none', alpha=h.alpha, transform=trans)
        ly = y0 + height * 0.72
        line = _ml.Line2D(
            [x0, x0 + width], [ly, ly],
            color=h.lc, lw=h.lw, ls=h.ls, transform=trans)
        return [rect, line]

# ── Plot ──────────────────────────────────────────────────────────────────────
BIN_COLORS = ["#3498db", "#27ae60", "#e67e22", "#c0392b"]
GBAR_MIN, GBAR_MAX = 3e-16, 3e-11
gbar_c = np.logspace(np.log10(GBAR_MIN), np.log10(GBAR_MAX), 300)
a0_si  = A0_CODE * CODE_TO_SI

fig, axes = plt.subplots(3, 2, figsize=(10, 10.5), sharex=True, sharey=True)
axes = axes.flatten()

print("\nGenerating plot curves ...")
for i, (ax, ms, mh0, bcol, blbl) in enumerate(
        zip(axes, MBAR, MH_MOSTER, BIN_COLORS, BIN_LABELS)):

    bdata = BINS[i]
    r_mpc = bdata[:,0]
    gb    = gbar_si(r_mpc, ms)
    idx   = np.argsort(gb)

    # r_kpc for smooth model curves (from g_bar, Keplerian approx)
    r_kpc_c = np.sqrt(G_SI * ms * MSUN_KG / gbar_c) / KPC_M

    # MOND deep limit — thick very light grey, drawn first (behind everything)
    ax.plot(gbar_c, np.sqrt(gbar_c * a0_si),
            color="#e0e0e0", lw=5.0, ls="-", zorder=0,
            label=rf"MOND deep limit ($\chi^2_\nu={C2_MOND:.2f}$)")

    # Newton 1:1 — solid light grey line edge to edge
    ax.plot([GBAR_MIN, GBAR_MAX], [GBAR_MIN, GBAR_MAX],
            color="#cccccc", lw=1.2, ls="-", zorder=1, label="Newton 1:1")

    # CDM global (solid orange)
    print(f"  bin{i}: CDM global ...", end=" ", flush=True)
    g_cdm_g = gobs_cdm(r_kpc_c, ms, mh0, XI0_BEST)
    ax.plot(gbar_c, g_cdm_g,
            color="#e67e22", lw=2.1, ls="-", zorder=3,
            label=(rf"CDM+2h $\xi_0={XI0_BEST:.1f}$ "
                   rf"($\chi^2_\nu={C2_CDM:.2f}$, 1 param)"))
    print("done")

    # CDM per-bin (dashed orange)
    mh_pb, c2_pb = PB_CDM[i]
    print(f"  bin{i}: CDM per-bin ...", end=" ", flush=True)
    g_cdm_pb = gobs_cdm(r_kpc_c, ms, mh_pb, XI0_BEST)
    ax.plot(gbar_c, g_cdm_pb,
            color="#e67e22", lw=1.5, ls="--", zorder=3,
            label=(rf"CDM per-bin $M_h={mh_pb:.1e}$"
                   rf" ($\chi^2_\nu={c2_pb:.2f}$)"))
    print("done")

    # HMG global (solid cyan, thick, translucent)
    g_hmg_g = gobs_hmg(r_kpc_c, ms, S_BEST)
    ax.plot(gbar_c, g_hmg_g,
            color="#00b4d8", lw=3.2, ls="-", alpha=0.60, zorder=5,
            label=(rf"HMG $s={S_BEST:.3f}$ "
                   rf"($\chi^2_\nu={C2_HMG:.2f}$, 1 param)"))

    # HMG per-bin (dashed cyan, translucent)
    s_pb, c2_hmg_pb = PB_HMG[i]
    g_hmg_pb = gobs_hmg(r_kpc_c, ms, s_pb)
    ax.plot(gbar_c, g_hmg_pb,
            color="#00b4d8", lw=2.0, ls="--", alpha=0.60, zorder=5,
            label=rf"HMG per-bin $s={s_pb:.3f}$ ($\chi^2_\nu={c2_hmg_pb:.2f}$)")

    # MICE isolated — narrow B21-style band (z_true lower, z_photo upper) in yellow
    gb_mice, go_mice, go_p16, go_p84 = MICE_PERBIN[i]
    # light blue: R > R_MIN_VALID_MPC region (B21 Fig. 5: unreliable KiDS isolation)
    ax.axvspan(GBAR_MIN, float(gb_mice.max()),
               color="#aed6f1", alpha=0.12, zorder=0)
    # pure band (no centre line): z_true lower edge, z_photo upper edge
    _mice_fb = ax.fill_between(gb_mice, go_p16, go_p84,
                               color="#f1c40f", alpha=0.14, zorder=1)
    ax.plot(gb_mice, go_p16, color="#f1c40f", lw=0.8, ls="-", alpha=0.55, zorder=1)
    ax.plot(gb_mice, go_p84, color="#f1c40f", lw=0.8, ls="-", alpha=0.55, zorder=1)

    # Data
    ax.fill_between(gb[idx], bdata[idx,2], bdata[idx,3],
                    color=bcol, alpha=0.18, zorder=4)
    ax.errorbar(gb[idx], bdata[idx,1],
                yerr=[bdata[idx,1]-bdata[idx,2], bdata[idx,3]-bdata[idx,1]],
                fmt="o", color=bcol, ms=4.5, lw=1.0, capsize=2, zorder=6,
                label="KiDS-1000")

    # Info box: halo parameters for this bin
    ax.text(0.97, 0.05,
            rf"$M_h={mh0:.1e}\,M_\odot$, $r_{{200}}={_r200(mh0):.0f}$ kpc"
            f"\n" rf"$c={_conc(mh0):.1f}$, $b_g={_b_g(mh0):.2f}$, "
            rf"$\xi_0^*={XI0_BEST:.1f}$",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=7, color="0.30",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.80", alpha=0.85))

    ax.axvline(a0_si, color="0.60", lw=0.8, ls="--", zorder=1)

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.grid(True, which='major', color='#dddddd', lw=0.5, alpha=0.7, zorder=0)
    ax.set_xlim(GBAR_MIN, GBAR_MAX)
    ax.set_ylim(5e-15, 1e-10)
    ax.text(0.04, 0.97, blbl, transform=ax.transAxes,
            ha='left', va='top', fontsize=8,
            bbox=dict(boxstyle='square,pad=0.15', fc='white', ec='none', alpha=0.85))
    _h, _l = ax.get_legend_handles_labels()
    _mice_h = _BandLineHandle("#f1c40f", "#f1c40f", lw=0.8, alpha=0.14)
    ax.legend(handles=_h + [_mice_h],
              labels=_l + [r"MICE $\Lambda$CDM isolated (this work)"],
              handler_map={_BandLineHandle: _HandlerBandLine()},
              fontsize=6.8, loc="lower right", framealpha=0.93,
              handlelength=2.1, labelspacing=0.30)

# ── Bottom row: morphological subsets (= fig 3 content) ───────────────────────
C_LATE  = '#2ecc71';  C_EARLY = '#8e44ad'
C_BLUE  = '#3498db';  C_RED   = '#e91e63'   # fuchsia-red

# HMG curve colours for morphological panels: blue-green / blue-purple
C_LATE_HMG  = '#1abc9c'   # blue-greenish teal
C_EARLY_HMG = '#6c63ff'   # blue-purple

# ── Load MICE morphological data ──────────────────────────────────────────────
_MORPH_DIR = Path(__file__).resolve().parent / "data"

def _load_morph_mice(fname, nbins=18):
    """Load allbins file and bin in log(gbar) to get a smooth RAR curve."""
    d    = np.loadtxt(_MORPH_DIR / fname, comments="#")
    lgb  = d[:, 0]   # log10_gbar
    lgo  = d[:, 1]   # log10_gobs
    lp16 = d[:, 2]
    lp84 = d[:, 3]
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
    bc   = np.array(bc)
    return 10**bc, 10**np.array(go_b), 10**np.array(p16_b), 10**np.array(p84_b)

def _clip_morph(data):
    gb, go, p16, p84 = data
    m = (gb >= GBAR_MORPH_MIN) & (gb <= GBAR_MORPH_MAX)
    return gb[m], go[m], p16[m], p84[m]

_mice_sl = _clip_morph(_load_morph_mice("morph_sersic_late_allbins.txt"))
_mice_se = _clip_morph(_load_morph_mice("morph_sersic_early_allbins.txt"))
_mice_cb = _clip_morph(_load_morph_mice("morph_color_blue_allbins.txt"))
_mice_cr = _clip_morph(_load_morph_mice("morph_color_red_allbins.txt"))

morph_panels = [
    (gb_late,  go_late,  ge_late,  s_late,  c2_late,
     gb_early, go_early, ge_early, s_early, c2_early,
     r'Sérsic index bins',
     r'Late ($n<2.5$)', r'Early ($n\geq2.5$)',
     C_LATE, C_EARLY, '^', 's',
     C_LATE_HMG, C_EARLY_HMG,
     _mice_sl, _mice_se,
     r'MICE disc ($B/T<0.5$)', r'MICE spheroid ($B/T\geq0.5$)',
     "late", "early"),
    (gb_blue,  go_blue,  ge_blue,  s_blue,  c2_blue,
     gb_red,   go_red,   ge_red,   s_red,   c2_red,
     'Color bins', 'Blue (late)', 'Red (early)',
     C_BLUE, C_RED, 'o', 'o',
     C_LATE_HMG, C_EARLY_HMG,
     _mice_cb, _mice_cr,
     r'MICE blue ($g{-}r<0.6$)', r'MICE red ($g{-}r\geq0.6$)',
     "blue", "red"),
]

for ax, (gb_l, go_l, ge_l, sl, c2l,
          gb_e, go_e, ge_e, se, c2e,
          title, lab_l, lab_e, c_l, c_e, mk_l, mk_e,
          c_hmg_l, c_hmg_e,
          mice_l, mice_e, lab_ml, lab_me,
          key_l, key_e) in zip(axes[4:], morph_panels):

    ax.plot(gbar_c, np.sqrt(gbar_c * a0_si),
            color="#e0e0e0", lw=5.0, ls="-", zorder=0, label="MOND deep limit")
    ax.plot([GBAR_MIN, GBAR_MAX], [GBAR_MIN, GBAR_MAX],
            color="#cccccc", lw=1.2, ls="-", zorder=1, label="Newton 1:1")

    # MICE morphological — pure band, same criteria as mass-bin panels
    gb_ml, go_ml, p16_ml, p84_ml = mice_l
    gb_me, go_me, p16_me, p84_me = mice_e
    # light blue: region R > R_MIN (both bands clipped to same gbar range)
    ax.axvspan(GBAR_MIN, float(max(gb_ml.max(), gb_me.max())),
               color="#aed6f1", alpha=0.12, zorder=0)
    _mfb_l = ax.fill_between(gb_ml, p16_ml, p84_ml, color="#f1c40f", alpha=0.14, zorder=1)
    ax.plot(gb_ml, p16_ml, color="#f1c40f", lw=0.8, ls="-",  alpha=0.55, zorder=1)
    ax.plot(gb_ml, p84_ml, color="#f1c40f", lw=0.8, ls="-",  alpha=0.55, zorder=1)
    _mfb_e = ax.fill_between(gb_me, p16_me, p84_me, color="#d4ac0d", alpha=0.14, zorder=1)
    ax.plot(gb_me, p16_me, color="#d4ac0d", lw=0.8, ls="--", alpha=0.55, zorder=1)
    ax.plot(gb_me, p84_me, color="#d4ac0d", lw=0.8, ls="--", alpha=0.55, zorder=1)

    ax.errorbar(gb_l, go_l, yerr=ge_l, fmt=mk_l, color=c_l, ms=5,
                label=lab_l, capsize=3, alpha=0.9, zorder=5)
    ax.errorbar(gb_e, go_e, yerr=ge_e, fmt=mk_e, color=c_e, ms=5,
                label=lab_e, capsize=3, alpha=0.9, zorder=5)

    ax.plot(gbar_c, hmg_pred_full(gbar_c, sl, _r_of[key_l](gbar_c)), "-",
            color=c_hmg_l, lw=2.0, zorder=4,
            label=rf"HMG $s={sl:.3f}$ ($\chi^2_\nu={c2l:.2f}$)")
    ax.plot(gbar_c, hmg_pred_full(gbar_c, se, _r_of[key_e](gbar_c)), "-",
            color=c_hmg_e, lw=2.0, zorder=4,
            label=rf"HMG $s={se:.3f}$ ($\chi^2_\nu={c2e:.2f}$)")

    # CDM (orange): solid for the left subset (late/blue), dashed for the right (early/red).
    # Clipped to each subset's data g_bar range (the MICE r(g_bar) map is only valid there;
    # beyond it r would be extrapolated, flattening the curve artificially).
    _gl = gbar_c[(gbar_c >= gb_l.min()) & (gbar_c <= gb_l.max())]
    _ge = gbar_c[(gbar_c >= gb_e.min()) & (gbar_c <= gb_e.max())]
    _ms_l, _mh_l, _c2cl = MH_MORPH[key_l]
    ax.plot(_gl, gobs_cdm(_r_of[key_l](_gl), _ms_l, _mh_l, XI0_BEST), "-",
            color="#e67e22", lw=1.8, zorder=3,
            label=rf"CDM {lab_l.split()[0]} ($\chi^2_\nu={_c2cl:.2f}$)")
    _ms_e, _mh_e, _c2ce = MH_MORPH[key_e]
    ax.plot(_ge, gobs_cdm(_r_of[key_e](_ge), _ms_e, _mh_e, XI0_BEST), "--",
            color="#e67e22", lw=1.8, zorder=3,
            label=rf"CDM {lab_e.split()[0]} ($\chi^2_\nu={_c2ce:.2f}$)")

    ax.axvline(a0_si, color="0.60", lw=0.8, ls="--", zorder=1)

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.grid(True, which='major', color='#dddddd', lw=0.5, alpha=0.7, zorder=0)
    ax.set_xlim(GBAR_MIN, GBAR_MAX)
    ax.set_ylim(5e-15, 1e-10)
    ax.text(0.04, 0.97, title, transform=ax.transAxes,
            ha='left', va='top', fontsize=8,
            bbox=dict(boxstyle='square,pad=0.15', fc='white', ec='none', alpha=0.85))
    _h, _l = ax.get_legend_handles_labels()
    _mh_l = _BandLineHandle("#f1c40f", "#f1c40f", ls="-",  lw=0.8, alpha=0.14)
    _mh_e = _BandLineHandle("#d4ac0d", "#d4ac0d", ls="--", lw=0.8, alpha=0.14)
    ax.legend(handles=_h + [_mh_l, _mh_e],
              labels=_l + [lab_ml, lab_me],
              handler_map={_BandLineHandle: _HandlerBandLine()},
              fontsize=6.8, loc="lower right", framealpha=0.93,
              handlelength=2.1, labelspacing=0.30)

XLABEL = r"$g_\mathrm{bar}=GM_\mathrm{bar}/r^2\;[\mathrm{m\,s^{-2}}]$"
YLABEL = r"$g_\mathrm{obs}\;[\mathrm{m\,s^{-2}}]$"

# Bottom xlabel — last row
for ax in axes[4:]:
    ax.set_xlabel(XLABEL, fontsize=10)

# Top xlabel — first row (mirrored)
for ax in axes[:2]:
    ax.tick_params(axis='x', which='both', top=True, labeltop=True)
    ax.xaxis.set_label_position('top')
    ax.set_xlabel(XLABEL, fontsize=10)

# Left ylabel — left column
for ax in axes[::2]:
    ax.set_ylabel(YLABEL, fontsize=10)

# Right ylabel — right column (mirrored)
for ax in axes[1::2]:
    ax.tick_params(axis='y', which='both', right=True, labelright=True)
    ax.yaxis.set_label_position('right')
    ax.set_ylabel(YLABEL, fontsize=10)

plt.tight_layout(h_pad=0.25, w_pad=0.25)

out = Path(__file__).resolve().parent / "outputs" / "fig1_kids_rar.png"
out.parent.mkdir(exist_ok=True)
plt.savefig(out, dpi=200, bbox_inches="tight")
plt.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
print(f"\nSaved: {out}")
