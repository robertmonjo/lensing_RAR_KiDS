"""
hmg_core.py -- shared backbone for the reproducible lensing-RAR pipeline.

Physical constants, HMG / MOND / CDM (truncated-NFW + two-halo) model functions,
the KiDS-1000 data (Brouwer+2021), and self-contained loaders that read only
from ./data (no absolute paths, no external directories).

HMG references: Monjo (2025, ApJ 982, 70); Monjo & Banik (2025, PASA).
"""
from pathlib import Path
import math
import numpy as np
from scipy.optimize import brentq

DATA_DIR = Path(__file__).resolve().parent / "data"

# -- Physical constants (code units: km/s, kpc, Msun) --------------------------
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
RHO_M_MPC  = OMEGA_M * (RHO_CRIT * (MPC_TO_KPC)**3)           # Msun Mpc^-3
SIN2_GU    = math.sin(math.pi/3)**2
SIN2_GCEN  = 1.0
A0_SI      = A0_CODE * CODE_TO_SI
_G_PC      = 4.52e-30
_PC_M      = 3.086e16
ESD2G      = 4.0 * _G_PC * _PC_M

# MICE validity range (Brouwer+2021 Sect. 5.3): inner = resolution, outer = GGL noise
R_MIN_VALID_MPC = 0.30
R_MAX_VALID_MPC = 1.50
GBAR_MORPH_MAX  = 6.2e-14
GBAR_MORPH_MIN  = 2.5e-15

# ==============================================================================
# KiDS-1000 data (Brouwer+2021, arXiv:2106.11677)
#   BINS[i] rows: r_Mpc, g_obs, g_obs_p16, g_obs_p84   [SI, m/s^2]
# ==============================================================================
MSTAR = [1.5e10, 3.2e10, 4.6e10, 8.9e10]

def f_cold(mstar):
    """Cold-gas fraction (Boselli 2014): log10 f_cold = -0.69 log10(M*/Msun) + 6.63.
    Gives ~40% (bin 1) down to ~12% (bin 4)."""
    return 10.0 ** (-0.69 * np.log10(mstar) + 6.63)

# Total baryonic point mass per bin (stars + cold gas): the mass that sets g_bar at
# the >30 kpc lensing radii (M_gal = M_star (1+f_cold), Brouwer+2021).
MBAR = [m * (1.0 + f_cold(m)) for m in MSTAR]

BIN_LABELS = [
    r"$M_\star=1.5\times10^{10}\,M_\odot$",
    r"$M_\star=3.2\times10^{10}\,M_\odot$",
    r"$M_\star=4.6\times10^{10}\,M_\odot$",
    r"$M_\star=8.9\times10^{10}\,M_\odot$",
]

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

# -- MICE NFW P16-P84 fallback band (hardcoded from a Nieve run) --
#   rows: log10_gbar, log10_gobs_mean, log10_gobs_p16, log10_gobs_p84
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
    _a = np.asarray(_d, float)
    _MICE_NIEVE_PERBIN.append((10.0**_a[:, 0], 10.0**_a[:, 1], 10.0**_a[:, 2], 10.0**_a[:, 3]))

# ==============================================================================
# Model functions
# ==============================================================================
def mond_gobs(gbar):
    """MOND simple interpolation function, a0 fixed (no free params)."""
    a0 = A0_CODE * CODE_TO_SI
    return gbar / (1.0 - np.exp(-np.sqrt(gbar / a0)))

def _cos_over_gamma(xi2_s):
    xi2   = np.asarray(xi2_s, float)
    delta = np.abs(1.0 - xi2) / (1.0 + xi2 + 1e-30)
    s2    = SIN2_GU + (SIN2_GCEN - SIN2_GU) * np.clip(delta, 0.0, 1.0)
    gs    = np.arcsin(np.sqrt(np.clip(s2, 0.0, 1.0)))
    return np.cos(gs) / np.maximum(gs, 1e-10)

def gobs_hmg(r_kpc, ms, s):
    """Full HMG: xi_s^2 = 1/s^3 + v_H^2/(12 v_N^2)."""
    r   = np.asarray(r_kpc, float)
    v2n = G_CODE * ms / r
    vh2 = (r / T0)**2
    xi2 = 1.0/s**3 + vh2/(12.0*v2n + 1e-60)
    a_n = v2n / r
    return np.sqrt(np.maximum(a_n*(a_n + 2.0*C_KMS/T0*_cos_over_gamma(xi2)), 0.0)) * CODE_TO_SI

def _q_dl(x):
    delta = abs(x**2 - 1) / (x**2 + 1)
    s2 = SIN2_GU + (1.0 - SIN2_GU) * delta
    gam = math.asin(math.sqrt(max(min(s2, 1.0), 0.0)))
    return math.cos(gam) / max(gam, 1e-12)

def g_s_dl(s):
    """Morphological deep limit: xi_s^2 = 1/s^3 -> q argument x = 1/xi_s = s^{3/2}.
    delta(x)=delta(1/x), so _q_dl(s**1.5) equals q(1/s^3)."""
    return 2.0 * C_KMS / T0 * _q_dl(s**1.5) * CODE_TO_SI

def hmg_pred_dl(gbar, s):
    return np.sqrt(gbar * (gbar + g_s_dl(s)))

def const_g0_pred(gbar, g0):
    """Reduced 1-param submodel g_obs = sqrt(gbar (gbar + g0))."""
    return np.sqrt(gbar * (gbar + g0))

# -- CDM: Moster+2013 SHMR, Dutton&Maccio conc., Tinker+2010 bias, NFW + 2-halo -
_M1, _N, _BETA, _GAM = 10**11.590, 0.0351, 1.376, 0.608
def _ms_from_mh(mh):
    x = mh / _M1
    return mh * 2*_N / (x**(-_BETA) + x**_GAM)
def mhalo_from_mstar(ms):
    f = lambda lmh: math.log10(_ms_from_mh(10**lmh)) - math.log10(ms)
    return 10**brentq(f, 10.0, 15.0)
def _r200(mh):
    return (mh / (4*math.pi/3 * 200 * RHO_CRIT))**(1/3)   # kpc
def _conc(mh):
    return 10**(0.905 - 0.101*math.log10(mh / (1e12/0.7)))
def nfw_enclosed(r_kpc, mh):
    r200 = _r200(mh); c = _conc(mh); rs = r200/c
    x    = np.asarray(r_kpc, float) / rs
    fc   = math.log(1+c) - c/(1+c)
    return mh * (np.log(1+x) - x/(1+x)) / fc
def _b_g(mh):
    lm = math.log10(mh)
    if   lm < 11:  return 0.65
    elif lm < 12:  return 0.65 + 0.35*(lm - 11)
    elif lm < 13:  return 1.00 + 0.70*(lm - 12)
    else:          return 1.70
def gobs_cdm(r_kpc_arr, ms, mh, xi0):
    r_kpc  = np.asarray(r_kpc_arr, float)
    r200   = _r200(mh); r200_m = r200 / MPC_TO_KPC
    m_1h   = np.minimum(nfw_enclosed(r_kpc, mh), mh)
    r_mpc  = r_kpc / MPC_TO_KPC; bg = _b_g(mh)
    m_2h   = np.zeros_like(r_kpc); mask = r_kpc > r200
    if np.any(mask):
        rm = r_mpc[mask]
        m_2h[mask] = (4*math.pi * bg * RHO_M_MPC * xi0 / 1.2 * (rm**1.2 - r200_m**1.2))
    m_tot = ms + m_1h + m_2h
    return G_SI * m_tot * MSUN_KG / (r_kpc * KPC_M)**2

def gbar_si(r_mpc, ms):
    return G_SI * ms * MSUN_KG / (r_mpc * MPC_M)**2

# -- chi2 helpers (mass-bin panels) --------------------------------------------
def _logsig(b):     return (np.log10(b[:, 3]) - np.log10(b[:, 2])) / 2.0
def _chi2_arr(pred, bdata):
    sig = _logsig(bdata)
    return float(np.sum(((np.log10(bdata[:, 1]) - np.log10(pred)) / sig)**2))
def _r_kpc(bdata):  return bdata[:, 0] * MPC_TO_KPC

def chi2_dl(s, gbar, gobs, gerr):
    """Morphological (deep-limit) reduced chi2 in log-acceleration space."""
    sig = gerr / (gobs * np.log(10))
    return float(np.sum(((np.log10(gobs) - np.log10(hmg_pred_dl(gbar, s))) / sig)**2)) / (len(gbar) - 1)

# ==============================================================================
# Self-contained data loaders (read only from ./data)
# ==============================================================================
def load_morph(fname):
    """KiDS morphological ESD -> (gbar, gobs, gerr) in SI (Brouwer+2021 Fig-8)."""
    d    = np.loadtxt(DATA_DIR / fname, comments='#')
    gbar = d[:, 0]; bias = d[:, 4]
    gobs = ESD2G * d[:, 1] / bias
    gerr = ESD2G * d[:, 3] / bias
    ok   = gobs > 0
    return gbar[ok], gobs[ok], gerr[ok]

def load_mice_b21():
    """MICE B21-style band per bin -> list of (gb, go_lo, go_lo, go_hi) in SI,
    clipped to [R_MIN_VALID, R_MAX_VALID]. Returns None if any file missing."""
    result = []
    for b in range(1, 5):
        fname = DATA_DIR / ("rar_band_b21_bin%d.txt" % b)
        if not fname.exists():
            return None
        d = np.loadtxt(fname, comments="#")
        d = d[(d[:, 0] >= R_MIN_VALID_MPC) & (d[:, 0] <= R_MAX_VALID_MPC)]
        if len(d) == 0:
            return None
        gb = 10.0**d[:, 1]; go_lo = 10.0**d[:, 2]; go_hi = 10.0**d[:, 3]
        result.append((gb, go_lo, go_lo, go_hi))
    return result

def load_morph_mice(fname, nbins=18):
    """MICE morphological all-bins file, binned in log(gbar) -> smooth RAR band."""
    d = np.loadtxt(DATA_DIR / fname, comments="#")
    lgb, lgo, lp16, lp84 = d[:, 0], d[:, 1], d[:, 2], d[:, 3]
    edges = np.linspace(lgb.min(), lgb.max(), nbins + 1)
    bc, go_b, p16_b, p84_b = [], [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (lgb >= lo) & (lgb < hi)
        if m.sum() == 0: continue
        bc.append(0.5*(lo+hi)); go_b.append(np.median(lgo[m]))
        p16_b.append(np.median(lp16[m])); p84_b.append(np.median(lp84[m]))
    return (10**np.array(bc), 10**np.array(go_b), 10**np.array(p16_b), 10**np.array(p84_b))

def clip_morph_mice(data):
    gb, go, p16, p84 = data
    m = (gb >= GBAR_MORPH_MIN) & (gb <= GBAR_MORPH_MAX)
    return gb[m], go[m], p16[m], p84[m]
