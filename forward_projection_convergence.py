"""Numerical convergence of the forward Abel projection (Appendix A.4).

Usage: python forward_projection_convergence.py
Compares the settings of compare_fwd_models.py with all settings refined at once
(line-of-sight integral split at 10R and 1 Mpc), for HMG, MOND and CDM in the four
mass bins, and counts the grid points where rho_eff <= 0.
Output: outputs/forward_projection_convergence.json
"""
import sys, warnings, json, time
import os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
from scipy.integrate import quad, IntegrationWarning
from scipy.interpolate import interp1d
warnings.filterwarnings('ignore', category=IntegrationWarning)
import hmg_model as h

G_SI, MSUN_KG, MPC_M = h.G_SI, h.MSUN_KG, h.MPC_M
PC_M = 3.0857e16
ESD2G = 4.0 * G_SI * MSUN_KG / PC_M**2
S_BIN = [0.2724, 0.4957, 0.5285, 0.4268]
XI0 = 14.8
MH = [h.mhalo_from_mstar(ms) for ms in h.MSTAR]

BASE = dict(r_max=300.0, n_r=6000, z_max=200.0, n_Rg=100, n_int=1000, R_lo=0.001, epsrel=1e-5)
FINE = dict(r_max=600.0, n_r=12000, z_max=400.0, n_Rg=200, n_int=4000, R_lo=0.0005, epsrel=1e-7, split=True)

def g_fwd(R_arr, gfun, mbar, p):
    r = np.geomspace(3e-4, p['r_max'], p['n_r'])
    g3 = np.array([gfun(x) for x in r])
    M = (r * MPC_M)**2 * g3 / G_SI / MSUN_KG
    rho_raw = np.gradient(M, r) / (4 * np.pi * r**2)
    n_clip = int(np.sum(rho_raw[r <= p['z_max']] <= 0))
    rho = np.maximum(rho_raw, 0.0); good = rho > 0
    spl = interp1d(np.log(r[good]), np.log(rho[good]), kind='cubic', bounds_error=False,
                   fill_value=(np.log(rho[good][0]), np.log(rho[good][-1])))
    rho_at = lambda x: float(np.exp(spl(np.log(x))))
    Rg = np.logspace(np.log10(p['R_lo']), np.log10(R_arr.max() * 4), p['n_Rg'])
    def Sig(R):
        f = lambda z: rho_at(np.sqrt(R**2 + z**2))
        if not p.get('split'):
            return 2 * quad(f, 0, p['z_max'], limit=500, epsrel=p['epsrel'])[0]
        br = sorted({0.0, min(10 * R, 1.0), 1.0, p['z_max']})
        return 2 * sum(quad(f, a, c, limit=500, epsrel=p['epsrel'])[0] for a, c in zip(br[:-1], br[1:]))
    Sg = np.array([Sig(R) for R in Rg]) / 1e12
    Ss = interp1d(np.log(Rg), np.log(np.maximum(Sg, 1e-40)), kind='cubic',
                  bounds_error=False, fill_value='extrapolate')
    sig = lambda R: np.exp(Ss(np.log(R)))
    dS = np.array([2 / R**2 * np.trapezoid(Ri * sig(Ri), Ri) - sig(R)
                   for R in R_arr for Ri in [np.linspace(p['R_lo'], R, p['n_int'])]])
    return ESD2G * dS + ESD2G * mbar / (np.pi * R_arr**2 * 1e12), n_clip

out = []
for i, (b, mb) in enumerate(zip(h.BINS, h.MBAR)):
    R = b[:, 0]
    models = {'HMG': lambda r, mb=mb, s=S_BIN[i]: float(h.gobs_hmg(np.array([r * 1e3]), mb, s)[0]),
              'MOND': lambda r, mb=mb: float(h.mond_gobs(h.gbar_si(r, mb))),
              'CDM': lambda r, mb=mb, mh=MH[i]: float(h.gobs_cdm(np.array([r * 1e3]), mb, mh, XI0)[0])}
    for name, f in models.items():
        t = time.time()
        g0, c0 = g_fwd(R, f, mb, BASE)
        g1, c1 = g_fwd(R, f, mb, FINE)
        rel = np.abs(g1 / g0 - 1)
        mid = (R >= 0.16) & (R <= 0.8)
        x0 = h._chi2_arr(g0, b) / 14; x1 = h._chi2_arr(g1, b) / 14
        row = dict(bin=i + 1, model=name, max_rel_all=float(rel.max()), max_rel_mid=float(rel[mid].max()),
                   chi2nu_base=x0, chi2nu_fine=x1, clip_base=c0, clip_fine=c1, sec=round(time.time() - t, 1))
        out.append(row); print(row, flush=True)
json.dump(out, open(os.path.join(HERE, 'outputs', 'forward_projection_convergence.json'), 'w'), indent=1)
print('MAX rel all', max(r['max_rel_all'] for r in out), 'mid', max(r['max_rel_mid'] for r in out),
      'max |dchi2nu|', max(abs(r['chi2nu_fine'] - r['chi2nu_base']) for r in out),
      'clipped', sum(r['clip_base'] + r['clip_fine'] for r in out))
