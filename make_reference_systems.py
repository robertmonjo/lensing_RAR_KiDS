"""
Reproduce the external reference systems shown in Fig. A.2(a) / Table A.1.

For the multi-object samples we RE-FIT the HMG neighbourhood scale s of every member
with the SAME expression used for the KiDS subsamples,
    xi_s^2 = 1/s^3 + v_H^2/(12 v_N^2),   g_obs = sqrt(g_N (g_N + g_s)),  g_s = (2c/t0) q(xi_s^2),
so the reported s are directly comparable.  Error bars are the SAMPLE DISPERSION
(spread between members), not a fit uncertainty:
  * SPARC   : 50 of the 61 McGaugh (2007) galaxies (those with >=4 valid rotation-curve
              points; the 11 with fewer are dropped) -> s and r_eq per galaxy.
  * HIFLUGCS: 10 clusters (Li 2023, via Monjo & Banik 2025) -> s and r_eq per cluster.
  * X-COP   : 5 complete clusters (Eckert 2022, via Monjo 2025b) -> r_eq from M500,bar.

  * gal-gal WL: Mistele (2024) stacked circular velocities (4 mass bins x 5 radii).
  * Milky Way : published radial + vertical two-branch fit.

For EVERY system both s<->1/s solution branches are FITTED independently (never s_mirror = 1/s)
and a `degenerate` flag records whether the data can statistically separate them -- a
likelihood-ratio (Wilks) test on the TOTAL Delta chi2 between branches (see the DELTA_DEG block
below).  make_figA2_landscape.py fills both stars when degenerate, else opens the disfavoured one.

Data live in data/external_reference/ (each file carries a provenance header citing its
published source).
Output: outputs/reference_systems.csv (+ per-object CSVs), read by make_figA2_landscape.py.
"""
import os
import csv
import math
import numpy as np

G_CODE     = 4.30091727e-6      # (km/s)^2 kpc Msun^-1
T0         = 14.11              # kpc (km/s)^-1
C_KMS      = 299792.458         # km/s
CODE_TO_SI = 3.24078e-14        # (km/s)^2 kpc^-1 -> m s^-2
HERE       = os.path.dirname(os.path.abspath(__file__))
DATA       = os.path.join(HERE, "data", "external_reference")
OUT        = os.path.join(HERE, "outputs")

def _q(x):
    x = np.maximum(np.asarray(x, float), 1e-12)
    d = np.abs(x*x - 1) / (x*x + 1)
    s2 = math.sin(math.pi/3)**2 + (1 - math.sin(math.pi/3)**2) * d
    g = np.arcsin(np.sqrt(np.clip(s2, 0, 1)))
    return np.cos(g) / np.maximum(g, 1e-12)

def r_newton_kpc(M):
    return (G_CODE * M * T0**2) ** (1.0/3.0)

def _fit_s(r_kpc, gbar_code, gobs_code, err=None, s_range=(0.2, 12.0)):
    """Fit single s (code units); returns argmin over a grid within s_range."""
    sg = np.linspace(s_range[0], s_range[1], 3000)
    def pred(s):
        hub = r_kpc / (12.0 * gbar_code * T0**2)      # v_H^2/(12 v_N^2)
        gs = 2.0 * C_KMS / T0 * _q(np.sqrt(1.0/s**3 + hub))
        return np.sqrt(gbar_code * (gbar_code + gs))
    w = 1.0 if err is None else 1.0/np.asarray(err)**2
    chi = np.array([np.sum(w*(np.log10(gobs_code) - np.log10(pred(s)))**2) for s in sg])
    i = int(np.argmin(chi))
    return float(sg[i]), float(chi[i])

# ── SPARC: fit s and r_eq per galaxy (McGaugh 2007 rotation curves) ─────────────
gal = {}
for ln in open(os.path.join(DATA, "McGaugh2007.txt")):
    p = ln.split()
    if len(p) < 7 or p[0] == "Name":
        continue
    try:
        R, Vobs, Vgas, Vst, Qmax, Qpop = map(float, p[1:7])
    except ValueError:
        continue
    if R > 0 and Vobs > 0:
        gal.setdefault(p[0], []).append((R, Vobs, Vgas, Vst, Qpop))

# Fit BOTH branches per galaxy (the deep-limit s <-> 1/s degeneracy is unbroken here:
# these systems are system-dominated, so q(xi)=q(1/xi) gives two near-equal-chi2 solutions).
sparc_hi, sparc_lo, sparc_req = [], [], []   # s>1 branch, s<1 branch, r_eq (Mpc)
sparc_chi_hi, sparc_chi_lo = [], []          # chi2 of each branch (to pick the filled one)
_sp_wh = _sp_wl = 0.0                         # weighted branch chi2 sums (for the degeneracy test)
for nm, pts in gal.items():
    R  = np.array([q[0] for q in pts]); Vo = np.array([q[1] for q in pts])
    Vg = np.array([q[2] for q in pts]); Vs = np.array([q[3] for q in pts]); Qp = np.array([q[4] for q in pts])
    Vbar2 = Vg**2 + Qp*Vs**2                       # baryonic velocity^2 (gas + M/L*stars)
    m = (Vbar2 > 0) & (Vo > 0)
    if m.sum() < 4:            # need >=4 valid points for a single-s fit -> drops 11/61
        continue
    gbar = Vbar2[m] / R[m]; gobs = Vo[m]**2 / R[m]  # (km/s)^2/kpc
    s_hi, c_hi = _fit_s(R[m], gbar, gobs, s_range=(1.0, 12.0))    # Hubble branch  s>1
    s_lo, c_lo = _fit_s(R[m], gbar, gobs, s_range=(0.15, 1.0))    # mirror branch  s<1
    im = int(np.argmax(R[m])); Mbar = gbar[im] * R[m][im]**2 / G_CODE
    sparc_hi.append(s_hi); sparc_lo.append(s_lo); sparc_req.append(r_newton_kpc(Mbar)/1000.0)
    sparc_chi_hi.append(c_hi); sparc_chi_lo.append(c_lo)
    _sv = np.sqrt((0.043*Vo[m])**2 + 4.6**2); _sl = 2*_sv/(Vo[m]*np.log(10.0))   # Lelli+2016 proxy
    _, _wh = _fit_s(R[m], gbar, gobs, err=_sl, s_range=(1.0, 12.0))
    _, _wl = _fit_s(R[m], gbar, gobs, err=_sl, s_range=(0.15, 1.0))
    _sp_wh += _wh; _sp_wl += _wl
sparc_hi = np.array(sparc_hi); sparc_lo = np.array(sparc_lo); sparc_req = np.array(sparc_req)
sparc_chi_hi = np.array(sparc_chi_hi); sparc_chi_lo = np.array(sparc_chi_lo)
# Sample size: galaxies with valid data vs those actually fitted (>=4 points).
print(f"  [SPARC] {len(sparc_req)} of {len(gal)} McGaugh (2007) galaxies used (>=4 valid points)")
# per-galaxy filled branch = lower-chi2 solution; open branch = the mirror (small stars in figure)
sparc_smin  = np.where(sparc_chi_hi <= sparc_chi_lo, sparc_hi, sparc_lo)
sparc_sopen = np.where(sparc_chi_hi <= sparc_chi_lo, sparc_lo, sparc_hi)
with open(os.path.join(OUT, "sparc_per_galaxy.csv"), "w", newline="") as fh:
    ww = csv.writer(fh); ww.writerow(["r_eq", "s_fill", "s_open"])
    for i in range(len(sparc_req)):
        ww.writerow([round(sparc_req[i], 4), round(float(sparc_smin[i]), 3),
                     round(float(sparc_sopen[i]), 3)])
_SPARC_HI_WINS = sparc_chi_hi.sum() <= sparc_chi_lo.sum()   # aggregate filled branch (total chi2)

# ── HIFLUGCS: fit s and r_eq per cluster (stacked RAR per cluster) ──────────────
cl = {}
for ln in open(os.path.join(DATA, "clusterRAR.dat")):
    if ln.startswith("#"):
        continue
    p = ln.split()
    cl.setdefault(p[0], []).append((float(p[2]), 10**float(p[3]), 10**float(p[4]),
                                    (float(p[5]) + float(p[6]))/2))
hif_sf, hif_so, hif_req, hif_chi = [], [], [], []   # s<1 (filled), s>1 mirror (fitted), r_eq, chi2
_hf_clo = _hf_chi = 0.0
for nm, pts in cl.items():
    r = np.array([q[0] for q in pts]); gb = np.array([q[1] for q in pts])
    gt = np.array([q[2] for q in pts]); er = np.array([q[3] for q in pts])
    gbc = gb / CODE_TO_SI; gtc = gt / CODE_TO_SI
    s_lo, c_lo = _fit_s(r, gbc, gtc, err=er, s_range=(0.2, 1.0))     # neighbourhood branch s<1
    s_hi, c_hi = _fit_s(r, gbc, gtc, err=er, s_range=(1.0, 12.0))    # mirror s>1 (FITTED, not 1/s)
    im = int(np.argmax(r)); Mbar = gbc[im] * r[im]**2 / G_CODE
    hif_sf.append(s_lo); hif_so.append(s_hi); hif_req.append(r_newton_kpc(Mbar)/1000.0)
    hif_chi.append(min(c_lo, c_hi)/len(pts)); _hf_clo += c_lo; _hf_chi += c_hi
hif_sf = np.array(hif_sf); hif_so = np.array(hif_so); hif_req = np.array(hif_req); hif_chi = np.array(hif_chi)
good = hif_chi < 2.5                               # well-fit clusters for r_eq range
with open(os.path.join(OUT, "hiflugcs_per_cluster.csv"), "w", newline="") as fh:
    ww = csv.writer(fh); ww.writerow(["r_eq", "s_fill", "s_open"])
    for i in range(len(hif_req)):                  # both branches fitted per cluster
        ww.writerow([round(float(hif_req[i]), 4), round(float(hif_sf[i]), 3), round(float(hif_so[i]), 3)])

# ── X-COP: fit s and r_eq per cluster (Eckert 2022 hydrostatic RAR, R>=1000 kpc) ──
xc = {}
for ln in open(os.path.join(DATA, "xcop_rar_perpoint.txt")):
    if ln.startswith("#") or not ln.strip():
        continue
    p = ln.split()
    xc.setdefault(p[0], []).append((float(p[1]), float(p[2]), float(p[3]), float(p[4])))
xcop_sf, xcop_so, xcop_req = [], [], []; _xc_clo = _xc_chi = 0.0
for nm, pts in xc.items():
    r = np.array([q[0] for q in pts]); gb = np.array([q[1] for q in pts])
    gt = np.array([q[2] for q in pts]); er = np.array([q[3] for q in pts])
    gbc = gb / CODE_TO_SI; gtc = gt / CODE_TO_SI
    slog = er / (gt * np.log(10.0))                    # log-space 1sigma (matches Table A.1)
    s_lo, c_lo = _fit_s(r, gbc, gtc, err=slog, s_range=(0.2, 1.0))  # neighbourhood branch s<1
    s_hi, c_hi = _fit_s(r, gbc, gtc, err=slog, s_range=(1.0, 12.0)) # mirror s>1 (FITTED, not 1/s)
    im = int(np.argmax(r)); Mbar = gbc[im] * r[im]**2 / G_CODE
    xcop_sf.append(s_lo); xcop_so.append(s_hi); xcop_req.append(r_newton_kpc(Mbar)/1000.0)
    _xc_clo += c_lo; _xc_chi += c_hi
xcop_sf = np.array(xcop_sf); xcop_so = np.array(xcop_so); xcop_req = np.array(xcop_req)
with open(os.path.join(OUT, "xcop_per_cluster.csv"), "w", newline="") as fh:
    ww = csv.writer(fh); ww.writerow(["r_eq", "s_fill", "s_open"])
    for i in range(len(xcop_req)):                     # both branches fitted per cluster
        ww.writerow([round(float(xcop_req[i]), 4), round(float(xcop_sf[i]), 3), round(float(xcop_so[i]), 3)])

def pct(a):
    return float(np.median(a)), float(np.percentile(a, 16)), float(np.percentile(a, 84))

# ── s <-> 1/s branch degeneracy: which stars are filled in Fig. A.2 ───────────────
# HMG admits two solution branches per system (a neighbourhood branch s<1 and a Hubble branch
# s>1).  BOTH are FITTED explicitly here (never assumed s_mirror = 1/s): the s<->1/s symmetry is
# exact only in the deep limit v_H^2/(12 v_N^2) << 1/s^3, and the Hubble term shifts the mirror
# away from 1/s (e.g. gal-gal WL: s=2.54 but mirror 0.36, not 1/2.54=0.39).
#
# WHICH BRANCH IS DRAWN OPEN (disfavoured) vs FILLED?  This is a likelihood-ratio test, NOT a
# v_H/v_N rule.  The two branches are two values of the SAME 1-parameter model, so fixing s to
# the disfavoured branch vs. the free best fit is a nested hypothesis -> Wilks' theorem:
#     Delta chi2 = chi2(disfavoured branch) - chi2(best branch)  ~  chi2 with 1 d.o.f.,
# and the disfavoured branch is rejected at sqrt(Delta chi2) sigma  (p = P(chi2_1 > Delta chi2)).
# We call the two branches DEGENERATE (both stars filled) when they cannot be separated at 2 sigma,
# i.e. Delta chi2 < 4  (a convention: Delta chi2 = 1/4/9 -> 1/2/3 sigma).  Delta chi2 uses the
# TOTAL (summed, unreduced) chi2 -- the proper statistic -- NOT the reduced chi2_nu.
#
# The two are easily confused: the per-point REDUCED chi2_nu(s) can look almost symmetric about
# s=1 (what Fig. A.2b shows) while the branches are still highly distinguishable, because a tiny
# per-point asymmetry summed over many precise points gives a large TOTAL Delta chi2.  Example --
# X-COP: reduced chi2_nu median 7.45 (s<1) vs 7.69 (s>1), only 3% apart (looks symmetric), yet
# summed over 330 points Delta chi2 = 103 -> 10 sigma (clearly distinguishable).
#
# Measured here (Delta chi2 -> significance):  Milky Way 0.0 (0 sigma) and gal-gal WL 0.67
# (0.8 sigma) are DEGENERATE (both filled); SPARC 9.2 (3.0 sigma), HIFLUGCS 84 (9 sigma) and
# X-COP 103 (10 sigma) are DISTINGUISHABLE (filled + open).  SPARC is the only marginal case.
DELTA_DEG = 4.0                                    # 2-sigma threshold on Delta chi2 (see above)

def _row(name, r_eq, r_eq_lo, r_eq_hi, s, s_lo, s_hi, N,
         s_alt="", s_alt_lo="", s_alt_hi="", degenerate=False):
    return dict(name=name, r_eq=round(r_eq, 3), r_eq_lo=round(r_eq_lo, 3), r_eq_hi=round(r_eq_hi, 3),
                s=round(s, 2), s_lo=round(s_lo, 2), s_hi=round(s_hi, 2), N=N,
                s_alt=(round(s_alt, 2) if s_alt != "" else ""),
                s_alt_lo=(round(s_alt_lo, 2) if s_alt_lo != "" else ""),
                s_alt_hi=(round(s_alt_hi, 2) if s_alt_hi != "" else ""),
                degenerate=int(bool(degenerate)))

rows = []
# SPARC: both degenerate branches. Filled (s,s_lo,s_hi) = lower total-chi2 branch; open (s_alt) = mirror.
_hm, _hl, _hh = pct(sparc_hi); _lm, _ll, _lh = pct(sparc_lo); _rm, _rl, _rh = pct(sparc_req)
_sp_deg = abs(_sp_wh - _sp_wl) < DELTA_DEG
if _SPARC_HI_WINS:
    rows.append(_row("SPARC", _rm, _rl, _rh, _hm, _hl, _hh, len(sparc_req), _lm, _ll, _lh, _sp_deg))
else:
    rows.append(_row("SPARC", _rm, _rl, _rh, _lm, _ll, _lh, len(sparc_req), _hm, _hl, _hh, _sp_deg))
print(f"  [SPARC] s>1 median={_hm:.2f} [{_hl:.2f},{_hh:.2f}] chi2tot={sparc_chi_hi.sum():.3f}  "
      f"s<1 median={_lm:.2f} [{_ll:.2f},{_lh:.2f}] chi2tot={sparc_chi_lo.sum():.3f}  "
      f"filled={'s>1' if _SPARC_HI_WINS else 's<1'}  (per-gal: {int((sparc_chi_hi<=sparc_chi_lo).sum())} hi / "
      f"{int((sparc_chi_lo<sparc_chi_hi).sum())} lo)")

_hfm, _hfl, _hfh = pct(hif_sf); _hom, _hol, _hoh = pct(hif_so)
print(f"  [HIFLUGCS] s<1 median={_hfm:.2f} [{_hfl:.2f},{_hfh:.2f}]  s>1 mirror median={_hom:.2f} "
      f"[{_hol:.2f},{_hoh:.2f}]  good-cluster r_eq=[{hif_req[good].min():.2f},{hif_req[good].max():.2f}] Mpc")
# Adopted r_eq = 4.0 (-0.5,+0.8) Mpc (well-fit subset); s from the per-cluster branch medians.
rows.append(_row("HIFLUGCS", 4.0, 3.5, 4.8, _hfm, _hfl, _hfh, len(hif_sf), _hom, _hol, _hoh,
                 degenerate=abs(_hf_clo - _hf_chi) < DELTA_DEG))
_xcm, _xcl, _xch = pct(xcop_sf); _xom, _xol, _xoh = pct(xcop_so); _xrm, _xrl, _xrh = pct(xcop_req)
rows.append(_row("X-COP", _xrm, _xrl, _xrh, _xcm, _xcl, _xch, len(xcop_sf),
                 _xom, _xol, _xoh, degenerate=abs(_xc_clo - _xc_chi) < DELTA_DEG))
print(f"  [Delta chi2 branches] SPARC={abs(_sp_wh-_sp_wl):.1f}  HIFLUGCS={abs(_hf_clo-_hf_chi):.1f}  "
      f"X-COP={abs(_xc_clo-_xc_chi):.1f}  (degenerate if <{DELTA_DEG})")
print(f"  [X-COP] s<1 median={_xcm:.2f} [{_xcl:.2f},{_xch:.2f}]  s>1 mirror median={_xom:.2f} "
      f"[{_xol:.2f},{_xoh:.2f}]  r_eq median={_xrm:.2f} Mpc  (N={len(xcop_sf)})")
# Milky Way (published nbar4 fit): two degenerate branches published in the paper's
# results table -- s=2.00 (+0.33,-0.30) [s>=1] and mirror s=0.50 (+0.09,-0.07) [s<1]; the s<1 branch
# has the (marginally) lower chi2, so it is the filled one. r_eq from M_b^con = 7.36e10 Msun (MI).
_mw_req = r_newton_kpc(7.36e10) / 1000.0
rows.append(_row("Milky Way", _mw_req, _mw_req, _mw_req, 0.50, 0.43, 0.59, 152, 2.00, 1.70, 2.33,
                 degenerate=True))   # deep (v_H/v_N=0): the two branches are exactly degenerate

# ── Galaxy-galaxy WL (Mistele 2024, 4 mass bins x 5 radii): fit BOTH branches ─────
# Strategy: fit per bin first; aggregate star = median of per-bin distribution,
# error bars = 16th–84th percentile dispersion (same convention as SPARC/clusters).
# Degeneracy test uses the all-20-points pooled chi2 (Wilks criterion).
_bins_gg = {}
for ln in open(os.path.join(DATA, "mistele_gg_wl.txt")):
    if ln.startswith("#") or not ln.strip():
        continue
    p = ln.split()
    bk = int(p[0])
    _mb = 10**float(p[2]); _R = 10**float(p[3]); _v = float(p[4])
    _sv = np.sqrt(float(p[5])**2 + float(p[6])**2)
    _bins_gg.setdefault(bk, []).append((_mb, _R, _v, _sv))

_gw_req_arr, _gw_sf_arr, _gw_so_arr = [], [], []
with open(os.path.join(OUT, "galgal_per_bin.csv"), "w", newline="") as fh:
    ww = csv.writer(fh); ww.writerow(["r_eq", "s_fill", "s_open"])
    for bk in sorted(_bins_gg):
        pts = _bins_gg[bk]
        _mb0 = pts[0][0]
        _Rb  = np.array([q[1] for q in pts])
        _vb  = np.array([q[2] for q in pts])
        _svb = np.array([q[3] for q in pts])
        _gbbar = G_CODE * _mb0 / _Rb**2
        _gbobs = _vb**2 / _Rb
        _slog  = 2.0 * _svb / (_vb * np.log(10.0))
        _req_b = r_newton_kpc(_mb0) / 1000.0
        _shi_b, _chi_hi_b = _fit_s(_Rb, _gbbar, _gbobs, err=_slog, s_range=(1.0, 12.0))
        _slo_b, _chi_lo_b = _fit_s(_Rb, _gbbar, _gbobs, err=_slog, s_range=(0.15, 1.0))
        _sf_b = _shi_b if _chi_hi_b <= _chi_lo_b else _slo_b
        _so_b = _slo_b if _chi_hi_b <= _chi_lo_b else _shi_b
        ww.writerow([round(_req_b, 4), round(_sf_b, 3), round(_so_b, 3)])
        _gw_req_arr.append(_req_b); _gw_sf_arr.append(_sf_b); _gw_so_arr.append(_so_b)
        print(f"    [gg-WL bin {bk}] r_eq={_req_b:.3f} Mpc  s_fill={_sf_b:.3f}  "
              f"s_open={_so_b:.3f}  Dchi2={abs(_chi_hi_b-_chi_lo_b):.2f}")
print("Wrote galgal_per_bin.csv")
_gw_req_arr = np.array(_gw_req_arr)
_gw_sf_arr  = np.array(_gw_sf_arr)
_gw_so_arr  = np.array(_gw_so_arr)
_gwr_m, _gwr_l, _gwr_h = pct(_gw_req_arr)
_gwf_m, _gwf_l, _gwf_h = pct(_gw_sf_arr)   # s>1 branch
_gwo_m, _gwo_l, _gwo_h = pct(_gw_so_arr)   # s<1 branch

# Degeneracy: all-20-points pooled chi2
_gw_r = []; _gw_gb = []; _gw_go = []; _gw_sl = []
for bk_pts in _bins_gg.values():
    for (_mb, _R, _v, _sv) in bk_pts:
        _gw_r.append(_R); _gw_gb.append(G_CODE*_mb/_R**2); _gw_go.append(_v**2/_R)
        _gw_sl.append(2.0*_sv/(_v*np.log(10.0)))
_gw_r = np.array(_gw_r); _gw_gb = np.array(_gw_gb); _gw_go = np.array(_gw_go); _gw_sl = np.array(_gw_sl)
_, _gw_chi_hi = _fit_s(_gw_r, _gw_gb, _gw_go, err=_gw_sl, s_range=(1.0, 12.0))
_, _gw_chi_lo = _fit_s(_gw_r, _gw_gb, _gw_go, err=_gw_sl, s_range=(0.15, 1.0))
_gw_deg = abs(_gw_chi_hi - _gw_chi_lo) < DELTA_DEG

rows.append(_row("gal-gal WL", _gwr_m, _gwr_l, _gwr_h, _gwf_m, _gwf_l, _gwf_h, 20,
                 _gwo_m, _gwo_l, _gwo_h, degenerate=_gw_deg))
print(f"  [gal-gal WL] r_eq={_gwr_m:.3f} [{_gwr_l:.3f},{_gwr_h:.3f}] Mpc  "
      f"s>1={_gwf_m:.3f} [{_gwf_l:.3f},{_gwf_h:.3f}]  "
      f"s<1={_gwo_m:.3f} [{_gwo_l:.3f},{_gwo_h:.3f}]  "
      f"Dchi2(pooled)={abs(_gw_chi_hi-_gw_chi_lo):.2f} -> "
      f"{'degenerate' if _gw_deg else 'distinguishable'}")

os.makedirs(OUT, exist_ok=True)
with open(os.path.join(OUT, "reference_systems.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=["name", "r_eq", "r_eq_lo", "r_eq_hi",
                                       "s", "s_lo", "s_hi", "s_alt", "s_alt_lo", "s_alt_hi", "N",
                                       "degenerate"])
    w.writeheader(); w.writerows(rows)
print("Wrote reference_systems.csv  (+ sparc_per_galaxy.csv)")
for r in rows:
    alt = f"  |  mirror s_alt={r['s_alt']}" if r['s_alt'] != "" else ""
    print(f"  {r['name']:10s} N={r['N']:3d}  s={r['s']} [{r['s_lo']},{r['s_hi']}]  "
          f"r_eq={r['r_eq']} [{r['r_eq_lo']},{r['r_eq_hi']}] Mpc{alt}")
