"""Table 2 (tab:mice_comparison): reduced chi^2 on the outer-radii subsets.

Subsets (KiDS-1000 isolated lenses, Brouwer et al. 2021, r >= 0.164 Mpc, 10 points per bin):
  N = 30 : mass bins 2-4
  N = 40 : mass bins 1-4

Statistics (Sect. 2.4): diagonal KiDS uncertainties with the one-sided width on the
side of the data point facing the prediction, in log10 g (chi2_nu_log) and in g
(chi2_nu_lin). Model parameters are the optima of chi2_nu_log; chi2_nu = chi2/(N-k).

Rows
  HMG, one global s (k = 1)                 fitted here
  HMG, one s per mass bin (k = 4, N = 40)   fitted here
  MOND, free a0 (k = 1)                     fitted here (McGaugh 2016 interpolation)
  MICE, Brouwer et al. (2021)               published value (full covariance, linear only)
  MICE, reproduced (k = 0)                  MICE_n-body/scripts/11_mice_lin_chi2.py

Output: outputs/tab_mice_comparison.csv
"""

import csv
import importlib.util
import os
import sys
import numpy as np
from scipy.optimize import minimize_scalar

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import hmg_model as h

_spec = importlib.util.spec_from_file_location(
    "mice_chi2", os.path.join(_HERE, "MICE_n-body", "scripts", "11_mice_lin_chi2.py"))
mice = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mice)

A0_STAR = 1.20e-10   # m/s^2
R_OUTER_MPC = 0.164


def outer(b):
    return b[b[:, 0] >= R_OUTER_MPC]


def chi2_lin(pred, b):
    g = b[:, 1]
    sig = np.where(pred >= g, b[:, 3] - g, g - b[:, 2])
    return float(np.sum(((g - pred) / sig) ** 2))


def chi2_log(pred, b):
    return h._chi2_arr(pred, b)


def hmg_pred(b, mbar, s):
    return h.gobs_hmg(h._r_kpc(b), mbar, s)


def mond_pred(b, mbar, a0):
    gbar = h.gbar_si(b[:, 0], mbar)
    return gbar / (1.0 - np.exp(-np.sqrt(gbar / a0)))


def fit(pred_fun, bins, masses, lo, hi):
    f = lambda p: sum(chi2_log(pred_fun(b, m, p), b) for b, m in zip(bins, masses))
    p = minimize_scalar(f, bounds=(lo, hi), method="bounded", options={"xatol": 1e-6}).x
    c_log = f(p)
    c_lin = sum(chi2_lin(pred_fun(b, m, p), b) for b, m in zip(bins, masses))
    return p, c_log, c_lin


rows = []
for label, sel in [("N30", slice(1, 4)), ("N40", slice(0, 4))]:
    bins = [outer(b) for b in h.BINS[sel]]
    masses = h.MBAR[sel]
    n = sum(len(b) for b in bins)
    print(f"\n{n} points (mass bins {sel.start + 1}-{sel.stop})")

    s, cl, cn = fit(hmg_pred, bins, masses, 0.2, 0.9)
    print(f"  HMG global   s = {s:.3f}          chi2_nu log = {cl/(n-1):.2f}  lin = {cn/(n-1):.2f}")
    rows.append({"subset": label, "model": "HMG (one global s)", "N": n, "k": 1,
                 "parameter": f"s={s:.3f}", "chi2_nu_lin": round(cn / (n - 1), 2),
                 "chi2_nu_log": round(cl / (n - 1), 2)})

    if label == "N40":
        cl_pb = cn_pb = 0.0; svals = []
        for b, m in zip(bins, masses):
            sb, cb, nb = fit(hmg_pred, [b], [m], 0.1, 2.0)
            cl_pb += cb; cn_pb += nb; svals.append(f"{sb:.3f}")
        print(f"  HMG per bin  s = {', '.join(svals)}  chi2_nu log = {cl_pb/(n-4):.2f}  lin = {cn_pb/(n-4):.2f}")
        rows.append({"subset": label, "model": "HMG (one s per mass bin)", "N": n, "k": 4,
                     "parameter": "s=" + "/".join(svals), "chi2_nu_lin": round(cn_pb / (n - 4), 2),
                     "chi2_nu_log": round(cl_pb / (n - 4), 2)})

    a, cl, cn = fit(lambda b, m, x: mond_pred(b, m, x * A0_STAR), bins, masses, 0.5, 4.0)
    print(f"  MOND free a0 = {a:.2f} a0*       chi2_nu log = {cl/(n-1):.2f}  lin = {cn/(n-1):.2f}")
    rows.append({"subset": label, "model": "MOND (free a0)", "N": n, "k": 1,
                 "parameter": f"a0={a:.2f}a0*", "chi2_nu_lin": round(cn / (n - 1), 2),
                 "chi2_nu_log": round(cl / (n - 1), 2)})

    if label == "N30":
        rows.append({"subset": label, "model": "MICE (Brouwer et al. 2021, full covariance)",
                     "N": n, "k": 0, "parameter": "-", "chi2_nu_lin": 1.66, "chi2_nu_log": "-"})
    nm, cl, cn = mice.mice_chi2(list(range(sel.start + 1, sel.stop + 1)))
    assert nm == n
    print(f"  MICE (z_true stack)               chi2_nu log = {cl/n:.2f}  lin = {cn/n:.2f}")
    rows.append({"subset": label, "model": "MICE (reproduced, z_true stack)", "N": n, "k": 0,
                 "parameter": "-", "chi2_nu_lin": round(cn / n, 2), "chi2_nu_log": round(cl / n, 2)})

out_dir = os.path.join(_HERE, "outputs"); os.makedirs(out_dir, exist_ok=True)
path = os.path.join(out_dir, "tab_mice_comparison.csv")
with open(path, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=["subset", "model", "N", "k", "parameter", "chi2_nu_lin", "chi2_nu_log"])
    w.writeheader(); w.writerows(rows)
print(f"\nWrote {path}")
