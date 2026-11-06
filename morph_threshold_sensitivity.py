"""Sensitivity of the morphological HMG fits (Table 1) to the MICE2 proxy classification.

The HMG scale of each KiDS morphological subset is fitted with the g_bar -> radius map of the
mass-matched MICE2 subsets (data/morph_gbar_r_mice.txt: B/T = 0.50, g - r = 0.60, matching
seed 1).  The fit is repeated with the maps obtained by moving the bulge-fraction cut to 0.45,
the colour cut to 0.55 and 0.65, and with seeds 2-4 of the random stellar-mass matching
(data/morph_map_variants/, written by MICE_n-body/scripts/run_mice_pipeline.sh).  For every
map and subset the script reports the best-fit s on each basin (s < 1 and s > 1) and the
difference Delta chi2 = chi2(s > 1) - chi2(s < 1) between the two minima.

Output: outputs/morph_threshold_sensitivity.csv
"""
import csv
import os
import numpy as np
from scipy.optimize import minimize_scalar
import hmg_model as h

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
FILES = {"late": "Sersicbin_1", "early": "Sersicbin_2", "blue": "Colorbin_1", "red": "Colorbin_2"}
MAPS = {"default": None}
MAPS.update({v: os.path.join(h.DATA_DIR, "morph_map_variants", f"map_{v}.txt")
             for v in ["bt045_gr060", "bt050_gr055", "bt050_gr065", "seed2", "seed3", "seed4"]})


def basin_fit(gb, go, ge, rk, lo, hi):
    """Best-fit s in [lo, hi] and its total chi2 (log-acceleration residuals)."""
    sig = ge / (go * np.log(10))
    c = lambda s: float(np.sum(((np.log10(go) - np.log10(h.gobs_hmg_gbar(gb, rk, s))) / sig) ** 2))
    r = minimize_scalar(c, bounds=(lo, hi), method="bounded")
    return r.x, r.fun


def main():
    os.makedirs(OUT, exist_ok=True)
    data = {k: h.load_morph(f"Fig-8_RAR-KiDS-isolated_{f}.txt") for k, f in FILES.items()}
    rows = []
    for name, path in MAPS.items():
        maps = h.morph_r_maps(path)
        for k, (gb, go, ge) in data.items():
            rk = maps[k](gb)
            s_lo, c_lo = basin_fit(gb, go, ge, rk, 0.15, 0.999)
            s_hi, c_hi = basin_fit(gb, go, ge, rk, 1.001, 8.0)
            rows.append({"map": name, "subset": k, "s_lo": round(s_lo, 3), "s_hi": round(s_hi, 3),
                         "s_best": round(s_lo if c_lo <= c_hi else s_hi, 3),
                         "dchi2_hi_minus_lo": round(c_hi - c_lo, 1)})
    with open(os.path.join(OUT, "morph_threshold_sensitivity.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"{'map':12s} {'subset':6s} {'s_lo':>6s} {'s_hi':>6s} {'s_best':>6s} {'dchi2':>7s}")
    for r in rows:
        print(f"{r['map']:12s} {r['subset']:6s} {r['s_lo']:6.3f} {r['s_hi']:6.3f} {r['s_best']:6.3f} "
              f"{r['dchi2_hi_minus_lo']:+7.1f}")
    for k in FILES:
        sb = [r["s_best"] for r in rows if r["subset"] == k]
        dc = [r["dchi2_hi_minus_lo"] for r in rows if r["subset"] == k]
        print(f"  {k:6s} s_best {min(sb):.3f}-{max(sb):.3f}   Delta chi2 {min(dc):+.1f} to {max(dc):+.1f}")
    print("Wrote outputs/morph_threshold_sensitivity.csv")


if __name__ == "__main__":
    main()
