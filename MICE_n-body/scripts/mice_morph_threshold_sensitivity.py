#!/usr/bin/env python3
"""Sensitivity of the MICE2 morphological stacks to the proxy thresholds.

The MICE2 subsets use proxies of the KiDS classifiers: bulge fraction B/T (default
split 0.50) for the Sersic index and rest-frame g - r (default 0.60) for u - r.
07_morphology_split.py is run for B/T in {0.45, 0.50, 0.55} and g - r in {0.55, 0.60,
0.65}; for every varied threshold the stacked g_obs of each subset is compared with the
default, mass bin by mass bin at the same radii, and the maximum |g_obs / g_obs,default - 1|
of each mass bin is reported (empty where either run lacks lenses in that bin).  The same
comparison between the default thresholds with seeds 2-4 of the random stellar-mass
matching and seed 1 gives the sampling spread against which the threshold changes are read.

Usage:
    python mice_morph_threshold_sensitivity.py --morphfile ../data/MICE2_b21/mice2_morph_b21_fullsky.fits \\
        --isodir ../data/MICE2_b21/isolated --outdir ../data/MICE2_b21/sensitivity [--compare-only]

Output: OUTDIR/bt{BT}_gr{GR}/morph_*.txt and OUTDIR/mice_morph_threshold_sensitivity.csv
"""
import argparse
import csv
import os
import subprocess
import sys
import numpy as np

SCRIPT_07 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "07_morphology_split.py")
DEFAULT = (0.50, 0.60)
VARIED = {"B/T": [(0.45, 0.60), (0.55, 0.60)], "g-r": [(0.50, 0.55), (0.50, 0.65)]}
SEEDS = [2, 3, 4]
TYPES = {"B/T": {"sersic_late": "late", "sersic_early": "early"},
         "g-r": {"color_blue": "blue", "color_red": "red"}}


def tag(bt, gr, seed=1):
    t = f"bt{int(round(bt * 100)):03d}_gr{int(round(gr * 100)):03d}"
    return t if seed == 1 else f"{t}_seed{seed}"


def run_split(args, bt, gr, seed=1):
    sub = os.path.join(args.outdir, tag(bt, gr, seed))
    os.makedirs(sub, exist_ok=True)
    cmd = [sys.executable, SCRIPT_07, "--morphfile", args.morphfile, "--isodir", args.isodir,
           "--outdir", sub, "--bt_split", str(bt), "--gr_split", str(gr), "--seed", str(seed)]
    print("Running", " ".join(cmd[2:]), flush=True)
    subprocess.run(cmd, check=True)


def load_bin(sub, key, b):
    """(r_Mpc, g_obs) of morph_{key}_bin{b}.txt, or None if the subset is empty in that bin."""
    path = os.path.join(sub, f"morph_{key}_bin{b}.txt")
    if not os.path.exists(path):
        return None
    d = np.loadtxt(path, comments="#", ndmin=2)
    return d[:, 4], 10.0 ** d[:, 1]


def deltas(args, ref, var, key):
    """Per mass bin: maximum |g_var / g_ref - 1| [%] over common radii (None if a run lacks the bin)."""
    out = {}
    for b in range(1, 5):
        d = load_bin(os.path.join(args.outdir, ref), key, b)
        v = load_bin(os.path.join(args.outdir, var), key, b)
        if d is None or v is None:
            out[b] = None
            continue
        common = np.intersect1d(np.round(d[0], 6), np.round(v[0], 6))
        gd = np.array([d[1][np.argmin(np.abs(d[0] - r))] for r in common])
        gv = np.array([v[1][np.argmin(np.abs(v[0] - r))] for r in common])
        out[b] = float(np.max(np.abs(gv / gd - 1.0))) * 100.0
    return out


def fmt(x):
    return "" if x is None else f"{x:.2f}"


def compare(args):
    rows = []
    for proxy, varied in VARIED.items():
        for key, label in TYPES[proxy].items():
            seed = [deltas(args, tag(*DEFAULT), tag(*DEFAULT, s), key) for s in SEEDS]
            spread = {b: max((d[b] for d in seed if d[b] is not None), default=None) for b in range(1, 5)}
            for bt, gr in varied:
                thr = bt if proxy == "B/T" else gr
                dv = deltas(args, tag(*DEFAULT), tag(bt, gr), key)
                row = {"proxy": proxy, "type": label, "threshold": thr}
                for b in range(1, 5):
                    row[f"delta_bin{b}_pct"] = fmt(dv[b])
                    row[f"seed_spread_bin{b}_pct"] = fmt(spread[b])
                rows.append(row)
                print(f"  {proxy:>4} {label:>6} {thr:.2f}  " + "  ".join(
                    f"bin {b}: {fmt(dv[b]) or '-':>6} % (seeds {fmt(spread[b]) or '-':>6} %)" for b in range(1, 5)))
    path = os.path.join(args.outdir, "mice_morph_threshold_sensitivity.csv")
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("Wrote", path)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--morphfile", required=True)
    p.add_argument("--isodir", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--compare-only", action="store_true",
                   help="skip 07_morphology_split.py and compare existing runs")
    args = p.parse_args()
    if not args.compare_only:
        for bt, gr in [DEFAULT] + VARIED["B/T"] + VARIED["g-r"]:
            run_split(args, bt, gr)
        for s in SEEDS:
            run_split(args, *DEFAULT, seed=s)
    compare(args)


if __name__ == "__main__":
    main()
