"""Stellar-mass matching of two galaxy subsets (Brouwer et al. 2021, Sect. 5.4).

Only the overlap of the two log10 M* distributions is kept: in every bin of width
DLM both subsets are randomly reduced to the smaller of their two counts, so the
matched subsets share one stellar-mass distribution. Used by 07_morphology_split.py
and 09_morph_gbar_r.py with the same bins and seed, so both select the same lenses.
"""
import numpy as np

DLM = 0.05
SEED = 1


def match_mass(lm, m1, m2, dlm=DLM, seed=SEED):
    """Boolean masks (k1, k2) selecting mass-matched members of subsets m1 and m2."""
    rng = np.random.default_rng(seed)
    edges = np.arange(np.floor(lm.min() / dlm) * dlm, lm.max() + dlm, dlm)
    idx = np.digitize(lm, edges)
    k1 = np.zeros(lm.size, bool)
    k2 = np.zeros(lm.size, bool)
    for e in np.unique(idx):
        a = np.where(m1 & (idx == e))[0]
        b = np.where(m2 & (idx == e))[0]
        n = min(a.size, b.size)
        if n:
            k1[rng.choice(a, n, replace=False)] = True
            k2[rng.choice(b, n, replace=False)] = True
    return k1, k2
