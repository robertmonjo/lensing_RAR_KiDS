"""Per-morphology g_bar(R) map at 15 radii, from the MICE2 isolated lenses split
by B/T (Sersic proxy) and g-r colour, in the convention of Brouwer et al. (2021):
g_bar = G M_gal / R^2 (point mass), M_gal = M* (1 + f_cold) with
log10 f_cold = -0.69 log10(M*/Msun) + 6.63 (Boselli et al. 2014; Brouwer et al. 2021, Eq. 23).
Brouwer et al. set the g_bar bins lens by lens, so at fixed g_bar the annulus of lens i
lies at R_i = sqrt(G M_gal,i / g_bar); the map uses the mean R over the lenses,
g_bar(R) = (<sqrt(G M_gal)> / R)^2. As in Brouwer et al. (2021, Sect. 5.4), the two subsets
of each split are first reduced to the same stellar-mass distribution, so both subsets of a
split share one map. Output: r_Mpc + log10_gbar for late/early/blue/red."""
import argparse
import os
import numpy as np
from astropy.io import fits
from scipy.spatial import cKDTree
from mass_matching import match_mass

G_SI=6.674e-11; MSUN_KG=1.989e30; MPC_M=3.0857e22
R=np.logspace(np.log10(0.03), np.log10(3.0), 15)   # Mpc
_WORKDIR = os.environ.get("MICE_WORKDIR",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # -> MICE_n-body/
_ap = argparse.ArgumentParser(description="g_bar-radius map of the MICE morphological subsets")
_ap.add_argument("--isodir", default=os.path.join(_WORKDIR, "data", "MICE2_b21", "isolated"),
                 help="directory with mice2_isolated_bin{1..4}.fits")
_ap.add_argument("--morphfile", default=os.path.join(_WORKDIR, "data", "MICE2_b21",
                                                      "mice2_morph_b21_fullsky.fits"))
_ap.add_argument("--mstar-out", default=None,
                 help="CSV with the median log10 M* of each mass-matched subset")
_ap.add_argument("--bt-split", type=float, default=0.5, help="B/T threshold: late < bt_split <= early")
_ap.add_argument("--gr-split", type=float, default=0.6, help="g-r threshold: blue < gr_split <= red")
_ap.add_argument("--seed", type=int, default=1, help="seed of the random stellar-mass matching")
_args = _ap.parse_args()
ISODIR = _args.isodir

def mgal(lm):                                        # Msun
    return 10.0**lm*(1.0+10.0**(-0.69*lm+6.63))
def gbar_map(lm):                                    # (N,) -> (15,) SI
    sq=np.mean(np.sqrt(G_SI*mgal(lm)*MSUN_KG))
    return (sq/(R*MPC_M))**2

def colnames(hdu): return [c.name for c in hdu.columns]
def pick(cols,*opts):
    for o in opts:
        for c in cols:
            if c.lower()==o: return c
    raise KeyError(f"{opts} not in {cols}")

# ---- morphology catalog ----
with fits.open(_args.morphfile) as h:
    d=h[1].data; cn=colnames(h[1])
    mra=d[pick(cn,'ra_gal','ra')].astype(float)
    mdec=d[pick(cn,'dec_gal','dec')].astype(float)
    mz=d[pick(cn,'z_cgal','z','z_cgal_v')].astype(float)
    bt=d[pick(cn,'bulge_fraction')].astype(float)
    gr=d[pick(cn,'gr_cos','g_r','gr')].astype(float)
print("morph cols:",cn)
tree=cKDTree(np.column_stack([mra,mdec,mz]))

# ---- isolated lenses (4 bins) ----
LM=[]; BT=[]; GR=[]
for b in range(1,5):
    with fits.open(f"{ISODIR}/mice2_isolated_bin{b}.fits") as h:
        d=h[1].data; cn=colnames(h[1])
        ra=d[pick(cn,'ra_gal','ra')].astype(float)
        dec=d[pick(cn,'dec_gal','dec')].astype(float)
        z=d[pick(cn,'z_cgal','z')].astype(float)
        lm=d[pick(cn,'lm_msun')].astype(float)
    dist,idx=tree.query(np.column_stack([ra,dec,z]),k=1)
    ok=dist<1e-6
    LM.append(lm[ok]); BT.append(bt[idx[ok]]); GR.append(gr[idx[ok]])
    print(f"bin{b}: {len(lm)} iso, matched {ok.sum()}")
LM=np.concatenate(LM); BT=np.concatenate(BT); GR=np.concatenate(GR)
print(f"TOTAL matched isolated: {len(LM)}")
print(f"with 0.45<=B/T<0.55: {int(np.sum((BT>=0.45)&(BT<0.55)))}")

def dump(mask, label):
    lgb = np.log10(gbar_map(LM[mask]))
    print(f"### {label}  n_map={mask.sum()}")   # lenses of both matched subsets of the split
    for i in range(15):
        print(f"{R[i]:.5f} {lgb[i]:.6f}")


rows = []
BTS, GRS = _args.bt_split, _args.gr_split
for (l1, m1), (l2, m2) in [(("LATE_sersic", BT < BTS), ("EARLY_sersic", BT >= BTS)),
                          (("BLUE_color", GR < GRS), ("RED_color", GR >= GRS))]:
    k1, k2 = match_mass(LM, m1, m2, seed=_args.seed)
    both = k1 | k2                          # identical mass distribution -> one map per split
    print(f"# {l1}/{l2}: {m1.sum()}/{m2.sum()} lenses -> {k1.sum()}/{k2.sum()} after mass matching; "
          f"median log10 M* {np.median(LM[k1]):.3f}/{np.median(LM[k2]):.3f}")
    dump(both, l1)
    dump(both, l2)
    rows += [(l1, np.median(LM[k1]), k1.sum()), (l2, np.median(LM[k2]), k2.sum())]
print("### DONE")
if _args.mstar_out:
    with open(_args.mstar_out, "w") as fh:
        fh.write("subset,median_log10_mstar,n\n")
        for lab, lm, n in rows:
            fh.write(f"{lab},{lm:.4f},{n}\n")
