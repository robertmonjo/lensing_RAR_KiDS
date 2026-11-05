"""Per-morphology g_bar(R) map at 15 radii, from the MICE2 isolated lenses split
by B/T (Sersic proxy) and g-r colour, in the convention of Brouwer et al. (2021):
g_bar = G M_gal / R^2 (point mass), M_gal = M* (1 + f_cold) with
log10 f_cold = -0.69 log10(M*/Msun) + 6.63 (Boselli et al. 2014; Brouwer et al. 2021, Eq. 23).
Brouwer et al. set the g_bar bins lens by lens, so at fixed g_bar the annulus of lens i
lies at R_i = sqrt(G M_gal,i / g_bar); the map uses the mean R over the subset,
g_bar(R) = (<sqrt(G M_gal)> / R)^2. Output: r_Mpc + log10_gbar for late/early/blue/red."""
import os
import numpy as np
from astropy.io import fits
from scipy.spatial import cKDTree

G_SI=6.674e-11; MSUN_KG=1.989e30; MPC_M=3.0857e22
R=np.logspace(np.log10(0.03), np.log10(3.0), 15)   # Mpc
_WORKDIR = os.environ.get("MICE_WORKDIR",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # -> MICE_n-body/
ISODIR   = os.path.join(_WORKDIR, "data", "MICE2_isolated")

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
with fits.open(f"{ISODIR}/07_morph_raw.fits") as h:
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

def dump(mask,label):
    lgb=np.log10(gbar_map(LM[mask]))
    print(f"### {label}  n={mask.sum()}")
    for i in range(15):
        print(f"{R[i]:.5f} {lgb[i]:.6f}")

dump(BT<0.5 , "LATE_sersic")
dump(BT>=0.5, "EARLY_sersic")
dump(GR<0.6 , "BLUE_color")
dump(GR>=0.6, "RED_color")
print("### DONE")
