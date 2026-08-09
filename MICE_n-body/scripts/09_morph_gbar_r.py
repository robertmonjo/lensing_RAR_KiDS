"""Per-morphology mean g_bar(r) at the 15 fixed annuli, from the MICE2
isolated lenses split by B/T (Sersic proxy) and g-r colour. Replicates 07's
gbar_from_stellar exactly. Output: r_Mpc + log10_gbar for late/early/blue/red."""
import os
import numpy as np
from astropy.io import fits
from scipy.spatial import cKDTree

G_SI=6.674e-11; MSUN_KG=1.989e30; PC_M=3.086e16
G_PC3=G_SI*MSUN_KG/PC_M**3
R=np.logspace(np.log10(0.03), np.log10(3.0), 15)   # Mpc
_WORKDIR = os.environ.get("MICE_WORKDIR",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # -> MICE_n-body/
ISODIR   = os.path.join(_WORKDIR, "data", "MICE2_isolated")

def gas_fraction(lm):
    return 10.0**np.clip(-0.43*(lm-10.0)-0.49, -3.0, 0.0)
def gbar_stellar(lm):                                # (N,) -> (N,15) SI
    mstar=10.0**lm; fg=gas_fraction(lm); mbar=mstar*(1.0+fg/(1.0-fg))
    r_pc=R*1.0e6
    return 4.0*G_PC3*PC_M*mbar[:,None]/(np.pi*r_pc[None,:]**2)

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

def dump(mask,label):
    gb=gbar_stellar(LM[mask])           # (n,15)
    lgb=np.log10(gb.mean(axis=0))
    print(f"### {label}  n={mask.sum()}")
    for i in range(15):
        print(f"{R[i]:.5f} {lgb[i]:.6f}")

dump(BT<0.5 , "LATE_sersic")
dump(BT>=0.5, "EARLY_sersic")
dump(GR<0.6 , "BLUE_color")
dump(GR>=0.6, "RED_color")
print("### DONE")
