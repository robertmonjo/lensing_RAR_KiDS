"""Split isolated MICE2 lenses by morphology and colour, compute stacked RAR.

Two splits (matching Brouwer+2021 Fig. 8):
  1. Sérsic proxy — bulge_fraction (= B/T directly in MICE2):
       Late  (disc):  B/T < 0.5   ~  Sérsic n < 2.5
       Early (bulge): B/T >= 0.5  ~  Sérsic n >= 2.5
  2. Colour — gr_cos (rest-frame g−r from MICE2 SED fitting):
       Blue (late):  gr_cos < 0.6
       Red  (early): gr_cos >= 0.6

Requires:
  1. Isolated lens FITS files (from 02_select_isolated.py):
       mice2_isolated_bin{1..4}.fits
       Columns: ra_gal, dec_gal, z_cgal, lmstellar, lm_msun, lmhalo
  2. Morphology download from CosmoHub (07_morph_raw.fits):
       SQL:
         SELECT unique_gal_id, ra_gal, dec_gal, z_cgal,
                bulge_fraction, gr_cos
         FROM micecatv2_0_view
         WHERE z_cgal BETWEEN 0.01 AND 0.50
           AND lmstellar >= 9.20
           AND dec_gal BETWEEN -5.0 AND 5.0
           AND ra_gal BETWEEN 0.0 AND 60.0
         ORDER BY unique_gal_id

Usage:
    python 07_morphology_split.py \\
        --morphfile ../data/MICE2_isolated/07_morph_raw.fits \\
        --isodir    ../data/MICE2_isolated/ \\
        --outdir    ../data/MICE2_isolated/
"""

import argparse
import os
import numpy as np
from astropy.io import fits
from astropy.cosmology import FlatLambdaCDM
from scipy.spatial import cKDTree

COSMO = FlatLambdaCDM(H0=70.0, Om0=0.25)

# Physical constants (same as scripts 03-04)
G_SI           = 6.674e-11
MSUN_KG        = 1.989e30
PC_M           = 3.086e16
MPC_PC         = 1.0e6
KPC_M          = 3.0857e19
G_PC3_MSUN_S2  = G_SI * MSUN_KG / PC_M**3
CONV_ESD_TO_GOBS = 4.0 * G_PC3_MSUN_S2 * PC_M

H_MICE   = 0.70
LOG10_H2 = 2.0 * np.log10(H_MICE)

R_BINS_MPC = np.logspace(np.log10(0.03), np.log10(3.0), 15)

# NFW (Duffy+2008)
DUFFY_A = 5.71; DUFFY_B = -0.084; DUFFY_C = -0.47; DUFFY_MPIVOT = 2e12
RHO_CRIT_Z0 = (3 * (70e3 / 3.0857e22)**2) / (8 * np.pi * G_SI) * KPC_M**3 / MSUN_KG


def nfw_concentration(m_halo_msun, z):
    m_h = m_halo_msun * H_MICE
    return np.maximum(DUFFY_A * (m_h / DUFFY_MPIVOT)**DUFFY_B * (1.0 + z)**DUFFY_C, 1.0)


def nfw_delta_sigma(r_mpc, m200_msun, c, z):
    rho_c  = COSMO.critical_density(z).to("Msun/Mpc^3").value
    delta_c = (200.0 / 3.0) * c**3 / (np.log(1 + c) - c / (1 + c))
    r200   = (3 * m200_msun / (4 * np.pi * 200 * rho_c))**(1.0 / 3.0)
    r_s    = r200 / c
    x      = np.asarray(r_mpc, float) / r_s
    rho_s  = delta_c * rho_c

    def sigma_nfw(x):
        out = np.zeros_like(x)
        m1 = x < 1.0; m2 = x > 1.0; me = x == 1.0
        xx = x[m1]
        out[m1] = (2*rho_s*r_s/(xx**2-1) *
                   (1.0 - 2.0/np.sqrt(1-xx**2)*np.arctanh(np.sqrt((1-xx)/(1+xx)))))
        xx = x[m2]
        out[m2] = (2*rho_s*r_s/(xx**2-1) *
                   (1.0 - 2.0/np.sqrt(xx**2-1)*np.arctan(np.sqrt((xx-1)/(1+xx)))))
        out[me] = 2.0*rho_s*r_s/3.0
        return out

    def sigma_bar_nfw(x):
        out = np.zeros_like(x)
        m1 = x < 1.0; m2 = x > 1.0; me = x == 1.0
        xx = x[m1]
        out[m1] = (4*rho_s*r_s/xx**2 *
                   (2.0/np.sqrt(1-xx**2)*np.arctanh(np.sqrt((1-xx)/(1+xx))) + np.log(xx/2.0)))
        xx = x[m2]
        out[m2] = (4*rho_s*r_s/xx**2 *
                   (2.0/np.sqrt(xx**2-1)*np.arctan(np.sqrt((xx-1)/(1+xx))) + np.log(xx/2.0)))
        out[me] = 4*rho_s*r_s*(1.0 + np.log(0.5))
        return out

    return (sigma_bar_nfw(x) - sigma_nfw(x)) * 1e-12   # M_sun/pc^2


def gobs_from_halo(m_halo_msun, z, r_bins_mpc=R_BINS_MPC):
    c   = nfw_concentration(m_halo_msun / H_MICE, z)
    esd = nfw_delta_sigma(r_bins_mpc, m_halo_msun, c, z)
    return CONV_ESD_TO_GOBS * esd


def gas_fraction(lm_msun):
    log_fgas = -0.43 * (lm_msun - 10.0) - 0.49
    return 10.0**np.clip(log_fgas, -3.0, 0.0)


def gbar_from_stellar(lm_msun, r_bins_mpc=R_BINS_MPC):
    m_star = 10.0**lm_msun
    f_gas  = gas_fraction(lm_msun)
    m_bar  = m_star * (1.0 + f_gas / (1.0 - f_gas))
    r_pc   = r_bins_mpc * MPC_PC
    return 4.0 * G_PC3_MSUN_S2 * PC_M * m_bar / (np.pi * r_pc**2)


def stack_subsample(iso_data, mask, label=""):
    """iso_data: dict with arrays; mask: boolean selection."""
    n = mask.sum()
    if n == 0:
        print(f"  {label}: 0 galaxies, skip")
        return None
    print(f"  {label}: {n:,} galaxies")
    lm_msun = iso_data["lm_msun"][mask]
    lmhalo  = iso_data["lmhalo"][mask]
    z_gal   = iso_data["z_cgal"][mask]
    m_halo  = 10.0**lmhalo / H_MICE

    go_all = np.zeros((n, len(R_BINS_MPC)))
    gb_all = np.zeros((n, len(R_BINS_MPC)))
    for i in range(n):
        go_all[i] = gobs_from_halo(m_halo[i], z_gal[i])
        gb_all[i] = gbar_from_stellar(lm_msun[i])

    go_mean = go_all.mean(axis=0)
    go_p16  = np.percentile(go_all, 16, axis=0)
    go_p84  = np.percentile(go_all, 84, axis=0)
    gb_mean = gb_all.mean(axis=0)
    return {
        "r_mpc": R_BINS_MPC,
        "log10_gbar": np.log10(gb_mean),
        "log10_gobs": np.log10(go_mean),
        "log10_gobs_p16": np.log10(go_p16),
        "log10_gobs_p84": np.log10(go_p84),
        "n_gal": n,
    }


def match_morphology(iso_data, morph_ra, morph_dec, morph_z, morph_bt):
    """Match isolated galaxies to morphology catalog by (ra, dec, z)."""
    xyz_iso = np.column_stack([
        iso_data["ra_gal"], iso_data["dec_gal"], iso_data["z_cgal"]])
    xyz_morph = np.column_stack([morph_ra, morph_dec, morph_z])

    tree = cKDTree(xyz_morph)
    dist, idx = tree.query(xyz_iso, k=1)

    # Accept matches with distance < 1e-6 (exact float match)
    matched = dist < 1e-6
    bt_iso  = np.full(len(iso_data["ra_gal"]), np.nan)
    bt_iso[matched] = morph_bt[idx[matched]]

    n_total   = len(iso_data["ra_gal"])
    n_matched = matched.sum()
    print(f"  Matched {n_matched:,} / {n_total:,} galaxies "
          f"({100*n_matched/n_total:.1f}%)")
    return bt_iso


def _save_result(outdir, fname, res):
    if res is None:
        return
    path = os.path.join(outdir, fname)
    out  = np.column_stack([res["log10_gbar"], res["log10_gobs"],
                            res["log10_gobs_p16"], res["log10_gobs_p84"], res["r_mpc"]])
    np.savetxt(path, out,
               header=f"log10_gbar log10_gobs log10_p16 log10_p84 r_Mpc  (n={res['n_gal']})")
    print(f"  Saved: {path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--morphfile",  default="../data/MICE2_isolated/07_morph_raw.fits")
    p.add_argument("--isodir",     default="../data/MICE2_isolated/")
    p.add_argument("--outdir",     default="../data/MICE2_isolated/")
    p.add_argument("--bt_split",   type=float, default=0.5,
                   help="B/T threshold: late < bt_split <= early")
    p.add_argument("--gr_split",   type=float, default=0.6,
                   help="g-r threshold: blue < gr_split <= red")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    print(f"Loading morphology file: {args.morphfile}")
    with fits.open(args.morphfile) as hdul:
        t = hdul[1].data
        m_ra  = t["ra_gal"].astype(float)
        m_dec = t["dec_gal"].astype(float)
        m_z   = t["z_cgal"].astype(float)
        bt    = t["bulge_fraction"].astype(float)   # B/T directly
        gr    = t["gr_cos"].astype(float)            # rest-frame g-r color
    print(f"  {len(m_ra):,} morphology entries")

    results_late  = {}
    results_early = {}
    results_blue  = {}
    results_red   = {}

    for b in range(1, 5):
        isofile = os.path.join(args.isodir, f"mice2_isolated_bin{b}.fits")
        if not os.path.exists(isofile):
            print(f"Bin {b}: {isofile} not found, skipping")
            continue

        print(f"\nBin {b}:")
        with fits.open(isofile) as hdul:
            t_iso = hdul[1].data
            iso = {k: t_iso[k].astype(float)
                   for k in ["ra_gal","dec_gal","z_cgal","lm_msun","lmhalo"]}

        bt_iso = match_morphology(iso, m_ra, m_dec, m_z, bt)
        gr_iso = match_morphology(iso, m_ra, m_dec, m_z, gr)

        # ── Sérsic proxy (B/T) ────────────────────────────────────────────
        mask_late  = bt_iso < args.bt_split
        mask_early = bt_iso >= args.bt_split
        res_late   = stack_subsample(iso, mask_late,  label=f"BT-Late  (B/T<{args.bt_split})")
        res_early  = stack_subsample(iso, mask_early, label=f"BT-Early (B/T>={args.bt_split})")
        for label, res in [("sersic_late", res_late), ("sersic_early", res_early)]:
            _save_result(args.outdir, f"morph_{label}_bin{b}.txt", res)

        results_late[b]  = res_late
        results_early[b] = res_early

        # ── Colour (gr_cos) ───────────────────────────────────────────────
        mask_blue = gr_iso < args.gr_split
        mask_red  = gr_iso >= args.gr_split
        res_blue  = stack_subsample(iso, mask_blue, label=f"Blue (g-r<{args.gr_split})")
        res_red   = stack_subsample(iso, mask_red,  label=f"Red  (g-r>={args.gr_split})")
        for label, res in [("color_blue", res_blue), ("color_red", res_red)]:
            _save_result(args.outdir, f"morph_{label}_bin{b}.txt", res)

        results_blue[b] = res_blue
        results_red[b]  = res_red

    # Stacked over all bins (for morphological panels like B21 Fig 8)
    print("\nStacking all bins for morphological panels ...")
    for label, results in [("sersic_late",  results_late),
                           ("sersic_early", results_early),
                           ("color_blue",   results_blue),
                           ("color_red",    results_red)]:
        valid = [r for r in results.values() if r]
        if not valid:
            continue
        all_gb  = np.concatenate([r["log10_gbar"]     for r in valid])
        all_go  = np.concatenate([r["log10_gobs"]     for r in valid])
        all_p16 = np.concatenate([r["log10_gobs_p16"] for r in valid])
        all_p84 = np.concatenate([r["log10_gobs_p84"] for r in valid])
        idx = np.argsort(all_gb)
        fname = os.path.join(args.outdir, f"morph_{label}_allbins.txt")
        np.savetxt(fname,
                   np.column_stack([all_gb[idx], all_go[idx], all_p16[idx], all_p84[idx]]),
                   header="log10_gbar log10_gobs log10_p16 log10_p84")
        print(f"  Saved: {fname}")

    print("\nDone.")


if __name__ == "__main__":
    main()
