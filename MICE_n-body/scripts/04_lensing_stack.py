"""04_lensing_stack.py
Compute g_obs via actual MICE tangential-shear stacking ('Approach A-true').

Replaces the NFW-profile approximation in 04_mock_esd.py + 05_stack_rar.py.
For each isolated lens galaxy, background MICE galaxies (z_s > z_l + DZ_BUFFER)
are used as shear tracers:

    gamma_t  = -(gamma1_s * cos(2*phi) + gamma2_s * sin(2*phi))
    DeltaSigma(r) = mean[ Sigma_cr(z_l, z_s) * gamma_t ] over pairs in bin r
    g_obs(r) = CONV * DeltaSigma(r)

g_bar is also stacked from each lens's stellar+gas mass (Baldry+2012 correction).

Output: rar_stacked{suffix}_bin{b}.txt — same format as 05_stack_rar.py
        r_Mpc   log10_gbar   log10_gobs

Usage (from MICE_n-body/scripts/):
    conda activate mice
    python 04_lensing_stack.py --bin 1 2 3 4
    python 04_lensing_stack.py --bin 1 2 3 4 --suffix _photoz \\
        --lens-pattern mice2_isolated_photoz_bin{b}.fits
"""

import argparse
import numpy as np
import os
from astropy.io import fits
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
from scipy.spatial import cKDTree

# ── Constants ──────────────────────────────────────────────────────────────────
G_SI    = 6.674e-11       # m³ kg⁻¹ s⁻²
C_SI    = 2.998e8         # m s⁻¹
MSUN_KG = 1.989e30        # kg
MPC_M   = 3.0857e22       # m Mpc⁻¹
PC_M    = 3.086e16        # m pc⁻¹
DEG2RAD = np.pi / 180.0

# g_obs = CONV * DeltaSigma [M_sun/pc^2]  gives m/s^2
CONV_ESD = 4.0 * G_SI * MSUN_KG / PC_M**2   # m s⁻² per (M_sun pc⁻²)

# Cosmology (MICE2)
COSMO = FlatLambdaCDM(H0=70.0, Om0=0.25)

# Radial bins (log-spaced Mpc, matching existing pipeline)
R_EDGES = np.logspace(np.log10(0.02), np.log10(4.5), 16)   # 15 bins
R_MID   = np.sqrt(R_EDGES[:-1] * R_EDGES[1:])              # geometric midpoint
N_BINS  = len(R_MID)

DZ_BUFFER   = 0.05    # min Δz between lens and source
Z_LENS_MAX  = 0.35    # max lens redshift (beyond this, too few bg sources at z_s<0.50)
THETA_EXTRA = 1.05    # slight margin when building KD-tree search radius

# Paths (relative to this repo by default; override with $MICE_WORKDIR)
WORKDIR  = os.environ.get(
    "MICE_WORKDIR",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # -> MICE_n-body/
DATADIR  = os.path.join(WORKDIR, "data", "MICE2_isolated")
RAW_FITS = os.path.join(DATADIR, "mice2_raw_test.fits")


# ── Gas correction (Baldry+2012 as used in B21) ───────────────────────────────
def cold_gas_fraction(lm_star):
    """Return f_cold = M_gas/M_star (Baldry+2012 eq. for isolated galaxies)."""
    return 10.0 ** (-0.43 * (lm_star - 10.0) - 0.49)


# ── Sigma_cr in M_sun/pc^2 ────────────────────────────────────────────────────
def sigma_cr(z_l, z_s):
    """Critical surface density [M_sun/pc^2] for scalar z_l and array z_s."""
    D_l  = COSMO.angular_diameter_distance(z_l).to(u.Mpc).value
    D_s  = COSMO.angular_diameter_distance(z_s).to(u.Mpc).value
    D_ls = COSMO.angular_diameter_distance(z_l, z_s).to(u.Mpc).value
    ok   = D_ls > 0.0
    Scr  = np.zeros(len(z_s))
    # SI: kg/m^2
    Scr[ok] = (C_SI**2 / (4.0 * np.pi * G_SI)
               * D_s[ok] / (D_l * D_ls[ok]) / MPC_M)
    # M_sun/Mpc^2 → M_sun/pc^2
    Scr[ok] *= MPC_M**2 / MSUN_KG * 1e-12
    return Scr, ok


# ── Main stacking routine ─────────────────────────────────────────────────────
def stack_bin(lens_file, suffix, bin_num, src_ra, src_dec, src_z, src_g1, src_g2, tree):
    print(f"\n  Bin {bin_num} ({suffix}): loading lenses ...", flush=True)
    with fits.open(lens_file) as hf:
        d = hf[1].data
        ra_l   = d['ra_gal'].astype(np.float64)
        dec_l  = d['dec_gal'].astype(np.float64)
        z_l    = d['z_cgal'].astype(np.float64)
        lm_msun = d['lm_msun'].astype(np.float64)   # log10(M_star [M_sun])
    # Restrict to z_l < Z_LENS_MAX: beyond this threshold the source catalog
    # (z_s < 0.50) provides too few background galaxies for reliable GGL.
    keep = z_l < Z_LENS_MAX
    ra_l = ra_l[keep]; dec_l = dec_l[keep]; z_l = z_l[keep]; lm_msun = lm_msun[keep]
    N_lens = len(ra_l)
    print(f"  {N_lens} lenses after z_l<{Z_LENS_MAX} cut, "
          f"z ∈ [{z_l.min():.3f}, {z_l.max():.3f}]", flush=True)

    # Accumulators: ESD and gbar per radial bin
    # Use inverse-Sigma_cr^2 weighting (standard GGL estimator):
    #   ΔΣ = sum(w * gamma_t * Sigma_cr) / sum(w)   with w = Sigma_cr^{-2}
    #      = sum(gamma_t / Sigma_cr) / sum(Sigma_cr^{-2})
    ESD_num  = np.zeros(N_BINS)   # sum(gamma_t / Sigma_cr)
    ESD_den  = np.zeros(N_BINS)   # sum(1 / Sigma_cr^2)
    ESD_cnt  = np.zeros(N_BINS, dtype=int)
    gbar_sum = np.zeros(N_BINS)
    gbar_cnt = np.zeros(N_BINS, dtype=int)

    for il in range(N_lens):
        zl   = z_l[il]
        ral  = ra_l[il]
        decl = dec_l[il]

        # Angular diameter distance to lens [Mpc]
        D_A_l = COSMO.angular_diameter_distance(zl).to(u.Mpc).value

        # KD-tree query: sources within r_max / D_A_l (radians)
        theta_max_rad = (R_EDGES[-1] / D_A_l) * THETA_EXTRA
        decl_r = decl * DEG2RAD
        ral_r  = ral  * DEG2RAD
        lxyz   = np.array([np.cos(decl_r)*np.cos(ral_r),
                            np.cos(decl_r)*np.sin(ral_r),
                            np.sin(decl_r)])
        chord  = 2.0 * np.sin(theta_max_rad / 2.0)
        idx    = np.asarray(tree.query_ball_point(lxyz, chord))
        if idx.size == 0:
            continue

        # Background selection: z_s > z_l + DZ_BUFFER
        zs   = src_z[idx]
        bg   = zs > zl + DZ_BUFFER
        if bg.sum() == 0:
            continue

        ib   = idx[bg]
        zs_b = zs[bg]
        ras  = src_ra[ib];  decs = src_dec[ib]
        g1s  = src_g1[ib];  g2s  = src_g2[ib]

        # Flat-sky projected separation [Mpc] and position angle
        dra  = (ras - ral) * np.cos(decl_r) * DEG2RAD   # rad
        ddec = (decs - decl) * DEG2RAD                   # rad
        theta = np.hypot(dra, ddec)
        r_mpc = theta * D_A_l

        # Radial bin assignment
        bin_id = np.digitize(r_mpc, R_EDGES) - 1
        in_range = (bin_id >= 0) & (bin_id < N_BINS)
        if in_range.sum() == 0:
            continue

        ras_ok = r_mpc[in_range]; dra_ok = dra[in_range]; ddec_ok = ddec[in_range]
        bid_ok = bin_id[in_range]
        zs_ok  = zs_b[in_range]
        g1_ok  = g1s[in_range];  g2_ok = g2s[in_range]

        # Tangential shear: MICE convention (phi measured from East, CCW)
        # gamma_t = +(g1 cos(2phi) + g2 sin(2phi))  [MICE North-based shear]
        phi   = np.arctan2(ddec_ok, dra_ok)
        gam_t = +(g1_ok * np.cos(2.0*phi) + g2_ok * np.sin(2.0*phi))

        # Sigma_cr per pair [M_sun/pc^2]
        Scr, ok = sigma_cr(zl, zs_ok)

        # Accumulate with inverse-Sigma_cr^2 weighting
        for ib2 in range(N_BINS):
            m = ok & (bid_ok == ib2) & (Scr > 0)
            if m.sum() == 0:
                continue
            Scr_m = Scr[m]
            ESD_num[ib2] += np.sum(gam_t[m] / Scr_m)
            ESD_den[ib2] += np.sum(1.0 / Scr_m**2)
            ESD_cnt[ib2] += m.sum()

        # gbar: baryonic (stellar + gas) acceleration at R_MID
        lm_s   = lm_msun[il]
        f_cold = cold_gas_fraction(lm_s)
        m_bar  = (10.0 ** lm_s) * (1.0 + f_cold)   # M_sun
        for ib2 in range(N_BINS):
            r_pc = R_MID[ib2] * 1e6   # Mpc -> pc
            gb_si = G_SI * m_bar * MSUN_KG / (r_pc * PC_M)**2   # m/s^2
            gbar_sum[ib2] += gb_si
            gbar_cnt[ib2] += 1

        if (il % 500) == 0:
            print(f"    {il}/{N_lens}", flush=True)

    # Stacked values
    rows = []
    for ib2 in range(N_BINS):
        if ESD_den[ib2] == 0 or gbar_cnt[ib2] == 0:
            continue
        # ΔΣ = ESD_num / ESD_den  (inverse-Sigma_cr^2 weighted estimator)
        ds   = ESD_num[ib2] / ESD_den[ib2]   # DeltaSigma [M_sun/pc^2]
        gb   = gbar_sum[ib2] / gbar_cnt[ib2]  # mean gbar [m/s^2]
        gobs = CONV_ESD * ds                   # m/s^2
        if gobs <= 0.0 or gb <= 0.0:
            continue
        rows.append([R_MID[ib2], np.log10(gb), np.log10(gobs)])

    if not rows:
        print(f"  WARNING: no valid rows for bin {bin_num} ({suffix})", flush=True)
        return

    rows = np.array(rows)
    fname = f"rar_stacked{suffix}_bin{bin_num}.txt"
    out   = os.path.join(DATADIR, fname)
    hdr   = "r_Mpc  log10_gbar  log10_gobs"
    np.savetxt(out, rows, header=hdr, fmt="%.6e")
    print(f"  Saved: {out}")
    print(f"  Bins with data: {len(rows)}/{N_BINS}, "
          f"mean ESD_cnt/bin={ESD_cnt[ESD_cnt>0].mean():.0f}", flush=True)


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="Stack MICE GGL signal (Approach A-true)"
    )
    ap.add_argument("--bin",  nargs="+", type=int, default=[1, 2, 3, 4],
                    metavar="B", dest="bins")
    ap.add_argument("--suffix", default="",
                    help="output suffix, e.g. '_photoz'")
    ap.add_argument("--lens-pattern",
                    default="mice2_isolated_bin{b}.fits",
                    help="isolated lens FITS filename with {b} placeholder")
    args = ap.parse_args()

    # Load source catalog once
    print("Loading source catalog ...", flush=True)
    with fits.open(RAW_FITS) as hf:
        sd = hf[1].data
        src_ra  = sd['ra_gal'].astype(np.float64)
        src_dec = sd['dec_gal'].astype(np.float64)
        src_z   = sd['z_cgal'].astype(np.float64)
        src_g1  = sd['gamma1'].astype(np.float64)
        src_g2  = sd['gamma2'].astype(np.float64)
    print(f"  {len(src_ra)} sources, z ∈ [{src_z.min():.3f}, {src_z.max():.3f}]",
          flush=True)

    # Build KD-tree on unit sphere
    print("Building KD-tree ...", flush=True)
    ra_r  = src_ra  * DEG2RAD
    dec_r = src_dec * DEG2RAD
    xyz   = np.column_stack([
        np.cos(dec_r) * np.cos(ra_r),
        np.cos(dec_r) * np.sin(ra_r),
        np.sin(dec_r)
    ])
    tree = cKDTree(xyz)
    print("  Done.", flush=True)

    for b in args.bins:
        lens_file = os.path.join(DATADIR, args.lens_pattern.format(b=b))
        if not os.path.exists(lens_file):
            print(f"  Lens file not found: {lens_file}", flush=True)
            continue
        stack_bin(lens_file, args.suffix, b,
                  src_ra, src_dec, src_z, src_g1, src_g2, tree)

    print("\nDone. Run 08_b21_band.py next to combine ztrue + photoz.", flush=True)


if __name__ == "__main__":
    main()
