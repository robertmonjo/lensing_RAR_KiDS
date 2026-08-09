"""Compute mock ESD (Excess Surface Density) and g_obs for MICE2 isolated lenses.

Two approaches are implemented:
  A) Approach A: NFW profile using MICE halo masses (fast, approximate).
     The full shear-based approach (Approach A-true) requires the MICE
     source catalogue, which is a separate download from CosmoHub.
  B) Approach B: Direct point-mass ESD from stellar mass (equivalent to
     g_bar but without baryonic correction -- for testing only).

In practice, B21 used the full shear-based stacking.  This script implements
the NFW approach as the primary fast method, and notes where to plug in
the shear-based calculation.

g_obs conversion (B21 Eq. 7):
    g_obs [m/s^2] = 4 * G * Delta_Sigma [M_sun/pc^2] * [pc/m]

Requirements:
    pip install astropy numpy scipy

Usage:
    python 04_mock_esd.py --indir ../data/MICE2_isolated/ \
                           --outdir ../data/MICE2_isolated/
"""

import argparse
import os
import numpy as np
from astropy.io import fits
from astropy.cosmology import FlatLambdaCDM

COSMO = FlatLambdaCDM(H0=70.0, Om0=0.25)

# Physical constants
G_SI = 6.674e-11
MSUN_KG = 1.989e30
PC_M = 3.086e16
MPC_PC = 1.0e6
G_PC3_MSUN_S2 = G_SI * MSUN_KG / PC_M**3  # 4.52e-30

CONV_ESD_TO_GOBS = 4.0 * G_PC3_MSUN_S2 * PC_M  # M_sun/pc^2 -> m/s^2

# Radial bins matching B21 (15 log-spaced from 0.03 to 3 Mpc)
R_BINS_MPC = np.logspace(np.log10(0.03), np.log10(3.0), 15)

# NFW concentration from Duffy+2008 (Table 1, relaxed sample):
#   c = A * (M / M_pivot)^B * (1+z)^C
DUFFY_A = 5.71
DUFFY_B = -0.084
DUFFY_C = -0.47
DUFFY_MPIVOT = 2e12  # M_sun / h (pivot mass)

H_MICE = 0.70
LOG10_H = np.log10(H_MICE)


def nfw_concentration(m_halo_msun, z):
    """NFW concentration from Duffy+2008.  m_halo in M_sun (not h^-1 M_sun)."""
    m_h = m_halo_msun * H_MICE  # M_sun / h (as in Duffy+2008)
    c = DUFFY_A * (m_h / DUFFY_MPIVOT) ** DUFFY_B * (1.0 + z) ** DUFFY_C
    return np.maximum(c, 1.0)


def nfw_delta_sigma(r_mpc, m200_msun, c, z):
    """
    Excess surface density Delta_Sigma for an NFW profile.

    Delta_Sigma(r) = Sigma_bar(<r) - Sigma(r)
    where Sigma_bar is the mean surface density within r.

    Returns Delta_Sigma in M_sun / pc^2.
    """
    rho_c = COSMO.critical_density(z).to("Msun/Mpc^3").value  # M_sun/Mpc^3
    delta_c = (200.0 / 3.0) * c**3 / (np.log(1 + c) - c / (1 + c))
    r200 = (3 * m200_msun / (4 * np.pi * 200 * rho_c)) ** (1.0 / 3.0)  # Mpc
    r_s = r200 / c

    x = np.asarray(r_mpc, float) / r_s
    rho_s = delta_c * rho_c  # M_sun/Mpc^3

    # Projected (2D) NFW surface density Sigma(R) [M_sun/Mpc^2]
    def sigma_nfw(x):
        out = np.zeros_like(x)
        m1 = x < 1.0
        m2 = x > 1.0
        me = x == 1.0

        # x < 1
        xx = x[m1]
        out[m1] = (2 * rho_s * r_s / (xx**2 - 1) *
                   (1.0 - 2.0 / np.sqrt(1 - xx**2) *
                    np.arctanh(np.sqrt((1 - xx) / (1 + xx)))))

        # x > 1
        xx = x[m2]
        out[m2] = (2 * rho_s * r_s / (xx**2 - 1) *
                   (1.0 - 2.0 / np.sqrt(xx**2 - 1) *
                    np.arctan(np.sqrt((xx - 1) / (1 + xx)))))

        # x = 1
        out[me] = 2.0 * rho_s * r_s / 3.0

        return out

    # Mean surface density within projected radius R: Sigma_bar(<R)
    def sigma_bar_nfw(x):
        out = np.zeros_like(x)
        m1 = x < 1.0
        m2 = x > 1.0
        me = x == 1.0

        xx = x[m1]
        out[m1] = (4 * rho_s * r_s / (xx**2) *
                   (2.0 / np.sqrt(1 - xx**2) *
                    np.arctanh(np.sqrt((1 - xx) / (1 + xx))) +
                    np.log(xx / 2.0)))

        xx = x[m2]
        out[m2] = (4 * rho_s * r_s / (xx**2) *
                   (2.0 / np.sqrt(xx**2 - 1) *
                    np.arctan(np.sqrt((xx - 1) / (1 + xx))) +
                    np.log(xx / 2.0)))

        out[me] = 4 * rho_s * r_s * (1.0 + np.log(0.5))

        return out

    sig = sigma_nfw(x)       # M_sun/Mpc^2
    sig_bar = sigma_bar_nfw(x)  # M_sun/Mpc^2
    delta_sig = sig_bar - sig   # M_sun/Mpc^2

    # Convert to M_sun/pc^2: 1 Mpc = 1e6 pc => 1 M_sun/Mpc^2 = 1e-12 M_sun/pc^2
    return delta_sig * 1e-12


def process_bin(b, indir, outdir, r_bins_mpc=R_BINS_MPC, suffix=""):
    infile = os.path.join(indir, f"mice2_isolated{suffix}_bin{b}.fits")
    if not os.path.exists(infile):
        print(f"  {infile} not found, skipping.")
        return None

    with fits.open(infile) as hdul:
        t = hdul[1].data
        lm_halo = t["lmstellar"].astype(float)  # placeholder: use lmhalo if available
        lm_msun = t["lm_msun"].astype(float)
        z_gal = t["z_cgal"].astype(float)

    # Try to read lmhalo; fall back to abundance-matching estimate
    try:
        with fits.open(infile) as hdul:
            lm_halo = hdul[1].data["lmhalo"].astype(float)
        # MICE lmhalo is log10(M_halo / h^-1 M_sun); convert to M_sun
        m_halo_msun = 10.0 ** lm_halo / H_MICE
    except Exception:
        # Rough stellar-to-halo mass relation (Behroozi+2010 peak)
        lm_halo_msun = 12.0 + 1.5 * (lm_msun - 11.0)
        m_halo_msun = 10.0 ** lm_halo_msun
        print(f"  lmhalo not found; using abundance-matching estimate.")

    n_gal = len(lm_msun)
    print(f"  Bin {b}: {n_gal:,} galaxies")

    # Per-galaxy ESD profiles
    esd_all = np.zeros((n_gal, len(r_bins_mpc)))
    for i in range(n_gal):
        c = nfw_concentration(m_halo_msun[i], z_gal[i])
        esd_all[i] = nfw_delta_sigma(r_bins_mpc, m_halo_msun[i], c, z_gal[i])

    # g_obs from ESD
    g_obs_all = CONV_ESD_TO_GOBS * esd_all  # m/s^2

    # Stacked mean
    g_obs_mean = g_obs_all.mean(axis=0)
    g_obs_p16 = np.percentile(g_obs_all, 16, axis=0)
    g_obs_p84 = np.percentile(g_obs_all, 84, axis=0)

    # Save stacked mean
    out_mean = os.path.join(outdir, f"gobs_mock{suffix}_bin{b}.txt")
    np.savetxt(out_mean,
               np.column_stack([r_bins_mpc, g_obs_mean, g_obs_p16, g_obs_p84]),
               header="r_Mpc  g_obs_mean  g_obs_p16  g_obs_p84  [m/s^2]",
               comments="# ")
    print(f"  Saved: {out_mean}")

    # Save full array as .npy for band computation in step 5
    npy_out = os.path.join(outdir, f"gobs_full{suffix}_bin{b}.npy")
    np.save(npy_out, g_obs_all)
    print(f"  Saved full g_obs array to {npy_out}")

    return g_obs_mean


def parse_args():
    p = argparse.ArgumentParser(description="Compute mock ESD and g_obs for MICE2 lenses")
    p.add_argument("--indir", default="../data/MICE2_isolated/")
    p.add_argument("--outdir", default="../data/MICE2_isolated/")
    p.add_argument("--suffix", default="",
                   help="File suffix: reads mice2_isolated{suffix}_bin{b}.fits, "
                        "writes gobs_mock{suffix}_bin{b}.txt")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    print("Computing mock ESD (NFW approach, Duffy+2008 concentration)")
    print(f"g_obs conversion: 4 * G * Delta_Sigma * [pc/m]")
    print()

    for b in range(1, 5):
        print(f"Bin {b}:")
        process_bin(b, args.indir, args.outdir, suffix=args.suffix)
        print()

    print("Done.")
    print()
    print("NOTE: For the full shear-based approach (Approach A-true), replace")
    print("  nfw_delta_sigma() with tangential shear stacking from the MICE")
    print("  source catalogue.  This requires downloading mice2_source from")
    print("  CosmoHub (~10x larger than the lens catalogue).")


if __name__ == "__main__":
    main()
