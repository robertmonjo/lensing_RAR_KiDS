"""Download a subset of the MICE2 galaxy catalogue from CosmoHub.

Requirements:
    pip install cosmohub-api astropy

Usage:
    python 01_download_mice.py --user <email> --output ../data/MICE2_isolated/mice2_raw.fits

For a test run (small equatorial stripe, ~300 deg^2):
    python 01_download_mice.py --user <email> --test --output ../data/MICE2_isolated/mice2_raw_test.fits
"""

import argparse
import getpass
import os
import sys

try:
    from cosmohub.api import CosmoHub
    HAS_COSMOHUB = True
except ImportError:
    HAS_COSMOHUB = False

QUERY_FULL = """
SELECT
    unique_gal_id, ra_gal, dec_gal, z_cgal,
    lmstellar, lmhalo,
    kappa, gamma1, gamma2
FROM
    micecatv2_0_view
WHERE
    z_cgal BETWEEN 0.01 AND 0.50
    AND lmstellar >= 9.20
ORDER BY unique_gal_id
"""

QUERY_TEST = """
SELECT
    unique_gal_id, ra_gal, dec_gal, z_cgal,
    lmstellar, lmhalo,
    kappa, gamma1, gamma2
FROM
    micecatv2_0_view
WHERE
    z_cgal BETWEEN 0.01 AND 0.50
    AND lmstellar >= 9.20
    AND dec_gal BETWEEN -5.0 AND 5.0
    AND ra_gal BETWEEN 0.0 AND 60.0
ORDER BY unique_gal_id
"""

# NOTE on lmstellar threshold:
#   B21 uses M_star > 10^9.5 M_sun.  MICE2 lmstellar is log10(M_star/h^-2 M_sun).
#   With h=0.70: log10(h^-2) = -2*log10(0.70) = +0.3010
#   So B21 threshold in MICE units: 9.5 - 0.3010 = 9.199 ~ 9.20.
#   We also need neighbour galaxies for the isolation check, so use same cut for all.


def parse_args():
    p = argparse.ArgumentParser(description="Download MICE2 catalogue from CosmoHub")
    p.add_argument("--user", required=True, help="CosmoHub username (email)")
    p.add_argument("--output", default="../data/MICE2_isolated/mice2_raw.fits")
    p.add_argument("--test", action="store_true",
                   help="Use small equatorial test stripe (dec in [-5,5], ra in [0,60])")
    return p.parse_args()


def manual_download_instructions(test=False):
    query = QUERY_TEST if test else QUERY_FULL
    print("\ncosmohub-api is not installed.  Manual download instructions:")
    print("=" * 60)
    print("1. Go to https://cosmohub.pic.es and log in.")
    print("2. Click 'Catalogues' -> 'MICE2' -> 'mice2_mhalo'.")
    print("3. Click 'SQL Query' and paste the following SQL:")
    print()
    print(query.strip())
    print()
    print("4. Choose output format: FITS or HDF5.")
    print(f"5. Save the output to:")
    print(f"   ../data/MICE2_isolated/{'mice2_raw_test.fits' if test else 'mice2_raw.fits'}")
    print("=" * 60)


def main():
    args = parse_args()
    query = QUERY_TEST if args.test else QUERY_FULL

    label = "TEST stripe (dec in [-5,5], ra in [0,60])" if args.test else "FULL sky"
    print(f"MICE2 download: {label}")
    print(f"Output: {args.output}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    if not HAS_COSMOHUB:
        manual_download_instructions(test=args.test)
        sys.exit(0)

    password = getpass.getpass(f"CosmoHub password for {args.user}: ")
    ch = CosmoHub(args.user, password)
    print("Connected to CosmoHub.  Submitting query ...")
    print(query.strip())

    job = ch.submit_job(query.strip(), output_format="fits")
    print(f"Job submitted: ID={job.job_id}.  Waiting for completion ...")
    job.wait()

    if job.status == "COMPLETED":
        print(f"Downloading to {args.output} ...")
        job.download(args.output)
        print("Done.")
    else:
        print(f"Job failed with status: {job.status}")
        sys.exit(1)


if __name__ == "__main__":
    main()
