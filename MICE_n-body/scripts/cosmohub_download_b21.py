#!/usr/bin/env python3
"""Download MICE2 B21 catalog from CosmoHub via REST API.

Usage:
    source ~/venv_vgrav/bin/activate
    python3 cosmohub_download_b21.py --user YOUR_EMAIL@example.com \
        --output ../data/MICE2_b21/mice2_b21_full.fits

The script authenticates, submits the SQL job, polls until done, and streams
the FITS file to disk. Pass --format hdf5 for HDF5 output (smaller, faster to
parse for large files).
"""

import argparse
import getpass
import sys
import time
import os
import requests

COSMOHUB_BASE = "https://api.cosmohub.pic.es"

QUERY_B21 = """
SELECT
    unique_gal_id, ra_gal, dec_gal, z_cgal,
    lmstellar, lmhalo,
    kappa, gamma1, gamma2,
    sdss_r_true
FROM
    micecatv2_0_view
WHERE
    z_cgal BETWEEN 0.01 AND 1.40
    AND lmstellar >= 8.20
ORDER BY unique_gal_id
""".strip()


def login(session, user, password):
    """Login via GET /user with Basic auth. Returns JWT from X-Token header."""
    import base64
    cred = base64.b64encode(f"{user}:{password}".encode()).decode()
    r = session.get(
        f"{COSMOHUB_BASE}/user",
        headers={"Authorization": f"Basic {cred}"},
        timeout=30,
    )
    r.raise_for_status()
    token = r.headers.get("X-Token") or r.headers.get("x-token")
    if not token:
        raise RuntimeError("Auth succeeded (200) but no X-Token header in response")
    return token


def submit_job(session, query, fmt="fits"):
    """Submit a SQL query. Returns query id."""
    r = session.post(
        f"{COSMOHUB_BASE}/queries",
        json={"sql": query, "format": fmt},
        timeout=60,
    )
    r.raise_for_status()
    qid = r.json().get("id")
    print(f"  Query submitted: id={qid}")
    return qid


def poll_job(session, qid, interval=60):
    """Poll until status==SUCCEEDED. Returns final query dict (has download_results).
    NOTE: the /queries/?id= filter is ignored server-side; must filter client-side."""
    url = f"{COSMOHUB_BASE}/queries"
    while True:
        r = session.get(url, timeout=30)
        r.raise_for_status()
        items = r.json()
        d = next((q for q in items if q.get("id") == qid), None)
        if not d:
            print(f"  Warning: query {qid} not found yet, retrying...", flush=True)
            time.sleep(interval)
            continue
        status = (d.get("status") or "unknown").upper()
        size = d.get("size") or 0
        print(f"  Status: {status}  size: {size/1e9:.2f} GB", flush=True)
        if status == "SUCCEEDED":
            return d
        if status in ("FAILED", "ERROR", "CANCELLED"):
            print(f"Query ended with status: {status}")
            sys.exit(1)
        time.sleep(interval)


def download_result(dl_url, output_path):
    """Stream download from the pre-signed download_results URL."""
    print(f"  Downloading to {output_path} ...", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with requests.get(dl_url, stream=True, timeout=7200) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        done = 0
        with open(output_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8 * 1024 * 1024):
                f.write(chunk)
                done += len(chunk)
                if total:
                    pct = 100.0 * done / total
                    print(f"\r  {done/1e9:.2f} / {total/1e9:.2f} GB ({pct:.1f}%)",
                          end="", flush=True)
    print(f"\n  Done: {output_path}")


def parse_args():
    p = argparse.ArgumentParser(description="Download MICE2 B21 from CosmoHub")
    p.add_argument("--user", default=None, help="CosmoHub username (email)")
    p.add_argument("--credentials-file", default=None,
                   help="File with username on line 1, password on line 2 (non-interactive auth)")
    p.add_argument("--resume-query-id", type=int, default=None,
                   help="Resume polling an existing CosmoHub query id (skip submission)")
    p.add_argument("--output", default="../data/MICE2_b21/mice2_b21_full.fits")
    p.add_argument("--format", choices=["fits", "hdf5"], default="fits",
                   help="Output format (default: fits)")
    p.add_argument("--poll-interval", type=int, default=60,
                   help="Seconds between status polls (default: 60)")
    return p.parse_args()


def main():
    args = parse_args()

    if args.credentials_file:
        with open(args.credentials_file) as f:
            lines = [l.strip() for l in f.readlines()]
        if len(lines) < 2:
            print("ERROR: credentials file must have username on line 1, password on line 2")
            sys.exit(1)
        user, password = lines[0], lines[1]
        print(f"Authenticating as {user} (credentials from file) ...", flush=True)
    elif args.user:
        user = args.user
        password = getpass.getpass(f"CosmoHub password for {user}: ")
    else:
        print("ERROR: provide --user or --credentials-file")
        sys.exit(1)

    session = requests.Session()
    session.headers.update({"Accept": "application/json",
                             "Origin": "https://cosmohub.pic.es"})

    token = login(session, user, password)
    session.headers["Authorization"] = f"Token {token}"
    print("  Auth OK")

    if args.resume_query_id:
        qid = args.resume_query_id
        print(f"Resuming query id={qid} (skipping submission)", flush=True)
    else:
        print("Submitting query ...", flush=True)
        print(QUERY_B21)
        qid = submit_job(session, QUERY_B21, fmt=args.format)

    print(f"Polling every {args.poll_interval}s ...", flush=True)
    result = poll_job(session, qid, interval=args.poll_interval)

    dl_url = result.get("download_results")
    if not dl_url:
        print("ERROR: no download_results URL in query response")
        sys.exit(1)

    ext = "h5" if args.format == "hdf5" else "fits"
    out = args.output.replace(".fits", f".{ext}") if args.format == "hdf5" else args.output
    download_result(dl_url, out)


if __name__ == "__main__":
    main()
