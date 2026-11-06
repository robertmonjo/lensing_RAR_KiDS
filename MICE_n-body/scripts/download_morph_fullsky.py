#!/usr/bin/env python3
"""Download MICE2 morphological data (bulge_fraction + gr_cos) for the full
B21 footprint (no RA/Dec restriction, z<=0.50, lmstellar>=9.20).

Usage:
    source ~/venv_vgrav/bin/activate
    python3 download_morph_fullsky.py \
        --credentials-file cosmohub_credentials.txt \
        --output ../data/MICE2_b21/mice2_morph_b21_fullsky.fits
"""
import argparse, os, sys, time, base64, requests

COSMOHUB_BASE = "https://api.cosmohub.pic.es"

QUERY_MORPH_FULLSKY = """
SELECT unique_gal_id, ra_gal, dec_gal, z_cgal, bulge_fraction, gr_cos
FROM micecatv2_0_view
WHERE z_cgal BETWEEN 0.01 AND 0.50
  AND lmstellar >= 9.20
ORDER BY unique_gal_id
""".strip()


def login(session, user, password):
    cred = base64.b64encode(f"{user}:{password}".encode()).decode()
    r = session.get(f"{COSMOHUB_BASE}/user",
                    headers={"Authorization": f"Basic {cred}"}, timeout=30)
    r.raise_for_status()
    token = r.headers.get("X-Token") or r.headers.get("x-token")
    if not token:
        raise RuntimeError("Auth OK but no X-Token")
    return token


def submit_job(session, query, fmt="fits"):
    r = session.post(f"{COSMOHUB_BASE}/queries",
                     json={"sql": query, "format": fmt}, timeout=60)
    r.raise_for_status()
    qid = r.json().get("id")
    print(f"  Query submitted: id={qid}", flush=True)
    return qid


def poll_job(session, qid, interval=60):
    url = f"{COSMOHUB_BASE}/queries"
    while True:
        r = session.get(url, timeout=30)
        r.raise_for_status()
        items = r.json()
        d = next((q for q in items if q.get("id") == qid), None)
        if not d:
            print(f"  Query {qid} not found yet, retrying...", flush=True)
            time.sleep(interval)
            continue
        status = (d.get("status") or "unknown").upper()
        size = d.get("size") or 0
        print(f"  Status: {status}  size: {size/1e9:.2f} GB", flush=True)
        if status == "SUCCEEDED":
            return d
        if status in ("FAILED", "ERROR", "CANCELLED"):
            print(f"Query ended: {status}")
            sys.exit(1)
        time.sleep(interval)


def download_result(dl_url, output_path):
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
                    print(f"\r  {done/1e9:.2f} / {total/1e9:.2f} GB "
                          f"({100.0*done/total:.1f}%)", end="", flush=True)
    print(f"\n  Done: {output_path}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--credentials-file", required=True)
    p.add_argument("--output",
                   default="../data/MICE2_b21/mice2_morph_b21_fullsky.fits")
    p.add_argument("--format", choices=["fits", "hdf5"], default="fits")
    p.add_argument("--poll-interval", type=int, default=60)
    p.add_argument("--resume-query-id", type=int, default=None)
    args = p.parse_args()

    with open(args.credentials_file) as f:
        lines = [l.strip() for l in f.readlines()]
    if len(lines) < 2:
        print("ERROR: credentials file needs username (line 1) and password (line 2)")
        sys.exit(1)
    user, password = lines[0], lines[1]
    print(f"Authenticating as {user} ...", flush=True)

    session = requests.Session()
    session.headers.update({"Accept": "application/json",
                             "Origin": "https://cosmohub.pic.es"})

    token = login(session, user, password)
    session.headers["Authorization"] = f"Token {token}"
    print("  Auth OK", flush=True)

    if args.resume_query_id:
        qid = args.resume_query_id
        print(f"Resuming query {qid}", flush=True)
    else:
        print("Submitting morphology query (full sky, z<=0.50, lmstellar>=9.20)...", flush=True)
        print(f"  SQL:\n{QUERY_MORPH_FULLSKY}", flush=True)
        qid = submit_job(session, QUERY_MORPH_FULLSKY, fmt=args.format)

    print(f"Polling query {qid} every {args.poll_interval}s ...", flush=True)
    result = poll_job(session, qid, interval=args.poll_interval)

    dl_url = result.get("download_results")
    if not dl_url:
        print("ERROR: no download_results URL in response")
        sys.exit(1)
    download_result(dl_url, args.output)
    print("=== Download complete ===", flush=True)


if __name__ == "__main__":
    main()
