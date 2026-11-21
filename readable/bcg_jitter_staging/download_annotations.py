#!/usr/bin/env python3
"""Download NSRR MESA R-point CSVs with size/MD5 validation; never log the token."""
import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

ROOT = Path(__file__).resolve().parents[1]
FOLDER = "polysomnography/annotations-rpoints"


def session():
    result = requests.Session()
    result.mount(
        "https://",
        HTTPAdapter(
            max_retries=Retry(
                total=3, backoff_factor=2, status_forcelist=[429, 500, 502, 503, 504]
            )
        ),
    )
    return result


def checksum(path):
    md5 = hashlib.md5()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            md5.update(chunk)
    return md5.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--token-file", type=Path, default=Path.home() / ".config/nsrr/token"
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    token = os.environ.get("NSRR_TOKEN") or (
        args.token_file.read_text().strip() if args.token_file.exists() else ""
    )
    if not token:
        raise SystemExit(
            "NSRR token missing. Save it in ~/.config/nsrr/token with permissions 600."
        )
    client = session()
    try:
        profile = client.get(
            "https://sleepdata.org/api/v1/account/profile.json",
            params={"auth_token": token},
            timeout=60,
        )
        if (
            profile.status_code != 200
            or profile.json().get("authenticated") is not True
        ):
            raise SystemExit(
                "Saved token does not authenticate with NSRR. Run bcg_jitter_staging/configure_token.py."
            )
    except (requests.RequestException, ValueError):
        raise SystemExit("Could not securely verify NSRR authentication.") from None
    try:
        response = client.get(
            "https://sleepdata.org/api/v1/datasets/mesa/files.json",
            params={"path": FOLDER, "auth_token": token},
            timeout=60,
        )
    except requests.RequestException:
        raise SystemExit(
            "Could not connect securely to NSRR; inspect network/CA configuration."
        ) from None
    if response.status_code != 200:
        raise SystemExit(
            f"NSRR annotation listing returned HTTP {response.status_code}. Check token and MESA access."
        )
    entries = response.json()
    if not isinstance(entries, list):
        raise SystemExit("NSRR returned an unexpected listing.")
    wanted = {
        p.name.removeprefix("mesa-sleep-")
        for p in (ROOT / "data/derived/full_cohort_windows_v1/recordings").glob(
            "mesa-sleep-*"
        )
    }
    selected = [
        item
        for item in entries
        if item.get("is_file")
        and item["file_name"].removeprefix("mesa-sleep-").removesuffix("-rpoint.csv")
        in wanted
    ]
    available = {
        item["file_name"].removeprefix("mesa-sleep-").removesuffix("-rpoint.csv")
        for item in selected
    }
    unavailable = sorted(wanted - available)
    if args.limit:
        selected = selected[: args.limit]
    if not selected:
        raise SystemExit("No matching R-point annotation files in NSRR listing.")
    destination = ROOT / "data/raw/mesa" / FOLDER
    destination.mkdir(parents=True, exist_ok=True)

    def download(item):
        name = item["file_name"]
        if Path(name).name != name:
            raise ValueError("Unsafe NSRR filename")
        target = destination / name
        expected_size, expected_md5 = int(item["file_size"]), item["file_checksum_md5"]
        if (
            target.exists()
            and target.stat().st_size == expected_size
            and checksum(target) == expected_md5
        ):
            return {
                "file": name,
                "bytes": expected_size,
                "md5": expected_md5,
                "status": "verified_existing",
            }
        url = f"https://sleepdata.org/datasets/mesa/files/a/{token}/m/nsrr-gem-v8-0-0/{FOLDER}/{name}"
        temporary = target.with_suffix(".csv.part")
        # Exceptions may contain the authenticated URL. Replace them with filename-only errors.
        try:
            with session().get(url, stream=True, timeout=120) as response:
                if response.status_code != 200 or "text/html" in response.headers.get(
                    "Content-Type", ""
                ):
                    raise ValueError("No CSV download access")
                with temporary.open("wb") as file:
                    for chunk in response.iter_content(1024 * 1024):
                        file.write(chunk)
        except (requests.RequestException, ValueError):
            temporary.unlink(missing_ok=True)
            raise RuntimeError(
                f"Download failed for {name}; check MESA authorization/network."
            ) from None
        if (
            temporary.stat().st_size != expected_size
            or checksum(temporary) != expected_md5
        ):
            temporary.unlink(missing_ok=True)
            raise RuntimeError(f"NSRR checksum/size verification failed for {name}")
        with temporary.open(newline="", encoding="utf-8-sig") as file:
            fields = csv.DictReader(file).fieldnames or []
            if not {"RPoint", "Type"} <= set(fields):
                raise ValueError(f"Unexpected annotation CSV columns for {name}")
        temporary.replace(target)
        return {
            "file": name,
            "bytes": expected_size,
            "md5": expected_md5,
            "status": "downloaded",
        }

    report = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(download, item) for item in selected]):
            result = future.result()
            report.append(result)
            print(
                f"{len(report)}/{len(selected)} {result['file']} {result['status']}",
                flush=True,
            )
    expected = len(selected)
    manifest = {
        "expected": expected,
        "downloaded_or_verified": len(report),
        "files": report,
        "missing_from_nsrr_listing": unavailable,
        "limited_smoke": bool(args.limit),
    }
    (destination / "download_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    if len(report) != expected:
        raise SystemExit(
            f"Only {len(report)}/{expected} prepared participants have downloadable beat annotations."
        )


if __name__ == "__main__":
    try:
        main()
    except requests.RequestException:
        sys.exit(
            "NSRR network request failed; authenticated URLs have been suppressed."
        )
