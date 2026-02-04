from __future__ import annotations

import argparse
import os
import shutil
import sys
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

REQUIRED = ["ETHUSDC_1m.csv", "BTCUSDC_1m.csv"]


def copy_mode(src_dir: Path, out_dir: Path) -> None:
    src_dir = src_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    missing = []
    for name in REQUIRED:
        src = src_dir / name
        dst = out_dir / name
        if not src.exists():
            missing.append(str(src))
            continue
        shutil.copy2(src, dst)
        print(f"Copied: {src} -> {dst}")

    if missing:
        print("\nMissing files:")
        for m in missing:
            print(f"  - {m}")
        sys.exit(2)


def url_mode(url: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_zip = out_dir / "_market_data.zip"

    print(f"Downloading zip: {url}")
    urlretrieve(url, tmp_zip)

    with zipfile.ZipFile(tmp_zip, "r") as z:
        names = set(z.namelist())
        # allow zip to contain nested paths; extract only the required files
        extracted = 0
        for req in REQUIRED:
            match = next((n for n in names if n.endswith(req)), None)
            if not match:
                raise FileNotFoundError(f"{req} not found inside zip")
            z.extract(match, out_dir)
            src = out_dir / match
            dst = out_dir / req
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))
            # clean nested dirs if any (best-effort)
            try:
                src.parent.rmdir()
            except Exception:
                pass
            print(f"Extracted: {req} -> {dst}")
            extracted += 1

        print(f"Done. Extracted {extracted} files.")
    tmp_zip.unlink(missing_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data", help="Output data directory (default: data)")
    ap.add_argument("--src", help="Source directory containing CSVs (can be local path or \\LAPTOP\\share)")
    ap.add_argument("--url", help="Direct URL to a zip containing the CSVs")
    args = ap.parse_args()

    out_dir = Path(args.out)

    if bool(args.src) == bool(args.url):
        print("Provide exactly one of: --src OR --url")
        sys.exit(1)

    if args.src:
        copy_mode(Path(args.src), out_dir)
    else:
        url_mode(args.url, out_dir)


if __name__ == "__main__":
    main()
