#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

BUCKET = "https://cse-cic-ids2018.s3.amazonaws.com"
PREFIX = "Processed Traffic Data for ML Algorithms/"
DEST_DIR = Path("data/raw/processed_csv")
DEST_DIR.mkdir(parents=True, exist_ok=True)

NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}


def list_objects(prefix: str):
    marker = None
    while True:
        params = {"prefix": prefix}
        if marker:
            params["marker"] = marker
        url = BUCKET + "/?" + urllib.parse.urlencode(params)
        with urllib.request.urlopen(url) as resp:
            xml_data = resp.read()
        root = ET.fromstring(xml_data)
        contents = root.findall("s3:Contents", NS)
        if not contents:
            break
        for c in contents:
            key = c.find("s3:Key", NS).text
            size = int(c.find("s3:Size", NS).text)
            yield key, size
        is_truncated = root.find("s3:IsTruncated", NS)
        if is_truncated is None or is_truncated.text != "true":
            break
        next_marker = root.find("s3:NextMarker", NS)
        if next_marker is not None and next_marker.text:
            marker = next_marker.text
        else:
            marker = contents[-1].find("s3:Key", NS).text


def download_file(key: str, size: int):
    filename = Path(key).name
    dest = DEST_DIR / filename
    if dest.exists() and dest.stat().st_size == size:
        print(f"[skip] {filename}")
        return
    url = BUCKET + "/" + urllib.parse.quote(key)
    print(f"[get] {filename} ({size/1024/1024:.1f} MiB)")
    tmp = dest.with_suffix(dest.suffix + ".part")
    # basic streaming download
    with urllib.request.urlopen(url) as resp, open(tmp, "wb") as f:
        downloaded = 0
        start = time.time()
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)
            if downloaded % (50 * 1024 * 1024) == 0:
                elapsed = max(time.time() - start, 1e-6)
                speed = downloaded / 1024 / 1024 / elapsed
                print(f"  ... {downloaded/1024/1024:.1f} MiB ({speed:.1f} MiB/s)")
    os.replace(tmp, dest)


def main():
    keys = [(k, s) for k, s in list_objects(PREFIX) if k.endswith(".csv")]
    if not keys:
        print("No CSV files found.")
        return 1
    total = sum(s for _, s in keys)
    print(f"Found {len(keys)} CSV files, total {total/1024/1024:.1f} MiB")
    for key, size in keys:
        download_file(key, size)
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
