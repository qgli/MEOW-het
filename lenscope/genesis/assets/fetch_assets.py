#!/usr/bin/env python3
"""Download the optional asset library of the second-generation scenes, pinned to manifest.json.

Every CC0 / public-domain file listed in the manifest is fetched from its provider and checked against
the recorded size:
  - ambientCG materials (CC0): one 1K-JPG archive per material,
    https://ambientcg.com/get?file=<id>_1K-JPG.zip, extracted into pbr/<class>/<id>/;
  - The Metropolitan Museum of Art Open Access images (public domain): the object's primaryImageSmall
    (primaryImage when no small image exists);
  - Poly Haven HDRIs (CC0): the 1k .hdr file. The scene builder does not read them (it uses a procedural
    sky); they are fetched only for completeness of the library.
The per-material statistics caches are not part of the manifest; run compute_set_stats.py after the
download to write them. Files already present with the recorded size are skipped.

The exit status is non-zero if any file is missing or has a different size (a provider may re-encode an
image); such files are listed at the end.

Usage:
  python lenscope/genesis/assets/fetch_assets.py --out /path/to/genesis-assets [--jobs 8] [--only pbr]
Then point the scene builder at the library with GENESIS_ASSETS=/path/to/genesis-assets.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
import urllib.request
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
UA = {"User-Agent": "Mozilla/5.0 genesis-asset-fetch/1.1"}


def _get(url, timeout=300, tries=4):
    err = None
    for k in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r:
                return r.read()
        except Exception as e:  # network errors are retried with a growing delay
            err = e
            time.sleep(2.0 * (k + 1))
    raise RuntimeError(f"{url}: {err}")


def _have(out, entry):
    p = out / entry["path"]
    return p.is_file() and p.stat().st_size == entry["bytes"]


def fetch_ambientcg(out, asset_dir, entries):
    """One material: download the 1K-JPG archive and extract the files listed in the manifest."""
    if all(_have(out, e) for e in entries):
        return []
    asset_id = Path(asset_dir).name
    data = _get(f"https://ambientcg.com/get?file={asset_id}_1K-JPG.zip")
    wanted = {Path(e["path"]).name: e for e in entries}
    (out / asset_dir).mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        for name in z.namelist():
            if name in wanted:
                (out / asset_dir / name).write_bytes(z.read(name))
    return [e["path"] for e in entries if not _have(out, e)]


def fetch_met(out, entry):
    if _have(out, entry):
        return []
    obj = json.loads(_get(entry["source_url"], timeout=90))
    url = obj.get("primaryImageSmall") or obj.get("primaryImage")
    if not url:
        return [entry["path"]]
    p = out / entry["path"]
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(_get(url))
    return [] if _have(out, entry) else [entry["path"]]


def fetch_polyhaven(out, entry):
    if _have(out, entry):
        return []
    slug = entry["source_url"].rstrip("/").split("/")[-1]
    files = json.loads(_get(f"https://api.polyhaven.com/files/{slug}", timeout=90))
    p = out / entry["path"]
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(_get(files["hdri"]["1k"]["hdr"]["url"]))
    return [] if _have(out, entry) else [entry["path"]]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True, help="library directory (created if missing)")
    ap.add_argument("--manifest", default=str(HERE / "manifest.json"))
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--only", choices=["pbr", "decals", "hdri"], default=None,
                    help="fetch only one top-level folder of the library")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    files = json.loads(Path(a.manifest).read_text())["files"]
    acg, tasks = defaultdict(list), []
    for e in files:
        if a.only and not e["path"].startswith(a.only + "/"):
            continue
        prov = e["provider"]
        if prov == "ambientCG":
            acg[str(Path(e["path"]).parent)].append(e)
        elif prov.startswith("The Metropolitan Museum"):
            tasks.append((fetch_met, (out, e)))
        elif prov == "Poly Haven":
            tasks.append((fetch_polyhaven, (out, e)))
    tasks += [(fetch_ambientcg, (out, d, es)) for d, es in sorted(acg.items())]
    n_files = sum(len(es) for es in acg.values()) + sum(1 for f, _ in tasks if f is not fetch_ambientcg)
    print(f"[assets] {n_files} files ({len(acg)} materials) -> {out}", flush=True)

    bad = []

    def run(task):
        fn, args = task
        try:
            return fn(*args)
        except Exception as e:
            print(f"[assets] failed: {e}", flush=True)
            return [args[1] if isinstance(args[1], str) else args[1]["path"]]

    with ThreadPoolExecutor(max(1, a.jobs)) as ex:
        for i, miss in enumerate(ex.map(run, tasks), 1):
            bad += miss
            if i % 25 == 0 or i == len(tasks):
                print(f"[assets] {i}/{len(tasks)} items done, {len(bad)} problems", flush=True)

    if bad:
        print("[assets] missing or size mismatch:", *sorted(bad), sep="\n  ", flush=True)
        sys.exit(1)
    print("[assets] all files present with the recorded sizes", flush=True)


if __name__ == "__main__":
    main()
