#!/usr/bin/env python3
"""Inventory the optional second-generation asset bundle with source and license."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def classify(root: Path, path: Path) -> dict[str, str] | None:
    """Provider, license and source of a third-party asset file; None for any other file."""
    rel = path.relative_to(root)
    parts = rel.parts
    if parts[0] == "pbr" and len(parts) >= 3 and not path.name.startswith("."):
        asset_id = parts[2]
        return {
            "provider": "ambientCG",
            "license": "CC0 1.0",
            "source_url": f"https://ambientcg.com/view?id={asset_id}",
        }
    if parts[0] == "hdri" and path.suffix.lower() == ".hdr":
        slug = path.stem.removesuffix("_1k")
        return {
            "provider": "Poly Haven",
            "license": "CC0 1.0",
            "source_url": f"https://polyhaven.com/a/{slug}",
        }
    if parts[0] == "decals" and path.stem.startswith("met_"):
        object_id = path.stem.split("_", 1)[1]
        return {
            "provider": "The Metropolitan Museum of Art Open Access",
            "license": "Public Domain / CC0",
            "source_url": (
                "https://collectionapi.metmuseum.org/public/collection/v1/objects/"
                + object_id
            ),
        }
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    files = []
    counts: Counter[str] = Counter()
    for path in sorted(p for p in args.assets_root.rglob("*") if p.is_file()):
        source = classify(args.assets_root, path)
        if source is None:  # statistics caches, placeholders and other non-asset files
            continue
        counts[source["provider"]] += 1
        files.append(
            {
                "path": path.relative_to(args.assets_root).as_posix(),
                "bytes": path.stat().st_size,
                **source,
            }
        )

    payload = {
        "schema": "meow-genesis-assets-v1",
        "file_count": len(files),
        "provider_file_counts": dict(sorted(counts.items())),
        "notes": [
            "The bundle is optional; a missing bundle selects the pure-procedural fallback.",
            "Counts include all downloaded supporting formats, not just files read by Blender.",
        ],
        "files": files,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({key: payload[key] for key in payload if key != "files"}, indent=2))


if __name__ == "__main__":
    main()
