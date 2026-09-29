#!/usr/bin/env python3
"""Keep a connected subset of a sampled pose graph (breadth-first from the first pose) for a quick check."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict, deque
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--count", type=int, default=4)
    args = parser.parse_args()

    data = json.loads(args.source.read_text())
    by_id = {int(p["id"]): p for p in data["poses"]}
    adjacency: dict[int, list[int]] = defaultdict(list)
    for edge in data["edges"]:
        a, b = int(edge[0]), int(edge[1])
        adjacency[a].append(b)
        adjacency[b].append(a)

    start = int(data["poses"][0]["id"])
    queue = deque([start])
    selected = []
    seen = {start}
    while queue and len(selected) < args.count:
        node = queue.popleft()
        selected.append(node)
        for neighbour in sorted(adjacency[node]):
            if neighbour not in seen:
                seen.add(neighbour)
                queue.append(neighbour)
    if len(selected) != args.count:
        raise RuntimeError(f"only found {len(selected)} connected poses")

    selected_set = set(selected)
    data["poses"] = [by_id[node] for node in selected]
    data["edges"] = [
        edge for edge in data["edges"]
        if int(edge[0]) in selected_set and int(edge[1]) in selected_set
    ]
    data["smoke_subset"] = {
        "source": str(args.source),
        "selected_pose_ids": selected,
    }
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    args.destination.write_text(json.dumps(data, indent=2))
    print(json.dumps(data["smoke_subset"]))


if __name__ == "__main__":
    main()
