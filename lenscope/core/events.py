"""Structured event log of the pose-graph sampler: JSON lines on stderr with the prefix "AGEN " (for
lenscope, as in the Blender scripts of the engine), optionally appended to a file."""
import json
import sys
import time
from pathlib import Path

_SINK = None


def open_events(path: Path):
    global _SINK
    path.parent.mkdir(parents=True, exist_ok=True)
    _SINK = open(path, "a")


def log(event: str, **kw):
    rec = {"ts": round(time.time(), 3), "event": event, **kw}
    line = json.dumps(rec, ensure_ascii=False, default=str)
    print("AGEN " + line, file=sys.stderr, flush=True)
    if _SINK is not None:
        _SINK.write(line + "\n")
        _SINK.flush()
