#!/usr/bin/env python
"""Single-GPU MapAnything backbone inference latency (unoptimized baseline).

Builds MapAnything from config, loads a checkpoint (non-strict), feeds N (default 4)
random views at 518x518 that are pre-built on the device, and reports:
  * end-to-end wall time per forward, measured on the host (includes Python overhead)
  * pure GPU forward time via CUDA events
  * peak GPU memory
under fp32 and bf16 autocast. Warmup iterations are excluded. No torch.compile, no
TensorRT, no attention-kernel override: the out-of-the-box latency on one GPU.

--meow builds the MEOW network instead (aspect-ratio encoder and panorama wrap, as
scripts/meow_model.py builds it for the evaluators) and flags every view as a panorama,
so the wrap runs on every view.

Usage:
  python scripts/bench_latency.py --ckpt <checkpoint.pth> \
    --num-views 4 --res 518 --iters 30 --warmup 8 [--meow]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

from mapanything.models import init_model_from_config


def build_views(n, res, device, norm="dinov2", meow=False):
    """N synthetic views: random DINOv2-normalised RGB at res x res."""
    views = []
    for _ in range(n):
        img = torch.randn(1, 3, res, res, device=device)
        v = {
            "img": img,
            "true_shape": torch.tensor([[res, res]], device=device),
            "data_norm_type": [norm],
        }
        if meow:
            v["aspect_ratio"] = torch.tensor([1.0], device=device)
            v["pano_wrap"] = True
        views.append(v)
    return views


@torch.no_grad()
def timed(model, views, iters, warmup, use_bf16):
    dev = views[0]["img"].device
    amp = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_bf16 else _null()
    # warmup
    for _ in range(warmup):
        with amp:
            _ = model(views)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    # timed: host wall + cuda-event GPU time
    host = []
    gpu = []
    for _ in range(iters):
        if dev.type == "cuda":
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
        t0 = time.perf_counter()
        if dev.type == "cuda":
            s.record()
        with amp:
            _ = model(views)
        if dev.type == "cuda":
            e.record()
            torch.cuda.synchronize()
            gpu.append(s.elapsed_time(e))  # ms
        host.append((time.perf_counter() - t0) * 1000.0)
    return np.array(host), np.array(gpu)


class _null:
    def __enter__(self): return self
    def __exit__(self, *a): return False


def stats(name, arr):
    if len(arr) == 0:
        return
    p = np.percentile(arr, [50, 90, 99])
    print(f"  {name:<18} mean={arr.mean():7.1f}ms  p50={p[0]:7.1f}  "
          f"p90={p[1]:7.1f}  p99={p[2]:7.1f}  min={arr.min():7.1f}  "
          f"fps={1000.0/arr.mean():5.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--num-views", type=int, default=4)
    ap.add_argument("--res", type=int, default=518)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--meow", action="store_true",
                    help="build the MEOW network (aspect-ratio encoder, panorama wrap on every view)")
    args = ap.parse_args()

    dev = torch.device(args.device)
    print(f"[gpu] {torch.cuda.get_device_name(dev) if dev.type=='cuda' else 'cpu'}")
    print(f"[cfg] num_views={args.num_views} res={args.res} iters={args.iters} warmup={args.warmup}")

    c = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = c.get("model", c)
    if args.meow:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from meow_model import build_model_with_ar, load_ckpt_inplace

        print("[model] building the MEOW network (aspect-ratio encoder, panorama wrap)")
        model = build_model_with_ar(args.device, "wrap")
        n_missing = load_ckpt_inplace(model, sd)
        print(f"[ckpt] loaded {args.ckpt} (missing={n_missing}, aspect-ratio encoder only)")
    else:
        print("[model] building MapAnything from config")
        model = init_model_from_config("mapanything", device=args.device)
        model.eval()
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"[ckpt] loaded {args.ckpt} (missing={len(missing)}, unexpected={len(unexpected)})")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] params={n_params/1e6:.1f}M")

    views = build_views(args.num_views, args.res, dev, meow=args.meow)

    for use_bf16 in (False, True):
        tag = "bf16-autocast" if use_bf16 else "fp32"
        print(f"\n=== precision: {tag} ===")
        try:
            host, gpu = timed(model, views, args.iters, args.warmup, use_bf16)
            stats("end2end(host)", host)
            stats("gpu-forward", gpu)
            if dev.type == "cuda":
                print(f"  peak_mem={torch.cuda.max_memory_allocated()/1e9:.2f}GB")
                torch.cuda.reset_peak_memory_stats()
        except Exception as exc:  # noqa: BLE001
            print(f"  FAILED: {exc}")

    print("\n[note] unoptimized baseline: no torch.compile / TRT / fused attn override, "
          "single GPU, batch=1, random inputs (compute-representative).")


if __name__ == "__main__":
    main()
