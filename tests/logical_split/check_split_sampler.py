#!/usr/bin/env python3
"""Check the logical-split loaders of a training configuration against the loaders of the recipe.

Builds the training dataset of the composed configuration, draws the batches of every logical
rank with the regular training loader of the logical world and the batches of every physical
rank with the logical-split loader, and checks at every step of the requested epochs that the
shares of the physical ranks of a logical rank are disjoint, together form the logical batch,
have the expected sizes, and that all tuples of a logical batch share their features (aspect-ratio
bucket and number of views). Indices only: no data is loaded and no GPU is used.

Usage, from the repository root with PYTHONPATH set as for training:
    python tests/logical_split/check_split_sampler.py --world-size 8 --epochs 0 1 15 -- \
        +meow_stage=stage2 +meow_hardware=a100x8 \
        dataset.procthor_unicol.train.ROOT=<scenes> \
        dataset.procthor_unicol.train.splits_dir=<splits>
"""

import argparse
import collections
import sys
from pathlib import Path
from unittest import mock

from hydra import compose, initialize_config_dir

import mapanything.datasets as datasets
from mapanything.train.logical_split import get_logical_split_train_data_loader

LOADER_SETTINGS = (
    "num_workers",
    "collate_fn",
    "pin_memory",
    "multiprocessing_context",
    "persistent_workers",
    "prefetch_factor",
)


def recipe_loader(dataset, logical_rank, logical_world_size, **kwargs):
    """Regular training loader of one rank of the recipe's world."""
    with mock.patch.object(datasets, "get_rank", lambda: logical_rank), mock.patch.object(
        datasets, "get_world_size", lambda: logical_world_size
    ):
        return datasets.get_train_data_loader(dataset, **kwargs)


def batches(loader, epoch):
    loader.batch_sampler.set_epoch(epoch)
    return [[tuple(int(x) for x in item) for item in batch] for batch in loader.batch_sampler]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--world-size", type=int, required=True, help="physical ranks (GPUs)")
    parser.add_argument(
        "--logical-world-size",
        type=int,
        help="ranks of the recipe (default: train_params.logical_world_size)",
    )
    parser.add_argument("--epochs", type=int, nargs="+", default=[0])
    parser.add_argument("overrides", nargs="*", help="Hydra overrides as for training")
    opts = parser.parse_args()

    config_dir = Path(__file__).resolve().parents[2] / "configs"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(config_name="train", overrides=opts.overrides)
    train_params = cfg.train_params
    logical_world_size = opts.logical_world_size or train_params.get("logical_world_size")
    if logical_world_size is None:
        sys.exit("set train_params.logical_world_size or --logical-world-size")
    world_size = opts.world_size
    num_shards = world_size // logical_world_size

    # Same conversions as mapanything.train.training.train
    batch_size_map = train_params.get("feature_to_batch_size_map_override")
    if batch_size_map:
        batch_size_map = {int(k): int(v) for k, v in batch_size_map.items()}
    schedule = train_params.get("feature_weights_schedule")
    if schedule:
        schedule = [
            {
                "epoch_start": int(tier["epoch_start"]),
                "weights": {int(k): float(v) for k, v in tier["weights"].items()},
            }
            for tier in schedule
        ]
    kwargs = dict(
        max_num_of_imgs_per_gpu=train_params.max_num_of_imgs_per_gpu,
        feature_to_batch_size_map_override=batch_size_map or None,
        feature_weights_schedule=schedule or None,
    )

    print(f"Train dataset: {cfg.dataset.train_dataset}")
    dataset = eval(cfg.dataset.train_dataset, vars(datasets))

    # The logical-split loader keeps the worker settings of the regular loader
    reference = recipe_loader(
        dataset, 0, logical_world_size, num_workers=cfg.dataset.num_workers, **kwargs
    )
    split = get_logical_split_train_data_loader(
        dataset,
        logical_world_size=logical_world_size,
        rank=0,
        world_size=world_size,
        num_workers=cfg.dataset.num_workers,
        **kwargs,
    )
    failures = [
        f"loader setting {name}: {getattr(split, name)!r} != {getattr(reference, name)!r}"
        for name in LOADER_SETTINGS
        if getattr(split, name) != getattr(reference, name)
    ]

    references = [
        recipe_loader(dataset, rank, logical_world_size, num_workers=0, **kwargs)
        for rank in range(logical_world_size)
    ]
    splits = [
        get_logical_split_train_data_loader(
            dataset,
            logical_world_size=logical_world_size,
            rank=rank,
            world_size=world_size,
            num_workers=0,
            **kwargs,
        )
        for rank in range(world_size)
    ]
    batch_sampler = splits[0].batch_sampler
    print(f"Batch size by number of views: {batch_sampler.batch_size_by_num_views}")
    for rank, loader in enumerate(splits):
        recipe_steps = len(references[rank // num_shards])
        if len(loader) != recipe_steps:
            failures.append(f"rank {rank}: {len(loader)} steps per epoch, recipe {recipe_steps}")

    for epoch in opts.epochs:
        recipe_batches = [batches(loader, epoch) for loader in references]
        split_batches = [batches(loader, epoch) for loader in splits]
        steps, tuples, k_hist = 0, 0, collections.Counter()
        for logical_rank, logical in enumerate(recipe_batches):
            group = split_batches[logical_rank * num_shards : (logical_rank + 1) * num_shards]
            if any(len(shares) != len(logical) for shares in group):
                failures.append(f"epoch {epoch} logical rank {logical_rank}: step counts differ")
                continue
            for step, logical_batch in enumerate(logical):
                shares = [shares[step] for shares in group]
                features = {item[1:] for item in logical_batch}
                # Index tuples are (sample, aspect-ratio bucket[, number-of-views index])
                if isinstance(dataset.num_views, int):
                    num_views = dataset.num_views
                else:
                    num_views = dataset.num_views[logical_batch[0][2]]
                where = f"epoch {epoch} logical rank {logical_rank} step {step}"
                if len(features) != 1:
                    failures.append(f"{where}: features differ within the batch {features}")
                if len(logical_batch) != batch_sampler.logical_batch_size(num_views):
                    failures.append(f"{where}: {len(logical_batch)} tuples for K={num_views}")
                flat = [item for share in shares for item in share]
                if sorted(flat) != sorted(logical_batch) or len(set(flat)) != len(flat):
                    failures.append(f"{where}: shares do not partition the logical batch")
                for shard, share in enumerate(shares):
                    if share != logical_batch[shard::num_shards]:
                        failures.append(f"{where}: share {shard} != tuples {shard}::{num_shards}")
                    if len(share) != len(range(shard, len(logical_batch), num_shards)):
                        failures.append(f"{where}: share {shard} has {len(share)} tuples")
                steps += 1
                tuples += len(logical_batch)
                k_hist[num_views] += 1
        print(
            f"epoch {epoch}: {steps} logical batches ({tuples} tuples) checked on "
            f"{logical_world_size} logical / {world_size} physical ranks; "
            f"batches by K: {dict(sorted(k_hist.items()))}"
        )

    for failure in failures[:20]:
        print("FAIL", failure)
    print("PASS" if not failures else f"FAIL ({len(failures)} problems)")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
