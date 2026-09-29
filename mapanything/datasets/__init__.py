# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.
#
# Modified by the MEOW authors (2026): registers the first-generation scene dataset, uses spawned
# loader workers for the online camera sampler and passes batch-size and tuple-size schedule
# overrides.

"""
MapAnything Datasets
"""

import torch

from mapanything.datasets.procthor_unicol import ProcThorUnicol  # noqa
from mapanything.datasets.wai.aerialmegadepth import AerialMegaDepthWAI  # noqa
from mapanything.datasets.wai.ase import ASEWAI  # noqa
from mapanything.datasets.wai.blendedmvs import BlendedMVSWAI  # noqa
from mapanything.datasets.wai.dl3dv import DL3DVWAI  # noqa
from mapanything.datasets.wai.dynamicreplica import DynamicReplicaWAI  # noqa
from mapanything.datasets.wai.eth3d import ETH3DWAI  # noqa
from mapanything.datasets.wai.megadepth import MegaDepthWAI  # noqa
from mapanything.datasets.wai.mpsd import MPSDWAI  # noqa
from mapanything.datasets.wai.mvs_synth import MVSSynthWAI  # noqa
from mapanything.datasets.wai.paralleldomain4d import ParallelDomain4DWAI  # noqa
from mapanything.datasets.wai.sailvos3d import SAILVOS3DWAI  # noqa
from mapanything.datasets.wai.scannetpp import ScanNetPPWAI  # noqa
from mapanything.datasets.wai.spring import SpringWAI  # noqa
from mapanything.datasets.wai.tav2_wb import TartanAirV2WBWAI  # noqa
from mapanything.datasets.wai.unrealstereo4k import UnrealStereo4KWAI  # noqa
from mapanything.utils.train_tools import get_rank, get_world_size


def get_test_data_loader(
    dataset, batch_size, num_workers=8, shuffle=False, drop_last=False, pin_mem=True
):
    "Get simple PyTorch dataloader corresponding to the testing dataset"
    # PyTorch dataset
    if isinstance(dataset, str):
        dataset = eval(dataset)

    world_size = get_world_size()
    rank = get_rank()

    if torch.distributed.is_initialized():
        sampler = torch.utils.data.DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
            drop_last=drop_last,
        )
    elif shuffle:
        sampler = torch.utils.data.RandomSampler(dataset)
    else:
        sampler = torch.utils.data.SequentialSampler(dataset)

    mp_ctx = None
    if num_workers > 0 and _dataset_has_camera_sampling(dataset):
        # The camera sampler renders on the GPU inside the workers, which requires spawned workers
        mp_ctx = "spawn"
        print(f"Test dataloader: multiprocessing_context=spawn (num_workers={num_workers})", flush=True)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_mem,
        drop_last=drop_last,
        multiprocessing_context=mp_ctx,
    )

    return data_loader


def get_test_many_ar_data_loader(
    dataset, batch_size, num_workers=8, drop_last=False, pin_mem=True
):
    "Get PyTorch dataloader corresponding to the testing dataset that supports many aspect ratios"
    # PyTorch dataset
    if isinstance(dataset, str):
        dataset = eval(dataset)

    world_size = get_world_size()
    rank = get_rank()

    # Get BatchedMultiFeatureRandomSampler
    sampler = dataset.make_sampler(
        batch_size,
        shuffle=True,
        world_size=world_size,
        rank=rank,
        drop_last=drop_last,
        use_dynamic_sampler=False,
    )

    # Init the data laoder
    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_mem,
        drop_last=drop_last,
    )

    return data_loader


class DynamicBatchDatasetWrapper:
    """
    Wrapper dataset that handles DynamicBatchedMultiFeatureRandomSampler output.

    The dynamic sampler returns batches (lists of tuples) instead of individual samples.
    This wrapper ensures that the underlying dataset's __getitem__ method gets called
    with individual tuples as expected.
    """

    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, batch_indices):
        """
        Handle batch of indices from DynamicBatchedMultiFeatureRandomSampler.

        Args:
            batch_indices: List of tuples like [(sample_idx, feat_idx_1, feat_idx_2, ...), ...]

        Returns:
            List of samples from the underlying dataset
        """
        if isinstance(batch_indices, (list, tuple)) and len(batch_indices) > 0:
            # If it's a batch (list of tuples), process each item
            if isinstance(batch_indices[0], (list, tuple)):
                return [self.dataset[idx] for idx in batch_indices]
            else:
                # Single tuple, call dataset directly
                return self.dataset[batch_indices]
        else:
            # Fallback for single index
            return self.dataset[batch_indices]

    def __len__(self):
        return len(self.dataset)

    def __getattr__(self, name):
        # Delegate all other attributes to the wrapped dataset. While a spawned worker unpickles
        # the wrapper, __dict__ is still empty; raise AttributeError for "dataset" itself instead
        # of recursing.
        if name == "dataset":
            raise AttributeError(name)
        return getattr(self.dataset, name)


def _dataset_has_camera_sampling(ds, depth=0):
    """Return True if a dataset with the online camera sampler occurs anywhere in a (possibly
    wrapped or concatenated) dataset graph. Such datasets need spawned dataloader workers because
    the sampler uses CUDA inside the workers; all other datasets keep forked workers."""
    if depth > 4:
        return False
    if getattr(ds, "camera_sampling", False):
        return True
    for attr in ("datasets", "dataset"):
        sub = getattr(ds, attr, None)
        if isinstance(sub, (list, tuple)):
            if any(_dataset_has_camera_sampling(x, depth + 1) for x in sub):
                return True
        elif sub is not None and sub is not ds:
            if _dataset_has_camera_sampling(sub, depth + 1):
                return True
    return False


def get_train_data_loader(
    dataset,
    max_num_of_imgs_per_gpu,
    num_workers=8,
    shuffle=True,
    drop_last=True,
    pin_mem=True,
    feature_to_batch_size_map_override=None,
    feature_weights_schedule=None,
):
    "Dynamic PyTorch dataloader corresponding to the training dataset"
    # PyTorch dataset
    if isinstance(dataset, str):
        dataset = eval(dataset)

    world_size = get_world_size()
    rank = get_rank()

    # Get DynamicBatchedMultiFeatureRandomSampler
    batch_sampler = dataset.make_sampler(
        shuffle=shuffle,
        world_size=world_size,
        rank=rank,
        drop_last=drop_last,
        max_num_of_images_per_gpu=max_num_of_imgs_per_gpu,
        use_dynamic_sampler=True,
        feature_to_batch_size_map_override=feature_to_batch_size_map_override,
        feature_weights_schedule=feature_weights_schedule,
    )

    # Wrap the dataset to handle batch format from dynamic sampler
    wrapped_dataset = DynamicBatchDatasetWrapper(dataset)

    mp_ctx, persistent = None, False
    if num_workers > 0 and _dataset_has_camera_sampling(dataset):
        # Workers must not be persistent: set_epoch() updates the main-process copy of the dataset
        # every epoch, and spawned workers pick up that state only when they are re-created.
        mp_ctx, persistent = "spawn", False
        print(
            f"Train dataloader: multiprocessing_context=spawn, persistent_workers=False "
            f"(num_workers={num_workers})",
            flush=True,
        )

    # Init the dynamic data loader
    data_loader = torch.utils.data.DataLoader(
        wrapped_dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        pin_memory=pin_mem,
        multiprocessing_context=mp_ctx,
        persistent_workers=persistent,
    )

    return data_loader
