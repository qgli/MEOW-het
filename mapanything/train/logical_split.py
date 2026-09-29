"""
Logical-split training: run a data-parallel recipe on GPUs that cannot hold its per-GPU batch.

A data-parallel recipe is defined by its sampling world: W_l ranks, each drawing batches from its
own sampler, whose batch size b_K depends on the number of views K. Every optimiser step averages
the gradients of the W_l batch means. Logical-split mode keeps this sampling world on
W_p = s * W_l processes (s >= 1 an integer): physical rank p builds the sampler of logical rank
p // s and takes every s-th tuple, starting at p % s, of each of its batches. Every tuple is a
micro-batch of its own. Gradients are accumulated locally under DDP no_sync and all-reduced once
per logical batch, and each tuple's loss is scaled by s / b_K, so the average over W_p processes
equals the recipe's average over W_l batch means. The learning-rate schedule advances once per
logical batch, and an epoch has as many steps as in the recipe.

Use it when the per-GPU batch of a recipe does not fit in memory: the recipe of W_l GPUs then
runs on s * W_l GPUs with the memory of one tuple per forward pass, and every optimiser step draws
the same tuples (sample index, aspect-ratio bucket and number of views) as the corresponding step
of the recipe. With s = 1 each GPU keeps a whole logical batch. Enable it with
train_params.logical_world_size (= W_l), as the hardware profiles in configs/meow_hardware do for
GPUs with 40 GB; the number of processes must be a multiple of it.

Known deviation: loss terms that average over all valid pixels of a batch average over the pixels
of one tuple instead, so every tuple has the same weight rather than a weight proportional to its
number of valid pixels, and gradients differ from those of a whole-batch step. The loss of every
tuple is exact (the same as in a batch of its own). The loss-instability check applies to each
tuple's loss instead of the batch loss.
"""

import contextlib
import math
import os
import pickle
import sys
from collections import defaultdict
from collections.abc import Mapping
from typing import Sized

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

import mapanything.datasets as datasets
import mapanything.utils.train_tools as train_tools
from mapanything.utils.inference import loss_of_one_batch_multi_view


def split_factor(logical_world_size, world_size):
    """
    Number of physical ranks per logical rank.

    Args:
        logical_world_size (int): Number of data-parallel ranks of the recipe.
        world_size (int): Number of training processes.

    Returns:
        int: s = world_size / logical_world_size.
    """
    logical_world_size = int(logical_world_size)
    if logical_world_size < 1 or world_size % logical_world_size:
        raise ValueError(
            f"train_params.logical_world_size={logical_world_size} must divide the number of "
            f"training processes ({world_size})"
        )
    return world_size // logical_world_size


def _batch_size_by_num_views(batch_sampler, num_views):
    """Map every number of views K to the batch size b_K drawn by a dynamic batch sampler."""
    size_map = batch_sampler.feature_to_batch_size_map

    def batch_size(feature_value):
        # Same lookup as DynamicBatchedMultiFeatureRandomSampler.__iter__
        size = size_map(feature_value) if callable(size_map) else size_map.get(feature_value, 1)
        return max(1, int(size))

    pool = range(batch_sampler.pool_sizes[batch_sampler.scaling_feature_idx])
    if isinstance(num_views, int):
        # A fixed number of views makes the aspect-ratio bucket the batch-size feature
        pairs = [(num_views, batch_size(value)) for value in pool]
    else:
        pairs = [(int(num_views[value]), batch_size(value)) for value in pool]
    sizes = {}
    for k, size in pairs:
        if sizes.setdefault(k, size) != size:
            raise ValueError(
                "Logical-split training needs one batch size per number of views, but "
                f"K={k} is drawn with batch sizes {sizes[k]} and {size}"
            )
    return sizes


class LogicalSplitBatchSampler:
    """
    Batch sampler that yields one physical rank's share of every batch of a logical rank.

    Args:
        logical_batch_sampler: Dynamic batch sampler of the logical rank.
        shard_index (int): Index of this physical rank among those of its logical rank.
        num_shards (int): Physical ranks per logical rank (s).
        num_views (int or list): Number(s) of views of the dataset.
    """

    def __init__(self, logical_batch_sampler, shard_index, num_shards, num_views):
        self.logical_batch_sampler = logical_batch_sampler
        self.shard_index = int(shard_index)
        self.num_shards = int(num_shards)
        self.batch_size_by_num_views = _batch_size_by_num_views(
            logical_batch_sampler, num_views
        )
        too_small = {
            k: size
            for k, size in self.batch_size_by_num_views.items()
            if size < self.num_shards
        }
        if too_small:
            raise ValueError(
                f"Every logical batch needs at least {self.num_shards} tuples (one per physical "
                f"rank); batch sizes by number of views: {too_small}"
            )

    def __iter__(self):
        for logical_batch in self.logical_batch_sampler:
            local_batch = logical_batch[self.shard_index :: self.num_shards]
            if not local_batch:
                raise RuntimeError("Logical-split sampler produced an empty local batch")
            yield local_batch

    def __len__(self):
        return len(self.logical_batch_sampler)

    def set_epoch(self, epoch):
        self.logical_batch_sampler.set_epoch(epoch)

    def logical_batch_size(self, num_views):
        """Size of the logical batch that a local batch with num_views views belongs to."""
        return self.batch_size_by_num_views[int(num_views)]

    def local_batch_size(self, num_views):
        """Number of tuples of that logical batch assigned to this physical rank."""
        return len(range(self.shard_index, self.logical_batch_size(num_views), self.num_shards))


@contextlib.contextmanager
def _sampling_world(rank, world_size):
    """Make mapanything.datasets build samplers for rank `rank` of `world_size` ranks.

    get_train_data_loader reads the rank and the world size through the module-level functions
    get_rank and get_world_size, which are replaced while the context is active.
    """
    saved = datasets.get_rank, datasets.get_world_size
    datasets.get_rank, datasets.get_world_size = (lambda: rank), (lambda: world_size)
    try:
        yield
    finally:
        datasets.get_rank, datasets.get_world_size = saved


def get_logical_split_train_data_loader(
    dataset,
    max_num_of_imgs_per_gpu,
    logical_world_size,
    num_workers=8,
    shuffle=True,
    drop_last=True,
    pin_mem=True,
    feature_to_batch_size_map_override=None,
    feature_weights_schedule=None,
    rank=None,
    world_size=None,
):
    """
    Training data loader of one physical rank in logical-split mode.

    Builds the loader of mapanything.datasets.get_train_data_loader for logical rank rank // s
    in a world of logical_world_size ranks, and replaces its batch sampler with a
    LogicalSplitBatchSampler; all other loader settings (workers, multiprocessing context,
    pinning) are kept.

    Args:
        dataset, max_num_of_imgs_per_gpu, num_workers, shuffle, drop_last, pin_mem,
        feature_to_batch_size_map_override, feature_weights_schedule:
            As in mapanything.datasets.get_train_data_loader; max_num_of_imgs_per_gpu and the
            batch-size map describe the batches of a logical rank.
        logical_world_size (int): Number of data-parallel ranks of the recipe.
        rank (int, optional): Physical rank. Defaults to the rank of this process.
        world_size (int, optional): Number of physical ranks. Defaults to the process group size.

    Returns:
        DataLoader: Loader whose items are this rank's share of each logical batch.
    """
    if not drop_last:
        raise ValueError("Logical-split training requires drop_last=True")
    rank = train_tools.get_rank() if rank is None else int(rank)
    world_size = train_tools.get_world_size() if world_size is None else int(world_size)
    num_shards = split_factor(logical_world_size, world_size)
    logical_world_size = world_size // num_shards
    logical_rank = rank // num_shards
    with _sampling_world(logical_rank, logical_world_size):
        logical_loader = datasets.get_train_data_loader(
            dataset=dataset,
            max_num_of_imgs_per_gpu=max_num_of_imgs_per_gpu,
            num_workers=num_workers,
            shuffle=shuffle,
            drop_last=drop_last,
            pin_mem=pin_mem,
            feature_to_batch_size_map_override=feature_to_batch_size_map_override,
            feature_weights_schedule=feature_weights_schedule,
        )
    batch_sampler = LogicalSplitBatchSampler(
        logical_loader.batch_sampler,
        shard_index=rank % num_shards,
        num_shards=num_shards,
        num_views=logical_loader.dataset.num_views,
    )
    print(
        f"Logical-split loader: rank {rank} of {world_size} takes tuples {rank % num_shards}::"
        f"{num_shards} of logical rank {logical_rank} of {logical_world_size}, "
        f"{len(batch_sampler)} steps per epoch",
        flush=True,
    )
    return DataLoader(
        logical_loader.dataset,
        batch_sampler=batch_sampler,
        num_workers=logical_loader.num_workers,
        collate_fn=logical_loader.collate_fn,
        pin_memory=logical_loader.pin_memory,
        timeout=logical_loader.timeout,
        worker_init_fn=logical_loader.worker_init_fn,
        multiprocessing_context=logical_loader.multiprocessing_context,
        generator=logical_loader.generator,
        prefetch_factor=logical_loader.prefetch_factor,
        persistent_workers=logical_loader.persistent_workers,
    )


def _select_sample(value, index, batch_size):
    """Entry `index` of a collated value, keeping a leading batch dimension of one."""
    if isinstance(value, (torch.Tensor, np.ndarray)):
        if value.ndim > 0 and value.shape[0] == batch_size:
            return value[index : index + 1]
        return value
    if isinstance(value, Mapping):
        return {key: _select_sample(item, index, batch_size) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        if len(value) == batch_size and all(isinstance(item, (str, bytes)) for item in value):
            # Collation keeps strings as a list with one entry per sample
            return type(value)([value[index]])
        # Collation turns a sequence field into a sequence of batched entries
        return type(value)(_select_sample(item, index, batch_size) for item in value)
    return value


def split_into_samples(batch):
    """
    Split a collated multi-view batch into batches of one tuple.

    Args:
        batch (list): Views as collated by the training data loader.

    Yields:
        list: The views of one tuple, each value with a leading batch dimension of one.
    """
    batch_size = int(batch[0]["img"].shape[0])
    for index in range(batch_size):
        yield [_select_sample(view, index, batch_size) for view in batch]


def _save_debug_material_and_exit(
    views, loss_value, loss_details, epoch, data_iter_step, micro_step, args, model_without_ddp
):
    """Report an unstable tuple loss, save the tuple and the model, and stop training."""
    print("Loss is {}, stopping training".format(loss_value), force=True)
    print(f"Loss Details: {loss_details}", force=True)
    print(
        f"Epoch: {epoch}, Data Iteration: {data_iter_step}, Tuple: {micro_step}", force=True
    )
    # Save the current tuple to the output folder for further inspection
    for view_idx, view in enumerate(views):
        view_cpu = {}
        for k, v in view.items():
            view_cpu[k] = v.cpu() if isinstance(v, torch.Tensor) else v
        with open(os.path.join(args.output_dir, f"batch_view_{view_idx}.pkl"), "wb") as f:
            pickle.dump(view_cpu, f)
    # Save the model to the output folder for further inspection
    checkpoint_debug_path = os.path.join(args.output_dir, "checkpoint-debug.pth")
    to_save_debug = {
        "args": args,
        "model": (
            model_without_ddp
            if isinstance(model_without_ddp, dict)
            else model_without_ddp.cpu().state_dict()
        ),
        "epoch": epoch,
        "data_iter_step": data_iter_step,
    }
    torch.save(to_save_debug, checkpoint_debug_path)
    print(f"Saved debugging material to {args.output_dir}", force=True)
    sys.exit(1)


def train_one_epoch_logical_split(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Sized,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    args,
    log_writer=None,
    param_groups_name_to_idx_map=None,
    param_groups_idx_to_name_map=None,
    model_without_ddp=None,
):
    """
    Trains the model for one epoch in logical-split mode.

    Same arguments and return value as mapanything.train.training.train_one_epoch. Each loader
    item is this rank's share of one logical batch and makes one optimiser step: its tuples are
    processed one at a time, and only the backward pass of the last one all-reduces the
    accumulated gradients.

    Returns:
        dict: Dictionary containing training metrics averaged over the epoch.
    """
    batch_sampler = data_loader.batch_sampler
    if not isinstance(batch_sampler, LogicalSplitBatchSampler):
        raise TypeError(
            "Logical-split training needs the loader of get_logical_split_train_data_loader"
        )
    if args.train_params.accum_iter != 1:
        raise ValueError(
            "Logical-split training accumulates the gradients of a logical batch; "
            "set train_params.accum_iter=1"
        )
    num_shards = batch_sampler.num_shards
    if isinstance(model, DistributedDataParallel):
        no_sync = model.no_sync
    else:
        no_sync = contextlib.nullcontext

    model.train(True)
    metric_logger = train_tools.MetricLogger(delimiter="  ")
    for submodule_name in param_groups_name_to_idx_map:
        lr_name = f"lr_{submodule_name}" if submodule_name != "default" else "lr"
        metric_logger.add_meter(
            lr_name, train_tools.SmoothedValue(window_size=1, fmt="{value:.6f}")
        )
    header = "Epoch: [{}]".format(epoch)

    if log_writer is not None:
        print("log_dir: {}".format(log_writer.log_dir))

    if hasattr(data_loader.dataset, "set_epoch"):
        data_loader.dataset.set_epoch(epoch)
    batch_sampler.set_epoch(epoch)

    optimizer.zero_grad()

    for data_iter_step, batch in enumerate(
        metric_logger.log_every(data_loader, args.train_params.print_freq, header)
    ):
        n_views = len(batch)
        local_batch_size = int(batch[0]["img"].shape[0])
        logical_batch_size = batch_sampler.logical_batch_size(n_views)
        if local_batch_size != batch_sampler.local_batch_size(n_views):
            raise RuntimeError(
                f"Local batch of {local_batch_size} tuples with {n_views} views does not match "
                f"a logical batch of {logical_batch_size} tuples split over {num_shards} ranks"
            )
        # Per-tuple sampling log: ProcThorUnicol attaches one JSON record per tuple to view 0;
        # they are appended to <output_dir>/sampling_logs/rank<r>.jsonl with epoch and step.
        if isinstance(batch[0], dict) and "sampling_log" in batch[0]:
            try:
                _sl_dir = os.path.join(args.output_dir, "sampling_logs")
                os.makedirs(_sl_dir, exist_ok=True)
                with open(os.path.join(_sl_dir, f"rank{train_tools.get_rank()}.jsonl"), "a") as _slf:
                    for _s in batch[0]["sampling_log"]:
                        _slf.write('{"ep":%d,"it":%d,"rec":%s}\n' % (epoch, data_iter_step, _s))
            except Exception as _sle:
                if data_iter_step == 0:
                    print(f"[sampling-log] WARN: {type(_sle).__name__}: {_sle}", flush=True)
        epoch_f = epoch + data_iter_step / len(data_loader)

        # We use a per iteration (instead of per epoch) lr scheduler
        train_tools.adjust_learning_rate(
            optimizer,
            epoch_f,
            args.train_params,
            param_groups_idx_to_name_map,
            args.train_params.submodule_configs,
        )

        # One micro-batch per tuple; the forward and backward passes of all but the last one run
        # under no_sync, so the gradients are all-reduced once per logical batch.
        loss_values = []
        loss_details_values = defaultdict(list)
        sync_backward_count = 0
        gradient_norm = None
        for micro_step, micro_batch in enumerate(split_into_samples(batch)):
            is_last = micro_step == local_batch_size - 1
            with contextlib.nullcontext() if is_last else no_sync():
                loss_tuple = loss_of_one_batch_multi_view(
                    micro_batch,
                    model,
                    criterion,
                    device,
                    use_amp=bool(args.train_params.amp),
                    amp_dtype=args.train_params.amp_dtype,
                    ret="loss",
                )
                loss, loss_details = loss_tuple  # criterion returns two values
                if n_views > 2:
                    loss = loss * (
                        2 / n_views
                    )  # scale the loss relative to the number of views (base is 2 views)
                loss_value = float(loss)

                check_instability = not math.isfinite(loss_value) or (
                    args.train_params.check_loss_instability
                    and loss_value > args.train_params.max_loss_value
                )
                if check_instability:
                    _save_debug_material_and_exit(
                        micro_batch,
                        loss_value,
                        loss_details,
                        epoch,
                        data_iter_step,
                        micro_step,
                        args,
                        model_without_ddp,
                    )
                loss_values.append(loss_value)
                for name, value in loss_details.items():
                    loss_details_values[name].append(float(value))

                # DDP averages over s processes per logical rank: s / b_K turns the sum over the
                # tuples of a logical batch into the batch mean of the recipe. The gradients are
                # clipped to max norm 1 and applied at the last tuple.
                gradient_norm = loss_scaler(
                    loss * (num_shards / logical_batch_size),
                    optimizer,
                    parameters=model.parameters(),
                    update_grad=is_last,
                    clip_grad=1.0,
                )
                sync_backward_count += int(is_last)

            del loss
            del micro_batch

        if sync_backward_count != 1:
            raise RuntimeError(
                f"Expected one synchronizing backward pass per logical batch, got "
                f"{sync_backward_count}"
            )

        # Zero out the gradients to prepare for the next logical batch
        optimizer.zero_grad()

        # Mean over the tuples processed by this rank
        loss_value = float(np.mean(loss_values))
        loss_details = {
            name: float(np.mean(values)) for name, values in loss_details_values.items()
        }
        del batch

        metric_logger.update(epoch=epoch_f)
        for submodule_name in param_groups_name_to_idx_map:
            lr_name = f"lr_{submodule_name}" if submodule_name != "default" else "lr"
            log_lr = optimizer.param_groups[
                param_groups_name_to_idx_map[submodule_name][0]
            ]["lr"]
            metric_logger.meters[lr_name].update(log_lr)
        metric_logger.update(loss=loss_value, **loss_details)

        if (data_iter_step + 1) % args.train_params.print_freq == 0:
            loss_value_reduce = train_tools.all_reduce_mean(
                loss_value
            )  # MUST BE EXECUTED BY ALL NODES
            if log_writer is None:
                continue
            # epoch_1000x is the x-axis in tensorboard; it calibrates curves across batch sizes
            epoch_1000x = int(epoch_f * 1000)
            log_writer.add_scalar("train_loss", loss_value_reduce, epoch_1000x)
            if gradient_norm is not None:
                log_writer.add_scalar("train_grad_norm", gradient_norm, epoch_1000x)
            for submodule_name in param_groups_name_to_idx_map:
                lr_name = (
                    f"train_lr_{submodule_name}"
                    if submodule_name != "default"
                    else "train_lr"
                )
                log_lr = optimizer.param_groups[
                    param_groups_name_to_idx_map[submodule_name][0]
                ]["lr"]
                log_writer.add_scalar(lr_name, log_lr, epoch_1000x)
            log_writer.add_scalar("train_iter", epoch_1000x, epoch_1000x)
            for name, val in loss_details.items():
                log_writer.add_scalar("train_" + name, val, epoch_1000x)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
