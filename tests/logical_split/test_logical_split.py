"""CPU tests of logical-split training (mapanything/train/logical_split.py).

Run from the repository root with PYTHONPATH set as for training:
    python -m pytest -q tests/logical_split

A toy model with a per-tuple mean loss is trained for one epoch in logical-split mode, with one
process (s = 1) and with two gloo processes that split every batch of one logical rank (s = 2).
The gradient of every optimiser step must equal the gradient of a whole-batch step on the
logical batch.
"""

import datetime
import os

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from omegaconf import OmegaConf
from torch.utils.data import default_collate

import mapanything.datasets as datasets
from mapanything.datasets.base.easy_dataset import EasyDataset
from mapanything.train import logical_split

NUM_VIEWS = [2, 3]
BATCH_SIZE_MAP = {0: 3, 1: 2}  # K = 2: three tuples (split 2 + 1), K = 3: two tuples
MAX_IMAGES = 6


class ToyDataset(EasyDataset):
    """Deterministic tuples of 2 or 3 views: a 3x4x4 image and a 2-vector target per view."""

    num_views = NUM_VIEWS
    _resolutions = [(4, 4)]

    def __len__(self):
        return 24

    def __getitem__(self, idx):
        sample_idx, _, k_idx = idx
        gen = torch.Generator().manual_seed(int(sample_idx))
        return [
            {
                "img": torch.randn(3, 4, 4, generator=gen, dtype=torch.float64),
                "target": torch.randn(2, generator=gen, dtype=torch.float64),
                "label": f"tuple{int(sample_idx)}",
            }
            for _ in range(self.num_views[k_idx])
        ]


class ToyModel(torch.nn.Module):
    """Per-view prediction from the view's features and the mean features of the tuple."""

    def __init__(self):
        super().__init__()
        gen = torch.Generator().manual_seed(0)
        self.encoder = torch.nn.Linear(48, 8).double()
        self.head = torch.nn.Linear(16, 2).double()
        with torch.no_grad():
            for p in self.parameters():
                p.copy_(0.3 * torch.randn(p.shape, generator=gen, dtype=torch.float64))

    def forward(self, views):
        feats = [torch.tanh(self.encoder(v["img"].flatten(1))) for v in views]
        context = torch.stack(feats).mean(0)
        return [{"pred": self.head(torch.cat([f, context], dim=1))} for f in feats]


def criterion(views, preds):
    """Mean over tuples of a per-tuple loss."""
    per_tuple = sum(((p["pred"] - v["target"]) ** 2).sum(1) for v, p in zip(views, preds))
    loss = per_tuple.mean()
    return loss, {"toy_loss": float(loss)}


class RecordingScaler:
    """Stand-in for NativeScalerWithGradNormCount that records the gradients of every step."""

    def __init__(self, model):
        self.model = model
        self.steps = []
        self.sync_flags = []

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, update_grad=True):
        # DDP sets require_backward_grad_sync to False inside no_sync()
        self.sync_flags.append(bool(getattr(self.model, "require_backward_grad_sync", True)))
        loss.backward()
        if update_grad:
            params = self.model.module if hasattr(self.model, "module") else self.model
            self.steps.append(
                {
                    "grads": {n: p.grad.clone() for n, p in params.named_parameters()},
                    "lr": optimizer.param_groups[0]["lr"],
                    "sync_flags": self.sync_flags,
                }
            )
            self.sync_flags = []
        return None


def make_args(output_dir):
    return OmegaConf.create(
        {
            "output_dir": str(output_dir),
            "train_params": {
                "logical_world_size": 1,
                "accum_iter": 1,
                "print_freq": 1000,
                "amp": 0,
                "amp_dtype": "fp32",
                "check_loss_instability": True,
                "max_loss_value": 1e9,
                "lr": 0.1,
                "min_lr": 0.001,
                "warmup_epochs": 1,
                "epochs": 4,
                "schedule_type": "linear_warmup_half_cycle_cosine_decay",
                "submodule_configs": {},
            },
        }
    )


def loader_kwargs():
    return dict(
        max_num_of_imgs_per_gpu=MAX_IMAGES,
        num_workers=0,
        pin_mem=False,
        feature_to_batch_size_map_override=BATCH_SIZE_MAP,
    )


def run_logical_split_epoch(output_dir, epoch=1):
    """One epoch of logical-split training in this process; returns the recorded steps."""
    loader = logical_split.get_logical_split_train_data_loader(
        ToyDataset(), logical_world_size=1, **loader_kwargs()
    )
    model = ToyModel()
    if dist.is_initialized():
        model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    scaler = RecordingScaler(model)
    stats = logical_split.train_one_epoch_logical_split(
        model,
        criterion,
        loader,
        optimizer,
        torch.device("cpu"),
        epoch,
        scaler,
        make_args(output_dir),
        param_groups_name_to_idx_map={"default": [0]},
        param_groups_idx_to_name_map={0: "default"},
        model_without_ddp=model,
    )
    shares = [[tuple(int(x) for x in item) for item in share] for share in loader.batch_sampler]
    return {
        "steps": scaler.steps,
        "stats": stats,
        "shares": shares,
        "local_sizes": [len(share) for share in shares],
    }


def whole_batch_reference(epoch=1):
    """Gradient of a whole-batch step (the regular training path) for every logical batch."""
    loader = datasets.get_train_data_loader(ToyDataset(), **loader_kwargs())
    loader.batch_sampler.set_epoch(epoch)
    logical_batches = [[tuple(int(x) for x in item) for item in b] for b in loader.batch_sampler]
    model = ToyModel()
    reference = []
    for batch, logical_batch in zip(loader, logical_batches):
        model.zero_grad()
        loss, _ = criterion(batch, model(batch))
        if len(batch) > 2:
            loss = loss * (2 / len(batch))
        loss.backward()
        reference.append(
            {
                "grads": {n: p.grad.clone() for n, p in model.named_parameters()},
                "batch_size": int(batch[0]["img"].shape[0]),
                "tuples": logical_batch,
            }
        )
    return reference


def assert_same_gradients(steps, reference):
    assert len(steps) == len(reference) > 0
    for step, expected in zip(steps, reference):
        for name, grad in expected["grads"].items():
            torch.testing.assert_close(step["grads"][name], grad, rtol=1e-12, atol=1e-14)


def test_single_process_matches_whole_batch(tmp_path):
    result = run_logical_split_epoch(tmp_path)
    reference = whole_batch_reference()
    # the epoch holds logical batches of both sizes
    assert {r["batch_size"] for r in reference} == set(BATCH_SIZE_MAP.values())
    assert result["local_sizes"] == [r["batch_size"] for r in reference]
    assert_same_gradients(result["steps"], reference)
    # one optimiser step per logical batch, at the learning rate of the logical step
    for i, step in enumerate(result["steps"]):
        epoch_f = 1 + i / len(result["steps"])
        expected_lr = 0.001 + (0.1 - 0.001) * 0.5 * (1 + np.cos(np.pi * (epoch_f - 1) / 3))
        assert step["lr"] == pytest.approx(expected_lr)
    assert np.isfinite(result["stats"]["loss"])


def _distributed_worker(rank, store, output_dir):
    dist.init_process_group(
        "gloo",
        init_method=f"file://{store}",
        rank=rank,
        world_size=2,
        timeout=datetime.timedelta(seconds=60),
    )
    try:
        result = run_logical_split_epoch(os.path.join(output_dir, f"rank{rank}"))
        torch.save(result, os.path.join(output_dir, f"result{rank}.pt"))
    finally:
        dist.destroy_process_group()


def test_two_processes_split_matches_whole_batch(tmp_path):
    for rank in range(2):
        os.makedirs(tmp_path / f"rank{rank}")
    mp.start_processes(
        _distributed_worker,
        args=(str(tmp_path / "store"), str(tmp_path)),
        nprocs=2,
        start_method="spawn",
    )
    reference = whole_batch_reference()
    results = [torch.load(tmp_path / f"result{rank}.pt") for rank in range(2)]
    for result in results:
        # the all-reduced gradient of every step equals the whole-batch gradient
        assert_same_gradients(result["steps"], reference)
        # all but the last tuple of a step run under no_sync
        for step, size in zip(result["steps"], result["local_sizes"]):
            assert step["sync_flags"] == [False] * (size - 1) + [True]
    # rank r takes tuples r::2 of every logical batch (a three-tuple batch is split 2 + 1)
    for rank, result in enumerate(results):
        assert result["shares"] == [expected["tuples"][rank::2] for expected in reference]
    assert results[0]["local_sizes"] != results[1]["local_sizes"]


def test_split_into_samples_matches_collation():
    samples = [
        [
            {
                "img": torch.full((3, 2, 2), float(i)),
                "depth": np.full((2, 2), i, dtype=np.float32),
                "idx": (i, 7, v),
                "scale": 0.5 * i,
                "label": f"s{i}",
                "names": [f"a{i}", f"b{i}", f"c{i}"],  # as many entries as tuples in the batch
                "meta": {"k": torch.tensor([i, i + 1]), "tag": f"t{i}"},
            }
            for v in range(2)
        ]
        for i in range(3)
    ]

    def collate(tuples):
        return [default_collate(list(views)) for views in zip(*tuples)]

    def assert_same(a, b):
        if torch.is_tensor(a):
            assert torch.is_tensor(b) and a.dtype == b.dtype and torch.equal(a, b)
        elif isinstance(a, dict):
            assert a.keys() == b.keys()
            for key in a:
                assert_same(a[key], b[key])
        elif isinstance(a, (list, tuple)):
            assert type(a) is type(b) and len(a) == len(b)
            for x, y in zip(a, b):
                assert_same(x, y)
        else:
            assert a == b

    singles = list(logical_split.split_into_samples(collate(samples)))
    assert len(singles) == 3
    for i, single in enumerate(singles):
        assert_same(single, collate([samples[i]]))


def test_training_hooks(tmp_path, monkeypatch):
    from mapanything.train import training

    def build(**kwargs):
        return training.build_dataset(
            ToyDataset(),
            num_workers=0,
            test=False,
            max_num_of_imgs_per_gpu=MAX_IMAGES,
            feature_to_batch_size_map_override=BATCH_SIZE_MAP,
            **kwargs,
        )

    split_loader = build(logical_world_size=1)
    assert isinstance(split_loader.batch_sampler, logical_split.LogicalSplitBatchSampler)
    assert not isinstance(build().batch_sampler, logical_split.LogicalSplitBatchSampler)

    calls = []
    monkeypatch.setattr(
        logical_split,
        "train_one_epoch_logical_split",
        lambda *args, **kwargs: calls.append(args) or {"loss": 0.0},
    )
    stats = training.train_one_epoch(
        None, None, split_loader, None, None, 0, None, make_args(tmp_path)
    )
    assert stats == {"loss": 0.0} and calls[0][2] is split_loader


class FixedViewsDataset(ToyDataset):
    num_views = 2
    _resolutions = [(4, 4), (4, 8)]


def test_sampler_settings_are_checked():
    def sampler(dataset, batch_size_map, logical_world_size=1, world_size=1):
        return logical_split.get_logical_split_train_data_loader(
            dataset,
            max_num_of_imgs_per_gpu=MAX_IMAGES,
            logical_world_size=logical_world_size,
            rank=0,
            world_size=world_size,
            num_workers=0,
            pin_mem=False,
            feature_to_batch_size_map_override=batch_size_map,
        ).batch_sampler

    assert sampler(ToyDataset(), BATCH_SIZE_MAP).batch_size_by_num_views == {2: 3, 3: 2}
    # keys missing from the map get batch size 1, as in the sampler
    assert sampler(ToyDataset(), {0: 3}).batch_size_by_num_views == {2: 3, 3: 1}
    # a fixed number of views with a batch size per aspect-ratio bucket is rejected
    with pytest.raises(ValueError, match="one batch size per number of views"):
        sampler(FixedViewsDataset(), {0: 3, 1: 2})
    assert sampler(FixedViewsDataset(), {0: 3, 1: 3}).batch_size_by_num_views == {2: 3}
    # every physical rank needs at least one tuple of every logical batch
    with pytest.raises(ValueError, match="at least 3 tuples"):
        sampler(ToyDataset(), BATCH_SIZE_MAP, world_size=3)
    with pytest.raises(ValueError, match="must divide"):
        sampler(ToyDataset(), BATCH_SIZE_MAP, logical_world_size=2, world_size=3)
