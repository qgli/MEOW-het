# Resources used by the training loader

- `gen1_splits/train.json`, `gen1_splits/val.json`: the first-generation scenes used for training
  (1,603 training and 401 validation scenes; scene `proc_scene_NNNNNN` is generated from seed
  `NNNNNN`). Written by `scripts/make_procthor_unicol_splits.py` (seed 20260513).
- `gen1_frame_stats.json`: per-frame statistics of the first-generation renders used by the loader in Stages 1
  and 2 (`MEOW_GEN1_FRAME_STATS`, see docs/TRAINING.md): frames facing a flat wall (colour standard deviation
  below 0.05) are left out of native-render tuples. It covers the 110 scenes rendered first.
- `gen1_feasibility/feasibility_3type_full.parquet`, `feasibility_6name_full.parquet`: for every scene,
  tuple size K = 2..8 and camera composition (three camera types, or the six render names), the
  success rate of 30 covisibility-constrained walks over the scene's native renders. The stratified
  native-render sampler (`n5_sampling_v2`) draws feasible compositions from these tables for
  first-generation scenes (Stage-2 training and the validation sets of Stages 2 and 3). Produced by `scripts/precompute_feasibility_map.py` (see docs/DATA_ENGINE.md).
- `gen2_splits/{train,val,test}.json`: the second-generation split of Stage 3 (469 / 15 / 15 of the 499 scenes of
  seeds 1000-1499 that completed the production chain; seed 1474 did not). Written by
  `lenscope/genesis/make_splits.py` (seed 20260729).
- `gen2_early_batch_splits/{train,val}.json`: the 189 / 12 scenes (seeds 5-219) of the earlier second-generation
  batch used by Stage-3 run A, built and rendered at 2048 x 1024 before the production batch; whether this code
  regenerates them identically has not been checked.
