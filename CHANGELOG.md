# Changelog

### Docker
- Replaced `pytorch/pytorch:1.13.1-cuda11.6` base image with `ubuntu:22.04` and a clean Python 3.10 virtual environment
- Switched from `pip`/`conda` to `uv` as the package manager
- Removed manual installations of libtorch, ONNX Runtime, Eigen, OpenCV, and Boost — simplified to only what is needed
- Removed in-tree `pyamg` build step from the Dockerfile (seems like it's not required in case of KMeans usage)
- `pogema-toolbox` is now installed separately with `--no-deps` in the Dockerfile instead of via `requirements.txt`

### Removed wandb dependency
- Removed `wandb` import and all `wandb.init()` / `wandb.log()` / `wandb.finish()` calls from:
  - `scrimp/driver.py`
  - `test_expert.py`
  - `test_imitation_learning_pyg.py`
  - `hmagat/train_imitation_learning_pyg.py`
  - `hmagat/post_train_quality_imp.py`
  - `hmagat/temperature_training.py`
- Removed `--wandb_project`, `--wandb_entity`, and `--wandb_tag` CLI arguments
- Removed `wandb` and `pogema-toolbox` from `requirements.txt`

### Bug fixes
- Replaced deprecated `np.int` (removed in NumPy 1.24) with `int` in `hmagat/collision_shielding.py` (3 occurrences)
- Fixed `bfs()` call in `grid_config_generator.py` to match its current signature (removed extra `map_w` argument)

### Reproducibility
- Added `random_state=self.seed` to `KMeans` in `hmagat/hypergraph_gen_strategies/base.py`

### New files
- `maps/` — YAML map definitions (mazes, movingai, random, warehouse)
- `test_with_pogema.py` — test script using the Pogema environment
- `results/` — the results of evaluation of HMAGAT on pogema-benchmark instances. Also contains some SVG examples where HMAGAT gets stuck.
