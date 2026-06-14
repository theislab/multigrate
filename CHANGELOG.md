# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog][],
and this project adheres to [Semantic Versioning][].

[keep a changelog]: https://keepachangelog.com/en/1.0.0/
[semantic versioning]: https://semver.org/spec/v2.0.0.html

## [1.0.0] - 2026-06-14

### Fixed

- `MultiVAE(losses=None)` no longer raises `TypeError: argument of type 'NoneType' is not iterable`. `losses=None` (the documented default, meaning MSE for all modalities) now works correctly in both the model wrapper (`_multivae.py`) and the torch module (`_multivae_torch.py`).
- `MultiVAE.load_query_data` no longer raises `AttributeError` when the reference model was trained with `integrate_on` but without an nb/zinb loss (i.e. `theta` is `None`).
- `StratifiedSampler` (used when `integrate_on` is set) no longer silently produces zero training batches on small datasets. Training will now always produce at least one batch per group, and raises a clear `ValueError` if the batch size is too large for the data.
- `organize_multimodal_anndatas` now preserves `.var` metadata from each modality. When the same column name appears in multiple modalities it is suffixed with the modality index (e.g. `mean_0`, `mean_1`). Fixes [#4](https://github.com/theislab/multigrate/issues/4).
- Defensive `.get("columns", [])` in the continuous-covariate registry lookup prevents a `KeyError` with certain scvi-tools versions. Fixes [#45](https://github.com/theislab/multigrate/issues/45).
- Removed deprecated `dtype=` argument from `anndata.AnnData()` call (anndata 0.12+).

### Changed

- Requires Python ≥ 3.12 (aligned with scvi-tools 1.4.3 minimum).
- Bumped core dependencies: `scvi-tools>=1.4.3`, `anndata>=0.12`; removed `numpy<2` and `jax<0.6` upper-bound pins.
- Migrated developer tooling from hatch to native [uv](https://docs.astral.sh/uv/). Dependency groups (`dev`, `test`, `doc`) are now declared via PEP 735 `[dependency-groups]` in `pyproject.toml`.
- CI rewritten to use `uv sync --group test` + `uv run pytest` against Python 3.12 and 3.13.
- ReadTheDocs build updated to use `uv sync --group doc` + `uv run sphinx-build`.
- How the masks for missing modalities are calculated: now assume that the modality is missing when all features are zero instead of the sum over all features being <= 0 previously (only makes a difference when the input can be negative, which doesn't happen with standard single-cell modalities)

### Added

- support for scvi-tools >= 1.0 ([#37](https://github.com/theislab/multigrate/pull/37))
- support for multiple NB lossed ([#34](https://github.com/theislab/multigrate/pull/34))
- tests
