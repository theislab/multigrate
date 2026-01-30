# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog][],
and this project adheres to [Semantic Versioning][].

[keep a changelog]: https://keepachangelog.com/en/1.0.0/
[semantic versioning]: https://semver.org/spec/v2.0.0.html

## [0.1.0] - TBA

### Added

- support for scvi-tools >= 1.0 ([#37](https://github.com/theislab/multigrate/pull/37))
- support for multiple NB lossed ([#34](https://github.com/theislab/multigrate/pull/34))
- tests

### Changed

- how the masks for missing modalities are calculated: now assume that the modality is missing when all features are zero instead of the sum over all features being <= 0 previously (only makes a difference when the input can be negative, which doesn't happen with standard single-cell modalities)
