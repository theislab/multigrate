# Multigrate: multiomic data integration for single-cell genomics

[![Tests][badge-tests]][link-tests]
[![Documentation][badge-docs]][link-docs]

ICML 2021 CompBio Workshop paper is on [bioarxiv](https://www.biorxiv.org/content/10.1101/2022.03.16.484643v1).

Also see the reproducibility [repo](https://github.com/theislab/multigrate_reproducibility).

## Getting started

Please refer to the [documentation][link-docs]. In particular, the

-   [API documentation][link-api].

## Installation

You need to have Python 3.10 or newer installed on your system. If you don't have
Python installed, we recommend installing [Miniconda](https://docs.conda.io/en/latest/miniconda.html).

Create a new conda environment and activate it:

```bash
conda create -n multigrate python=3.10
conda activate multigrate
```

There are several alternative options to install multigrate:

<!--
1) Install the latest release of `multigrate` from `PyPI <https://pypi.org/project/multigrate/>`_:

```bash
pip install multigrate
```
-->

1. Install the latest development version:

```bash
pip install git+https://github.com/theislab/multigrate.git@main
```

## Release notes

See the [changelog][changelog].

## Contact

If you found a bug, please use the [issue tracker][issue-tracker].

## Citation

If you use multigrate in your research, please consider citing

```
@article {Lotfollahi2022.03.16.484643,
	author = {Lotfollahi, Mohammad and Litinetskaya, Anastasia and Theis, Fabian J.},
	title = {Multigrate: single-cell multi-omic data integration},
	elocation-id = {2022.03.16.484643},
	year = {2022},
	doi = {10.1101/2022.03.16.484643},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2022/03/17/2022.03.16.484643},
	eprint = {https://www.biorxiv.org/content/early/2022/03/17/2022.03.16.484643.full.pdf},
	journal = {bioRxiv}
}
```

[badge-tests]: https://img.shields.io/github/workflow/status/theislab/multigrate/Test/main
[link-tests]: https://github.com/theislab/multigrate/actions/workflows/test.yaml
[badge-docs]: https://img.shields.io/readthedocs/multigrate
[issue-tracker]: https://github.com/theislab/multigrate/issues
[changelog]: https://multigrate.readthedocs.io/en/latest/changelog.html
[link-docs]: https://multigrate.readthedocs.io
[link-api]: https://multigrate.readthedocs.io/en/latest/api.html
