# Multigrate: multiomic data integration for single-cell genomics

[![Tests][badge-tests]][link-tests]
[![Documentation][badge-docs]][link-docs]

[badge-tests]: https://img.shields.io/github/actions/workflow/status/theislab/multigrate/test.yaml?branch=main
[link-tests]: https://github.com/theislab/multigrate/actions/workflows/test.yml
[badge-docs]: https://img.shields.io/readthedocs/multigrate
[badge-colab]: https://colab.research.google.com/assets/colab-badge.svg

## Getting started

Please refer to the [documentation][link-docs]. In particular, the

- [API documentation][link-api]

and the tutorials:

- [Paired integration and query-to-reference mapping](https://multigrate.readthedocs.io/en/latest/notebooks/paired_integration_cite-seq.html) [![Open In Colab][badge-colab]](https://colab.research.google.com/github/theislab/multigrate/blob/main/docs/notebooks/paired_integration_cite-seq.ipynb)
- [Trimodal integration and query-to-reference mapping](https://multigrate.readthedocs.io/en/latest/notebooks/trimodal_integration.html) [![Open In Colab][badge-colab]](https://colab.research.google.com/github/theislab/multigrate/blob/main/docs/notebooks/trimodal_integration.ipynb)

See the paired integration tutorial for advice on which distributions and losses to use for which modalities.

## Installation

You need to have Python 3.12 or newer installed on your system.

### Quick start

To install the latest release of `multigrate` from [PyPI][link-pypi]:

```bash
pip install multigrate
```

Or install the latest development version:

```bash
pip install git+https://github.com/theislab/multigrate.git@main
```

### Development setup

For development, we recommend using [`uv`](https://docs.astral.sh/uv/) for fast dependency management:

```bash
# Clone the repository
git clone https://github.com/theislab/multigrate.git
cd multigrate

# Install in development mode with all optional dependencies
uv sync --all-groups
```

## Release notes

See the [changelog][changelog].

## Contact

If you found a bug, please use the [issue tracker][issue-tracker].

## Citation

> Anastasia Litinetskaya, Maiia Schulman, Fabiola Curion, Artur Szalata, Alireza Omidi, Mohammad Lotfollahi, and Fabian Theis. 2022. "Integration and querying of multimodal single-cell data with PoE-VAE." _bioRxiv_. https://doi.org/10.1101/2022.03.16.484643.

## Reproducibility

Code and notebooks to reproduce the results from the paper are available at https://github.com/theislab/multigrate_reproducibility.

[issue-tracker]: https://github.com/theislab/multigrate/issues
[changelog]: https://multigrate.readthedocs.io/latest/changelog.html
[link-docs]: https://multigrate.readthedocs.io
[link-api]: https://multigrate.readthedocs.io/latest/api.html
[link-pypi]: https://pypi.org/project/multigrate
