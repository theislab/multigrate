from importlib.metadata import version

from . import data, distributions, model, module, nn, train

__all__ = ["data", "distributions", "model", "module", "nn", "train"]

__version__ = version("multigrate")
