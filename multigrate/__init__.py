from . import data, distributions, metrics, model, module, nn, train

__all__ = ["data", "distributions", "metrics", "model", "module", "nn", "train"]


try:
    import importlib.metadata as importlib_metadata
except:
    import importlib_metadata

package_name = "multigrate"
__version__ = importlib_metadata.version(package_name)
