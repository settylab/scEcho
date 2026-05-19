from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("scEcho")
except PackageNotFoundError:
    __version__ = "0.0.0+local"

__author__ = "Connor Finkbeiner"

from . import Echo_features, Echo_states, plotting, test_components, try_models, utils

__all__ = [
    "Echo_states",
    "Echo_features",
    "plotting",
    "try_models",
    "utils",
    "test_components",
    "__version__",
    "__author__",
]
