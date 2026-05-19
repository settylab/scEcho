from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("scEcho")
except PackageNotFoundError:
    __version__ = "0.0.0+local"

__author__ = "Connor Finkbeiner"

from . import Echo_features, echo_states, plotting, utils

__all__ = [
    "echo_states",
    "Echo_features",
    "plotting",
    "utils",
    "__version__",
    "__author__",
]
