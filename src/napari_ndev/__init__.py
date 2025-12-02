"""napari-ndev: No-code bioimage analysis toolkit for napari.

This package provides widgets for bioimage analysis workflows in napari.
Widgets are discovered via the napari plugin system (napari.yaml).

For programmatic use, the following modules are available:
- helpers: Utility functions for file handling and image processing
- measure: Functions for measuring label properties
- morphology: Functions for label morphology operations
- get_settings: Access to plugin settings

Modules are lazily imported to minimize startup time.
"""

import importlib
from typing import TYPE_CHECKING

try:
    from napari_ndev._version import version as __version__
except ImportError:
    __version__ = 'unknown'

# Lazy import get_settings since it's lightweight
from napari_ndev._settings import get_settings

__all__ = [
    '__version__',
    'get_settings',
    'helpers',
    'measure',
    'morphology',
]


def __getattr__(name: str):
    """Lazily import modules to speed up package import."""
    if name in ('helpers', 'measure', 'morphology'):
        module = importlib.import_module(f'.{name}', __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if TYPE_CHECKING:
    from napari_ndev import helpers, measure, morphology
