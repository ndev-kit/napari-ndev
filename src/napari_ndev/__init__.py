"""napari-ndev: No-code bioimage analysis toolkit for napari.

This package provides widgets for bioimage analysis workflows in napari.
Widgets are discovered via the napari plugin system (napari.yaml).

For programmatic use, the following modules are available:
- measure: Functions for measuring label properties
- morphology: Functions for label morphology operations

For I/O utilities and helpers, use ndevio:
    from ndevio import helpers
    from ndevio.widgets import UtilitiesContainer

Settings are managed by the ndev-settings package:
    from ndev_settings import get_settings

Modules are lazily imported to minimize startup time.
"""

import importlib
from typing import TYPE_CHECKING

try:
    from napari_ndev._version import version as __version__
except ImportError:
    __version__ = 'unknown'


__all__ = [
    '__version__',
    'measure',
    'morphology',
]


def __getattr__(name: str):
    """Lazily import modules to speed up package import."""
    if name in ('measure', 'morphology'):
        module = importlib.import_module(f'.{name}', __name__)
        globals()[name] = module
        return module
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')


if TYPE_CHECKING:
    from napari_ndev import measure, morphology
