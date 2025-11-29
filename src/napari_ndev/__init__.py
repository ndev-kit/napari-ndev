"""napari-ndev: No-code bioimage analysis toolkit for napari.

This package provides widgets for bioimage analysis workflows in napari.
Widgets are discovered via the napari plugin system (napari.yaml).

For programmatic use, the following modules are available:
- helpers: Utility functions for file handling and image processing
- measure: Functions for measuring label properties
- morphology: Functions for label morphology operations
- get_settings: Access to plugin settings
"""

try:
    from napari_ndev._version import version as __version__
except ImportError:
    __version__ = 'unknown'

# Import modules to make them available at package level for programmatic use
from napari_ndev import helpers, measure, morphology
from napari_ndev._settings import get_settings

__all__ = [
    '__version__',
    'get_settings',
    'helpers',
    'measure',
    'morphology',
]
