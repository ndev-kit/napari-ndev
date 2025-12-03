"""
Widget containers for the napari-ndev package.

The available containers include:
- ApocContainer: Container for APOC-related widgets.
- ApocFeatureStack: Container for stacking APOC features.
- MeasureContainer: Container for measurement-related widgets.
- UtilitiesContainer: Container for utility widgets.
- WorkflowContainer: Container for workflow management widgets.
- nDevContainer: Main application container.

Settings are managed by the ndev-settings package. The settings widget
is available via: Plugins > ndev Settings (ndev-settings)

Note: Widgets are discovered via napari.yaml and should not be imported directly.
They are loaded lazily by napari when needed.
"""
