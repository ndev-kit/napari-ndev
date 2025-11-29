"""
Widget containers for the napari-ndev package.

The available containers include:
- ApocContainer: Container for APOC-related widgets.
- ApocFeatureStack: Container for stacking APOC features.
- MeasureContainer: Container for measurement-related widgets.
- UtilitiesContainer: Container for utility widgets.
- WorkflowContainer: Container for workflow management widgets.
- SettingsContainer: Container for managing global plugin settings.
- nDevContainer: Main application container.

Note: Widgets are discovered via napari.yaml and should not be imported directly.
They are loaded lazily by napari when needed.
"""
