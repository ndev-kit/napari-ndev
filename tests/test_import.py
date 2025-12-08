import time

import pytest


def test_import_time():
    start_time = time.time()
    import napari_ndev  # noqa: F401

    end_time = time.time()

    import_time = end_time - start_time

    assert import_time < 1.0, 'napari_ndev took too long to import'

@pytest.mark.skip_ci
def test_widget_time(make_napari_viewer):

    start_time = time.time()
    
    viewer = make_napari_viewer()
    viewer.window.add_plugin_dock_widget('napari-ndev', 'nDev App')

    end_time = time.time()

    creation_time = end_time - start_time

    assert creation_time < 3.0, 'nDev App took too long to create'