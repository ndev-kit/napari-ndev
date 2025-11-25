from pathlib import Path
from unittest.mock import patch

import pytest
from ndevio import nImage

from napari_ndev import __version__
from napari_ndev.widgets._ndev_container import nDevContainer


def test_ndev_container_init_no_viewer():
    ndev = nDevContainer()

    assert ndev._version_label.value == f'v{__version__}'
    assert ndev._viewer is None
    assert ndev._apoc_container is not None
    assert ndev._measure_container is not None
    assert ndev._utilities_container is not None
    assert ndev._workflow_container is not None

    with patch('webbrowser.open') as mock_open:
        ndev._open_docs_link()
        mock_open.assert_called_once_with('https://ndev-kit.github.io/')

    with patch('webbrowser.open') as mock_open:
        ndev._open_bug_report_link()
        mock_open.assert_called_once_with('https://github.com/TimMonko/napari-ndev/issues')

@pytest.fixture
def test_cells3d2ch_image(resources_dir: Path):
    path = resources_dir / 'cells3d2ch.tiff'
    img = nImage(path)
    return path, img

def test_ndev_container_viewer(make_napari_viewer, test_cells3d2ch_image, tmp_path: Path):
    viewer = make_napari_viewer()

    ndev = nDevContainer(viewer=viewer)
    assert ndev._viewer is viewer

    path, img = test_cells3d2ch_image
    ndev._utilities_container._files.value = path
    ndev._utilities_container.open_images()

    # make sure images opened and there are callbacks to the widgets
    assert viewer.layers[0] is not None

    # check interacting with alyers in utilities container works
    ndev._utilities_container._save_directory.value = tmp_path
    ndev._utilities_container._save_name.value = 'test'
    layer_data = ndev._utilities_container.save_layers_as_ome_tiff()

    expected_save_loc = tmp_path / 'Image' / 'test.tiff'
    assert layer_data.shape.__len__() == 4
    assert expected_save_loc.exists()

    # check interacting with apoc container works
    assert ndev._apoc_container._image_layers.choices == (
        viewer.layers[0], viewer.layers[1]
    )
