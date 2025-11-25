from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import dask.array as da

# import npe2
import numpy as np
import pytest
from ndevio import napari_get_reader

###############################################################################

RGB_TIFF = "RGB.tiff" # has two scenes
MULTISCENE_CZI = r"0T-4C-0Z-7pos.czi"
PNG_FILE = "nDev-logo-small.png"
# GIF_FILE = "example.gif"
OME_TIFF = "cells3d2ch.tiff"

###############################################################################

def test_napari_viewer_open(resources_dir: Path, make_napari_viewer) -> None:
    """
    Test that the napari viewer can open a file with the napari-ndev plugin.

    In zarr>=3.0, the FSStore was removed and replaced with DirectoryStore.
    This test checks that the napari viewer can open any file because BioImage
    (nImage) would try to import the wrong FSStore from zarr. Now, the FSStore
    is shimmed to DirectoryStore with a compatibility patch in nImage.
    """
    viewer = make_napari_viewer()
    viewer.open(str(resources_dir / OME_TIFF), plugin='napari-ndev')

    assert viewer.layers[0].data.shape == (60, 66, 85)

@pytest.mark.parametrize(
    ("in_memory", "expected_dtype"),
    [
        (True, np.ndarray),
        (False, da.core.Array),
    ],
)
@pytest.mark.parametrize(
    ("filename", "expected_shape", "expected_meta"),
    [
        (
            RGB_TIFF,
            (1440, 1920, 3),
            {
                'name': '0 :: Image:0 :: RGB', # multiscene naming
                'scale': (264.5833333333333, 264.5833333333333),
                'rgb': True,
            }
        ),
    ],
)
def test_reader(
    resources_dir: Path,
    filename: str,
    in_memory: bool,
    expected_shape: tuple[int, ...],
    expected_dtype,
    expected_meta: dict[str, Any],
    make_napari_viewer,
    # npe2pm: TestPluginManager,
) -> None:
    make_napari_viewer()

    # Resolve filename to filepath
    if isinstance(filename, str):
        path = str(resources_dir / filename)

    # Get reader
    partial_napari_reader_function = napari_get_reader(path, in_memory=in_memory, open_first_scene_only=True)
    # Check callable
    assert callable(partial_napari_reader_function)

    # Get data
    layer_data = partial_napari_reader_function(path)

    # We only return one layer
    if layer_data is not None:
        data, meta, _ = layer_data[0]

        # Check layer data
        assert isinstance(data, expected_dtype)
        assert data.shape == expected_shape

        # Check meta
        meta.pop('metadata', None)
        assert meta == expected_meta

    # now check open all scenes
    partial_napari_reader_function = napari_get_reader(path, in_memory=in_memory, open_all_scenes=True)
    assert callable(partial_napari_reader_function)

    layer_data = partial_napari_reader_function(path)
    assert len(layer_data) == 2


@pytest.mark.parametrize(
    ("in_memory", "expected_dtype"),
    [
        (True, np.ndarray),
        (False, da.core.Array),
    ],
)
@pytest.mark.parametrize(
    ("filename", "expected_shape"),
    [
        (RGB_TIFF, (1440, 1920, 3)),
        (MULTISCENE_CZI, (32, 32)),
    ],
)
def test_for_multiscene_widget(
    make_napari_viewer,
    resources_dir: Path,
    filename: str,
    in_memory: bool,
    expected_dtype,
    expected_shape: tuple[int, ...],
) -> None:
    # Make a viewer
    viewer = make_napari_viewer()
    assert len(viewer.layers) == 0
    assert len(viewer.window._dock_widgets) == 0

    # Resolve filename to filepath
    if isinstance(filename, str):
        path = str(resources_dir / filename)

    # Get reader
    reader = napari_get_reader(path, in_memory)

    if reader is not None:
        # Call reader on path
        reader(path)

        if len(viewer.window._dock_widgets) != 0:
            # Get the second scene
            scene_widget = viewer.window._dock_widgets[
                f"{Path(filename).stem} :: Scenes"
            ].widget()._magic_widget
            assert scene_widget is not None
            assert scene_widget.viewer == viewer

            scenes = scene_widget._scene_list_widget.choices

            # Set to the first scene (0th choice is none)
            scene_widget._scene_list_widget.value = scenes[1]

            data = viewer.layers[0].data

            assert isinstance(data, expected_dtype)
            assert data.shape == expected_shape
        else:
            data, _, _ = reader(path)[0]
            assert isinstance(data, expected_dtype)
            assert data.shape == expected_shape

def test_napari_get_reader_multi_path(resources_dir: Path) -> None:
    # Get reader
    reader = napari_get_reader(
        [str(resources_dir / RGB_TIFF), str(resources_dir / MULTISCENE_CZI)],
        in_memory=True,
    )

    # Check callable
    assert reader is None

def test_napari_get_reader_ome_override(resources_dir: Path) -> None:
    reader = napari_get_reader(
        str(resources_dir / OME_TIFF),
    )

    assert callable(reader)

def test_napari_get_reader_unsupported(resources_dir: Path) -> None:
    reader = napari_get_reader(
        str(resources_dir / "measure_props_Labels.abcdefg"),
    )

    assert reader is None

def test_napari_get_reader_general_exception(caplog):
    """Test that general exceptions in determine_reader_plugin are handled correctly."""
    test_path = "non_existent_file.xyz"

    # Mock determine_reader_plugin to raise an exception
    with patch('ndevio._napari_reader.determine_reader_plugin') as mock_reader:
        mock_reader.side_effect = Exception("Test exception")

        reader = napari_get_reader(test_path)
        assert reader is None

        assert "Error reading file" in caplog.text

def test_napari_get_reader_png(resources_dir: Path) -> None:
    reader = napari_get_reader(
        str(resources_dir / PNG_FILE),
    )

    assert callable(reader)
