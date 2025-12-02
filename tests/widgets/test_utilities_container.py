from pathlib import Path

import natsort
import numpy as np
import pytest
from ndevio import nImage

from napari_ndev.widgets._utilities_container import UtilitiesContainer

image_2d = np.asarray([[0, 0, 1, 1], [0, 0, 1, 1], [2, 2, 1, 1], [2, 2, 1, 1]])
shapes_2d = np.array([[0.25, 0.25], [0.25, 2.75], [2.75, 2.75], [2.75, 0.25]])
labels_2d = np.asarray(
    [[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]]
)

image_4d = np.random.random((1, 1, 10, 10))
shapes_4d = [
    np.array([[0, 0, 1, 1], [0, 0, 1, 3], [0, 0, 5, 3], [0, 0, 5, 1]]),
    np.array([[0, 0, 5, 5], [0, 0, 5, 9], [0, 0, 9, 9], [0, 0, 9, 5]]),
]
labels_4d = np.array(
    [
        [
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                [0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                [0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                [0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                [0, 1, 1, 1, 0, 2, 2, 2, 2, 2],
                [0, 0, 0, 0, 0, 2, 2, 2, 2, 2],
                [0, 0, 0, 0, 0, 2, 2, 2, 2, 2],
                [0, 0, 0, 0, 0, 2, 2, 2, 2, 2],
                [0, 0, 0, 0, 0, 2, 2, 2, 2, 2],
            ]
        ]
    ]
)


@pytest.fixture(
    params=[
        (image_2d, shapes_2d, labels_2d, 'YX'),
        (image_4d, shapes_4d, labels_4d, 'TZYX'),
    ]
)
def test_data(request: pytest.FixtureRequest):
    return request.param


def test_save_shapes_as_labels(
    qtbot,
    make_napari_viewer,
    tmp_path: Path,
    test_data,
):
    test_image, test_shape, _, squeezed_dims = test_data

    viewer = make_napari_viewer()
    viewer.add_image(test_image)
    viewer.add_shapes(test_shape)
    container = UtilitiesContainer(viewer)

    container._viewer.layers.selection.active = viewer.layers['test_shape']
    container._save_directory.value = tmp_path
    container._save_name.value = 'test'

    container.save_layers_as_ome_tiff()

    # Wait for the threaded worker to complete
    with qtbot.waitSignal(
        container._layer_save_worker.finished, timeout=60000
    ):
        pass

    expected_save_loc = tmp_path / 'Shapes' / 'test.tiff'
    assert expected_save_loc.exists()
    saved_img = nImage(expected_save_loc)
    assert saved_img.shape[1] == 1  # single channel (C dimension is index 1)
    assert saved_img.channel_names == ['Shapes']


def test_save_labels(qtbot, make_napari_viewer, tmp_path: Path, test_data):
    _, _, test_labels, squeezed_dims = test_data

    viewer = make_napari_viewer()
    viewer.add_labels(
        test_labels
    )  # <- should add a way to specify this is the selected layer in the viewer
    viewer.layers.selection.active = viewer.layers['test_labels']
    container = UtilitiesContainer(viewer)

    container._save_directory.value = tmp_path
    container._save_name.value = 'test'

    container.save_layers_as_ome_tiff()

    # Wait for the threaded worker to complete
    with qtbot.waitSignal(
        container._layer_save_worker.finished, timeout=60000
    ):
        pass

    expected_save_loc = tmp_path / 'Labels' / 'test.tiff'

    assert expected_save_loc.exists()
    saved_img = nImage(expected_save_loc)
    assert saved_img.shape[1] == 1  # single channel (C dimension is index 1)
    assert saved_img.channel_names == ['Labels']


def test_save_image_layer(
    qtbot, make_napari_viewer, test_data, tmp_path: Path
):
    test_image, _, _, squeezed_dims = test_data
    viewer = make_napari_viewer()
    viewer.add_image(test_image)
    container = UtilitiesContainer(viewer)

    container._viewer.layers.selection.active = viewer.layers['test_image']
    container._channel_names.value = ['0']
    container._save_directory.value = tmp_path
    container._save_name.value = 'test'

    container.save_layers_as_ome_tiff()

    # Wait for the threaded worker to complete
    with qtbot.waitSignal(
        container._layer_save_worker.finished, timeout=60000
    ):
        pass

    expected_save_loc = tmp_path / 'Image' / 'test.tiff'

    assert expected_save_loc.exists()
    saved_img = nImage(expected_save_loc)
    assert saved_img.shape[1] == 1  # single channel (C dimension is index 1)
    assert saved_img.channel_names == ['0']


def test_save_multi_layer(
    qtbot, make_napari_viewer, test_data, tmp_path: Path
):
    test_image, _, test_labels, squeezed_dims = test_data
    viewer = make_napari_viewer()
    viewer.add_image(test_image)
    viewer.add_labels(test_labels)
    container = UtilitiesContainer(viewer)

    container._viewer.layers.selection = [
        viewer.layers['test_labels'],
        viewer.layers['test_image'],
    ]
    container._save_directory.value = tmp_path
    container._save_name.value = 'test'

    container.save_layers_as_ome_tiff()

    # Wait for the threaded worker to complete
    with qtbot.waitSignal(
        container._layer_save_worker.finished, timeout=60000
    ):
        pass

    expected_save_loc = tmp_path / 'Layers' / 'test.tiff'

    assert expected_save_loc.exists()
    saved_img = nImage(expected_save_loc)
    assert saved_img.shape[1] == 2  # two channels (C dimension is index 1)


@pytest.fixture
def test_rgb_image(resources_dir: Path):
    path = resources_dir / 'RGB.tiff'
    img = nImage(path)
    return path, img


def test_update_metadata_from_file(make_napari_viewer, test_rgb_image):
    viewer = make_napari_viewer()
    container = UtilitiesContainer(viewer)

    path, _ = test_rgb_image
    container._files.value = path
    container.update_metadata_on_file_select()

    assert container._save_name.value == 'RGB'
    assert (
        container._dim_shape.value
        == 'T: 1, C: 1, Z: 1, Y: 1440, X: 1920, S: 3'
    )
    assert container._squeezed_dims_order == 'YX'
    assert container._channel_names.value == "['red', 'green', 'blue']"


def test_update_metadata_from_layer(make_napari_viewer, test_data):
    test_image, _, _, _ = test_data
    viewer = make_napari_viewer()
    viewer.add_image(test_image, scale=(2, 3))
    container = UtilitiesContainer(viewer)

    container._viewer.layers.selection.active = viewer.layers['test_image']
    container.update_metadata_from_layer()

    assert (
        'Tried to update metadata, but could only update scale'
    ) in container._results.value
    assert container._scale_tuple.value == (1, 2, 3)


@pytest.fixture
def test_czi_image(resources_dir: Path):
    path = resources_dir / '0T-4C-0Z-7pos.czi'
    img = nImage(path)
    return path, img


def test_save_files_as_ome_tiff(test_czi_image, tmp_path: Path, qtbot):
    path, _ = test_czi_image
    container = UtilitiesContainer()
    container._files.value = path
    container._save_directory.value = tmp_path
    save_dir = tmp_path / 'ConcatenatedImages'

    container.save_files_as_ome_tiff()

    # Wait for the threaded worker to complete
    with qtbot.waitSignal(container._concat_worker.finished, timeout=60000):
        pass

    # check that there is 1 file
    assert len(list(save_dir.iterdir())) == 1
    # check the name of the file is 0T-4C-0Z-7pos.tiff
    assert (save_dir / '0T-4C-0Z-7pos.tiff').exists()


@pytest.mark.parametrize('num_files', [1, 3])
def test_select_next_images(resources_dir: Path, num_files: int):
    container = UtilitiesContainer()

    image_dir = resources_dir / 'test_czis'
    # get all the files in the directory
    all_image_files = list(image_dir.iterdir())
    # sort the files
    all_image_files = natsort.os_sorted(all_image_files)

    container._files.value = all_image_files[:num_files]

    container.select_next_images()

    selected_files = container._files.value
    if isinstance(selected_files, tuple):
        selected_files = list(selected_files)

    assert len(selected_files) == num_files

    for i in range(num_files):
        assert selected_files[i] == all_image_files[i + num_files]


def test_batch_concatenate_files(tmp_path: Path, resources_dir: Path, qtbot):
    container = UtilitiesContainer()
    image_dir = resources_dir / 'test_czis'
    all_image_files = list(image_dir.iterdir())

    all_image_files = natsort.os_sorted(all_image_files)

    container._files.value = all_image_files[:1]

    container._save_directory.value = tmp_path
    container._save_directory_prefix.value = 'test'
    container.batch_concatenate_files()

    # Wait for threaded batch to complete
    qtbot.waitUntil(
        lambda: not container._batch_runner.is_running, timeout=60000
    )

    expected_output_dir = tmp_path / 'test_ConcatenatedImages'

    assert expected_output_dir.exists()
    # 8 tiff files + 1 log file = 9 total
    output_files = list(expected_output_dir.iterdir())
    tiff_files = [f for f in output_files if f.suffix == '.tiff']
    assert len(tiff_files) == 8
    assert (expected_output_dir / 'batch_concatenate.log.txt').exists()


def test_batch_cancel_button(tmp_path: Path, resources_dir: Path, qtbot):
    """Test that cancel button stops the batch runner."""
    container = UtilitiesContainer()
    image_dir = resources_dir / 'test_czis'
    all_image_files = list(image_dir.iterdir())
    all_image_files = natsort.os_sorted(all_image_files)

    container._files.value = tuple(all_image_files[:1])
    container._save_directory.value = tmp_path
    container._save_directory_prefix.value = 'test'

    # Start the batch operation
    container.batch_concatenate_files()

    # Verify the batch is running
    assert container._batch_runner.is_running

    # Click cancel (the button text changes to 'Cancel' when running)
    container._concatenate_batch_button.clicked()

    # Wait for cancellation to complete
    qtbot.waitUntil(
        lambda: not container._batch_runner.is_running, timeout=10000
    )

    # Verify it stopped
    assert not container._batch_runner.is_running


def test_batch_error_callback(qtbot):
    """Test that error callback updates progress bar label."""
    from unittest.mock import MagicMock

    from nbatch import BatchContext

    container = UtilitiesContainer()

    # Create a mock context with a file_set item (files, save_name)
    mock_files = [Path('file1.tiff'), Path('file2.tiff')]
    ctx = MagicMock(spec=BatchContext)
    ctx.item = (mock_files, 'bad_file')

    test_exception = ValueError('Test error message')

    container._on_batch_error(ctx, test_exception)

    # Verify progress bar label was updated with error info
    assert 'Error on bad_file' in container._progress_bar.label
    assert 'Test error message' in container._progress_bar.label


def test_batch_button_state_toggle(qtbot):
    """Test that batch button toggles between run and cancel states."""
    container = UtilitiesContainer()

    # Initial state should be 'Batch Concat.'
    assert container._concatenate_batch_button.text == 'Batch Concat.'

    # Set to running state
    container._set_batch_button_state(running=True)
    assert container._concatenate_batch_button.text == 'Cancel'

    # Set back to not running
    container._set_batch_button_state(running=False)
    assert container._concatenate_batch_button.text == 'Batch Concat.'


def test_save_scenes_ome_tiff(test_czi_image, tmp_path: Path, qtbot):
    path, _ = test_czi_image
    container = UtilitiesContainer()
    container._files.value = path
    container._save_directory.value = tmp_path
    save_dir = tmp_path / 'ExtractedScenes'

    container.save_scenes_ome_tiff()

    # Wait for the threaded worker to complete
    with qtbot.waitSignal(container._scene_worker.finished, timeout=60000):
        pass

    # check that there are 7 files in the save dir
    assert len(list(save_dir.iterdir())) == 7


def test_extract_and_save_scenes_ome_tiff(test_czi_image, tmp_path: Path):
    """Test the pure function for extracting scenes directly."""
    from napari_ndev.widgets._utilities_container import (
        extract_and_save_scenes_ome_tiff,
    )

    path, _ = test_czi_image
    save_dir = tmp_path / 'ExtractedScenes'

    # Collect all yielded results
    results = list(extract_and_save_scenes_ome_tiff(path, save_dir))

    # Should have 7 scenes
    assert len(results) == 7
    # Each result should be (scene_idx, scene_name)
    assert all(isinstance(r, tuple) and len(r) == 2 for r in results)
    # Check files were created
    assert len(list(save_dir.iterdir())) == 7


def test_extract_and_save_scenes_ome_tiff_specific_scenes(
    test_czi_image, tmp_path: Path
):
    """Test extracting specific scenes only."""
    from napari_ndev.widgets._utilities_container import (
        extract_and_save_scenes_ome_tiff,
    )

    path, _ = test_czi_image
    save_dir = tmp_path / 'ExtractedScenes'

    # Extract only scenes 0 and 2
    results = list(
        extract_and_save_scenes_ome_tiff(path, save_dir, scenes=[0, 2])
    )

    assert len(results) == 2
    assert len(list(save_dir.iterdir())) == 2


def test_open_images(make_napari_viewer, test_rgb_image):
    viewer = make_napari_viewer()
    container = UtilitiesContainer(viewer)

    path, _ = test_rgb_image
    container._files.value = path
    container.open_images()

    assert (
        container._dim_shape.value
        == 'T: 1, C: 1, Z: 1, Y: 1440, X: 1920, S: 3'
    )
    assert container._squeezed_dims_order == 'YX'
    assert container._channel_names.value == "['red', 'green', 'blue']"


def test_canvas_export_figure(make_napari_viewer, tmp_path: Path):
    viewer = make_napari_viewer()
    viewer.add_image(image_4d)
    container = UtilitiesContainer(viewer)

    container._save_directory.value = tmp_path
    container._save_name.value = 'test'

    container.canvas_export_figure()

    expected_save_loc = tmp_path / 'Figures' / 'test_figure.png'

    assert 'Exported canvas' in container._results.value
    assert expected_save_loc.exists()
    assert expected_save_loc.stat().st_size > 0

    # make sure properly detects 3D mode doesn't work
    viewer.dims.ndisplay = 3
    container.canvas_export_figure()
    assert 'Exporting Figure only works in 2D mode' in container._results.value


def test_canvas_screenshot(make_napari_viewer, tmp_path: Path):
    viewer = make_napari_viewer()
    viewer.add_image(image_4d)
    container = UtilitiesContainer(viewer)

    container._save_directory.value = tmp_path
    container._save_name.value = 'test'

    container.canvas_screenshot()

    expected_save_loc = tmp_path / 'Figures' / 'test_canvas.png'

    assert 'Exported screenshot of canvas' in container._results.value
    assert expected_save_loc.exists()
    assert expected_save_loc.stat().st_size > 0


def test_rescale_by(make_napari_viewer):
    viewer = make_napari_viewer()
    image_2d = np.random.random((10, 10))
    image_3d = np.random.random((10, 10, 10))

    layer_2d = viewer.add_image(image_2d)
    layer_3d = viewer.add_image(image_3d)

    container = UtilitiesContainer(viewer)
    container._scale_tuple.value = (5, 2, 3)

    viewer.layers.selection = [layer_2d, layer_3d]

    container.rescale_by()

    assert layer_2d.scale[0] == 2
    assert layer_2d.scale[1] == 3
    assert layer_3d.scale[0] == 5
    assert layer_3d.scale[1] == 2
    assert layer_3d.scale[2] == 3


def test_get_dims_for_shape_layer():
    container = UtilitiesContainer()
    container._squeezed_dims_order = 'YX'
    container._squeezed_dims = (20, 30)

    dims = container._get_dims_for_shape_layer()

    assert dims == (20, 30)
