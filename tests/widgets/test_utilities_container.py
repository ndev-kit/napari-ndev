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
    make_napari_viewer, tmp_path: Path, test_data,
):
    test_image, test_shape, _, squeezed_dims = test_data

    viewer = make_napari_viewer()
    viewer.add_image(test_image)
    viewer.add_shapes(test_shape)
    container = UtilitiesContainer(viewer)

    container._viewer.layers.selection.active = viewer.layers['test_shape']
    container._save_directory.value = tmp_path
    container._save_name.value = 'test'

    shapes_layer = container.save_layers_as_ome_tiff()

    expected_save_loc = tmp_path / 'Shapes' / 'test.tiff'
    assert expected_save_loc.exists()
    assert shapes_layer.shape.__len__() == squeezed_dims.__len__() + 1
    assert nImage(expected_save_loc).channel_names == ['Shapes']


def test_save_labels(make_napari_viewer, tmp_path: Path, test_data):
    _, _, test_labels, squeezed_dims = test_data

    viewer = make_napari_viewer()
    viewer.add_labels(
        test_labels
    )  # <- should add a way to specify this is the selected layer in the viewer
    viewer.layers.selection.active = viewer.layers['test_labels']
    container = UtilitiesContainer(viewer)

    container._save_directory.value = tmp_path
    container._save_name.value = 'test'

    layer_data = container.save_layers_as_ome_tiff()

    expected_save_loc = tmp_path / 'Labels' / 'test.tiff'

    assert isinstance(layer_data, np.ndarray)
    assert expected_save_loc.exists()
    assert layer_data.shape.__len__() == squeezed_dims.__len__() + 1
    assert nImage(expected_save_loc).channel_names == ['Labels']


def test_save_image_layer(make_napari_viewer, test_data, tmp_path: Path):
    test_image, _, _, squeezed_dims = test_data
    viewer = make_napari_viewer()
    viewer.add_image(test_image)
    container = UtilitiesContainer(viewer)

    container._viewer.layers.selection.active = viewer.layers['test_image']
    container._channel_names.value = ['0']
    container._save_directory.value = tmp_path
    container._save_name.value = 'test'

    layer_data = container.save_layers_as_ome_tiff()

    expected_save_loc = tmp_path / 'Image' / 'test.tiff'

    assert isinstance(layer_data, np.ndarray)
    assert layer_data.shape.__len__() == squeezed_dims.__len__() + 1
    assert expected_save_loc.exists()
    assert nImage(expected_save_loc).channel_names == ['0']

def test_save_multi_layer(make_napari_viewer, test_data, tmp_path: Path):
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

    layer_data = container.save_layers_as_ome_tiff()

    expected_save_loc = tmp_path / 'Layers' / 'test.tiff'

    assert isinstance(layer_data, np.ndarray)
    assert layer_data.shape.__len__() == squeezed_dims.__len__() + 1
    assert expected_save_loc.exists()

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
    assert container._dim_shape.value == 'T: 1, C: 1, Z: 1, Y: 1440, X: 1920, S: 3'
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

def test_save_files_as_ome_tiff(test_czi_image, tmp_path: Path):
    path, _ = test_czi_image
    container = UtilitiesContainer()
    container._files.value = path
    container._save_directory.value = tmp_path
    save_dir = tmp_path / 'ConcatenatedImages'

    img_data = container.save_files_as_ome_tiff()

    # check that there is 1 file
    assert len(list(tmp_path.iterdir())) == 1
    # check the name of the file is 0T-4C-0Z-7pos.tiff
    assert (save_dir / '0T-4C-0Z-7pos.tiff').exists()
    assert img_data.shape.__len__() == 5

@pytest.mark.parametrize('num_files', [1,3])
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

def test_batch_concatenate_files(tmp_path: Path, resources_dir: Path):
    container = UtilitiesContainer()
    image_dir = resources_dir / 'test_czis'
    all_image_files = list(image_dir.iterdir())

    all_image_files = natsort.os_sorted(all_image_files)

    container._files.value = all_image_files[:1]

    container._save_directory.value = tmp_path
    container._save_directory_prefix.value = 'test'
    container.batch_concatenate_files()

    expected_output_dir = tmp_path / 'test_ConcatenatedImages'

    assert expected_output_dir.exists()
    assert len(list(expected_output_dir.iterdir())) == 8


def test_save_scenes_ome_tiff(test_czi_image, tmp_path: Path):
    path, _ = test_czi_image
    container = UtilitiesContainer()
    container._files.value = path
    container._save_directory.value = tmp_path
    save_dir = tmp_path / 'ExtractedScenes'

    container.save_scenes_ome_tiff()

    # check that there are 7 files in the save dir
    assert len(list(save_dir.iterdir())) == 7

def test_open_images(make_napari_viewer, test_rgb_image):
    viewer = make_napari_viewer()
    container = UtilitiesContainer(viewer)

    path, _ = test_rgb_image
    container._files.value = path
    container.open_images()

    assert container._dim_shape.value == "T: 1, C: 1, Z: 1, Y: 1440, X: 1920, S: 3"
    assert container._squeezed_dims_order == "YX"
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
