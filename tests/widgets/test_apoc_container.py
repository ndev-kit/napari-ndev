import logging
import os
import pathlib
import tempfile
from unittest.mock import patch

import numpy as np
import pytest

from napari_ndev.widgets._apoc_container import ApocContainer


def test_update_channel_order(make_napari_viewer):
    viewer = make_napari_viewer()

    wdg = ApocContainer(viewer)
    wdg._image_channels.choices = ['C0', 'C1', 'C2', 'C3']
    wdg._image_channels.value = ['C1', 'C3']
    wdg._update_channel_order()
    assert (
        wdg._channel_order_label.value
        == "Selected Channel Order: ['C1', 'C3']"
    )


@pytest.fixture
def dummy_classifier_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        classifier_file_path = os.path.join(tmpdir, 'dummy_classifier.cl')
        with open(classifier_file_path, 'w') as f:
            f.write(
                'OpenCL RandomForestClassifier\n'
                'classifier_class_name = PixelClassifier\n'
                'num_ground_truth_dimensions = 2\n'
                'num_classes = 3\n'
                'max_depth = 5\n'
                'num_trees = 100\n'
                # "positive_class_identifier = 2\n"
            )
        yield classifier_file_path


def test_update_classifier_metadata(make_napari_viewer, dummy_classifier_file):
    viewer = make_napari_viewer()
    wdg = ApocContainer(viewer)

    num_widgets = len(viewer.window._dock_widgets)
    # This automatically calls wdg._update_classifier_metadata() because of
    # wdg ._classifier_file.changed.connect
    wdg._classifier_file.value = dummy_classifier_file

    # additional widget because of calling wdg._classifier_statistics_table()
    assert len(viewer.window._dock_widgets) == 1 + num_widgets
    assert wdg._classifier_type.value == 'PixelClassifier'
    # assert wdg._classifier_channels.value == "Trained on 3 Channels"
    assert wdg._max_depth.value == 5
    assert wdg._num_trees.value == 100
    assert wdg._positive_class_id.value == 2


image_2d = np.asarray([[0, 0, 1, 1], [0, 0, 1, 1], [2, 2, 1, 1], [2, 2, 1, 1]])
shapes_2d = np.array([[0.25, 0.25], [0.25, 2.75], [2.75, 2.75], [2.75, 0.25]])
labels_2d = np.asarray(
    [[0, 0, 0, 1], [0, 0, 1, 0], [0, 2, 1, 0], [2, 2, 0, 0]]
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


@pytest.fixture
def empty_classifier_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        classifier_file_path = os.path.join(tmpdir, 'empty_classifier.cl')
        with open(classifier_file_path, 'w') as f:
            f.write('')
        yield classifier_file_path


@pytest.mark.skip_ci
def test_image_train(make_napari_viewer, test_data, empty_classifier_file):
    viewer = make_napari_viewer()
    test_image, _, test_labels, _ = test_data
    viewer.add_image(test_image)
    viewer.add_labels(test_labels)

    wdg = ApocContainer(viewer)

    wdg._image_layers.value = [viewer.layers['test_image']]
    wdg._label_layer.value = viewer.layers['test_labels']
    wdg._classifier_type.value = 'ObjectSegmenter'
    wdg._continue_training.value = False
    wdg._classifier_file.value = empty_classifier_file
    wdg._positive_class_id.value = 2
    wdg._max_depth.value = 2
    wdg._num_trees.value = 50

    PDFS = wdg._PDFS
    wdg._predefined_features.value = PDFS.small_quick

    wdg.image_train()
    with open(empty_classifier_file) as f:
        result_classifier = f.read()

    assert (
        wdg._single_result_label.value
        == "Trained on ['test_image'] using test_labels"
    )
    # check classifier contents after wdg.image_train()
    assert 'ObjectSegmenter' in result_classifier
    assert 'num_trees = 50' in result_classifier


@pytest.fixture
def trained_classifier_file(
    make_napari_viewer,
    test_data,
    empty_classifier_file,
):
    test_image_train(make_napari_viewer, test_data, empty_classifier_file)
    return empty_classifier_file


@pytest.mark.skip_ci
def test_image_predict(make_napari_viewer, test_data, trained_classifier_file):
    import pyclesperanto as cle

    viewer = make_napari_viewer()
    test_image, _, _, _ = test_data
    viewer.add_image(test_image)

    wdg = ApocContainer(viewer)

    wdg._image_layers.value = [viewer.layers['test_image']]
    wdg._classifier_file.value = trained_classifier_file

    result = wdg.image_predict()
    expected_layer_name = "empty_classifier :: ['test_image']"

    assert wdg._single_result_label.value == "Predicted ['test_image']"
    assert wdg._viewer.layers[expected_layer_name].visible
    assert np.asarray(cle.pull(result)).any()
    assert np.asarray(
        cle.pull(wdg._viewer.layers[expected_layer_name].data)
    ).any()


@pytest.mark.skip_ci
# def test_batch_predict_normal_operation(make_napari_viewer, tmp_path):
def test_batch_predict_normal_operation(tmp_path, qtbot):
    image_directory = pathlib.Path('tests/resources/Apoc/Images')
    num_files = len(list(image_directory.glob('*.tiff')))
    output_directory = tmp_path / 'output'
    output_directory.mkdir()

    classifier = pathlib.Path(
        'tests/resources/Apoc/Classifiers/newlabels_pixel_classifier.cl'
    )

    # Create an instance of ApocContainer
    # container = ApocContainer(make_napari_viewer())
    container = ApocContainer()
    container._image_directory.value = image_directory
    container._output_directory.value = output_directory
    # container._image_channels.value = ["IBA1"] # images need fixed
    container._image_channels.value = ['Labels']
    container._classifier_file.value = classifier

    container.batch_predict()

    # Wait for threaded batch to complete
    qtbot.waitUntil(
        lambda: not container._batch_runner.is_running, timeout=60000
    )

    # Check if the loop completes without exceptions
    assert container._progress_bar.value == num_files
    assert container._progress_bar.label == f'Predicted {num_files} Images'


def test_update_metadata_from_file():
    # Create an instance of ApocContainer
    wdg = ApocContainer()

    # Mock the get_directory_and_files function to return a sample file
    with patch(
        'napari_ndev.helpers.get_directory_and_files'
    ) as mock_get_directory_and_files:
        mock_get_directory_and_files.return_value = (
            '/path/to/directory',
            ['/path/to/file.tif'],
        )

        # Mock the nImage class to return sample channel names
        with patch('ndevio.nImage') as nImage:
            nImage.return_value.channel_names = ['C0', 'C1', 'C2']

            # Call the _update_metadata_from_file method
            wdg._update_metadata_from_file()

            # Check if the image channels choices are updated correctly
            assert list(wdg._image_channels.choices) == ['C0', 'C1', 'C2']


@pytest.mark.skip_ci
def test_batch_predict_exception_logging(tmp_path, qtbot):
    image_directory = pathlib.Path('tests/resources/Apoc/Images')

    num_files = len(list(image_directory.glob('*.tiff')))
    output_directory = tmp_path / 'output'
    output_directory.mkdir()

    # Create an instance of ApocContainer
    container = ApocContainer()
    container._image_directory.value = image_directory
    container._output_directory.value = output_directory
    # container._image_channels.value = ["IBA1"] # fix these images
    container._image_channels.value = ['Labels']

    # Create a custom exception class
    class CustomException(Exception):
        pass

    # Mock the custom_classifier.predict() method to raise the custom exception
    class MockClassifier:
        def predict(self, image):
            raise CustomException('Test exception')

    container._get_prediction_classifier_instance = lambda: MockClassifier()

    # Call the batch_predict() method
    container.batch_predict()

    # Wait for threaded batch to complete
    qtbot.waitUntil(
        lambda: not container._batch_runner.is_running, timeout=60000
    )

    # Check if error is reflected in progress bar
    assert 'Error' in container._progress_bar.label


def test_batch_train_button_toggle(qtbot):
    """Test that train button toggles between Train and Cancel states."""
    container = ApocContainer()

    # Initial state should be 'Train'
    assert container._batch_train_button.text == 'Train'

    # Set to running state
    container._set_train_button_state(running=True)
    assert container._batch_train_button.text == 'Cancel'

    # Set back to not running
    container._set_train_button_state(running=False)
    assert container._batch_train_button.text == 'Train'


def test_batch_predict_button_toggle(qtbot):
    """Test that predict button toggles between Predict and Cancel states."""
    container = ApocContainer()

    # Initial state should be 'Predict'
    assert container._batch_predict_button.text == 'Predict'

    # Set to running state
    container._set_predict_button_state(running=True)
    assert container._batch_predict_button.text == 'Cancel'

    # Set back to not running
    container._set_predict_button_state(running=False)
    assert container._batch_predict_button.text == 'Predict'


def test_batch_error_callback(qtbot):
    """Test that error callback updates progress bar label."""
    from unittest.mock import MagicMock

    from nbatch import BatchContext

    container = ApocContainer()

    # Create a mock context with a file item
    mock_item = MagicMock()
    mock_item.name = 'bad_file.tiff'
    ctx = MagicMock(spec=BatchContext)
    ctx.item = mock_item

    test_exception = ValueError('Test error message')

    container._on_batch_error(ctx, test_exception)

    # Verify progress bar label was updated with error info
    assert 'Error on bad_file.tiff' in container._progress_bar.label
    assert 'Test error message' in container._progress_bar.label
