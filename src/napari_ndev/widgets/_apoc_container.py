from __future__ import annotations

import os
import re
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from nbatch import BatchRunner
from magicclass.widgets import TabbedContainer
from magicgui.widgets import (
    CheckBox,
    ComboBox,
    Container,
    FileEdit,
    Label,
    LineEdit,
    ProgressBar,
    PushButton,
    RadioButtons,
    Select,
    SpinBox,
    Table,
)
from napari import layers

from napari_ndev import helpers

if TYPE_CHECKING:
    import napari


def get_channel_image(img, channel_index_list: list[int]) -> np.ndarray:
    """Get channel image data based on channel indices.

    Parameters
    ----------
    img : nImage
        The image object to extract channels from.
    channel_index_list : list[int]
        List of channel indices to extract.

    Returns
    -------
    np.ndarray
        The extracted channel image data.
    """
    if 'S' in img.dims.order:
        channel_img = img.get_image_data('TSZYX', S=channel_index_list)
    else:
        channel_img = img.get_image_data('TCZYX', C=channel_index_list)
    return channel_img


def train_on_file(
    image_file: Path,
    image_directory: Path,
    label_directory: Path,
    channel_index_list: list[int],
    classifier,
    feature_set: str,
) -> str:
    """Train classifier on a single image-label pair.

    Parameters
    ----------
    image_file : Path
        Path to the image file.
    image_directory : Path
        Directory containing image files.
    label_directory : Path
        Directory containing label files.
    channel_index_list : list[int]
        List of channel indices to use for training.
    classifier
        The APOC classifier instance to train.
    feature_set : str
        Feature set string for the classifier.

    Returns
    -------
    str
        Status message indicating success or failure.
    """
    from ndevio import nImage

    label_path = label_directory / image_file.name
    if not label_path.exists():
        raise FileNotFoundError(f'Label file missing for {image_file.name}')

    img = nImage(image_directory / image_file.name)
    channel_img = get_channel_image(img, channel_index_list)

    lbl = nImage(label_path)
    label = lbl.get_image_data('TCZYX', C=0)

    classifier.train(
        features=feature_set,
        image=np.squeeze(channel_img),
        ground_truth=np.squeeze(label),
        continue_training=True,
    )

    return f'Trained on {image_file.name}'


def predict_on_file(
    image_file: Path,
    output_directory: Path,
    channel_index_list: list[int],
    classifier,
) -> str:
    """Predict labels for a single image file.

    Parameters
    ----------
    image_file : Path
        Path to the image file to predict.
    output_directory : Path
        Directory to save the prediction output.
    channel_index_list : list[int]
        List of channel indices to use for prediction.
    classifier
        The APOC classifier instance to use for prediction.

    Returns
    -------
    str
        Status message indicating success.
    """
    from bioio.writers import OmeTiffWriter
    from ndevio import nImage

    from napari_ndev import helpers

    img = nImage(image_file)
    channel_img = get_channel_image(img, channel_index_list)
    squeezed_dim_order = helpers.get_squeezed_dim_order(img)

    result = classifier.predict(image=np.squeeze(channel_img))

    save_data = np.asarray(result)
    if save_data.max() > 65535:
        save_data = save_data.astype(np.int32)
    else:
        save_data = save_data.astype(np.int16)

    OmeTiffWriter.save(
        data=save_data,
        uri=output_directory / (image_file.stem + '.tiff'),
        dim_order=squeezed_dim_order,
        channel_names=['Labels'],
        physical_pixel_sizes=img.physical_pixel_sizes,
    )
    del result

    return f'Predicted {image_file.name}'


class ApocContainer(Container):
    """
    Container class for managing the ApocContainer widget in napari.

    Parameters
    ----------
    viewer : napari.viewer.Viewer
        The napari viewer instance.

    Attributes
    ----------
    _viewer : napari.viewer.Viewer
        The napari viewer instance.

    _image_directory : FileEdit
        Widget for selecting the image directory.

    _label_directory : FileEdit
        Widget for selecting the label directory.

    _output_directory : FileEdit
        Widget for selecting the output directory.

    _classifier_file : FileEdit
        Widget for selecting the classifier file.

    _classifier_type_mapping : dict
        Mapping of classifier types to their corresponding classes.

    _classifier_type : RadioButtons
        Widget for selecting the classifier type.

    _max_depth : SpinBox
        Widget for selecting the number of forests.

    _num_trees : SpinBox
        Widget for selecting the number of trees.

    _positive_class_id : SpinBox
        Widget for selecting the object label ID.

    _image_channels : Select
        Widget for selecting the image channels.

    _channel_order_label : Label
        Label widget for displaying the selected channel order.

    _PDFS : Enum
        Enum for predefined feature sets.

    _predefined_features : ComboBox
        Widget for selecting the features.

    _custom_features : LineEdit
        Widget for entering custom feature string.

    _open_custom_feature_generator : PushButton
        Button for opening the custom feature generator widget.

    _continue_training : CheckBox
        Checkbox for indicating whether to continue training.

    _batch_train_button : PushButton
        Button for training the classifier on image-label pairs.

    _batch_predict_button : PushButton
        Button for predicting labels with the classifier.

    _progress_bar : ProgressBar
        Progress bar widget.

    _image_layer : Select
        Widget for selecting the image layers.

    _label_layer : Widget
        Widget for selecting the label layers.

    _train_image_button : PushButton
        Button for training the classifier on selected layers using labels.

    _predict_image_layer : PushButton
        Button for predicting using the classifier on selected layers.

    _single_result_label : Label
        Label widget for displaying a single result.

    Methods
    -------
    _update_metadata_from_file()
        Update the metadata from the selected image directory.

    _update_channel_order()
        Update the channel order label based on the selected image channels.

    _set_value_from_pattern(pattern, content)
        Set the value from a pattern in the content.

    _process_classifier_metadata(content)
        Process the classifier metadata from the content.

    _update_classifier_metadata()
        Update the classifier metadata based on the selected classifier file.

    _classifier_statistics_table(custom_classifier)
        Display the classifier statistics table.

    _get_feature_set()
        Get the selected feature set.

    _get_training_classifier_instance()
        Get the training classifier instance based on the selected classifier
        type.

    _get_channel_image(img, channel_index_list)
        Get the channel image based on the selected channel index list.

    """

    def __init__(
        self,
        viewer: napari.viewer.Viewer = None,
    ):
        super().__init__(labels=False)
        self.min_width = 500  # TODO: remove this hardcoded value
        self._viewer = viewer if viewer is not None else None
        self._lazy_imports()
        self._initialize_cl_container()
        self._initialize_batch_container()
        self._initialize_viewer_container()
        self._initialize_custom_apoc_container()
        self._setup_widget_layout()
        self._connect_events()

    def _lazy_imports(self):
        import apoc

        self.apoc = apoc

    def _filter_layers(self, layer_type):
        # only do this if the viewer is not None
        if self._viewer is None:
            return []
        return [x for x in self._viewer.layers if isinstance(x, layer_type)]

    def _initialize_cl_container(self):
        self._classifier_file = FileEdit(
            label='Classifier File (.cl)',
            mode='w',
            tooltip='Create a .txt file and rename it to .cl ending.',
        )

        self._continue_training = CheckBox(
            label='Continue Training?',
            value=True,
            tooltip=(
                'Continue training only matters if classifier already exists.'
            ),
        )

        self._classifier_type_mapping = {
            'PixelClassifier': self.apoc.PixelClassifier,
            'ObjectSegmenter': self.apoc.ObjectSegmenter,
        }

        self._classifier_type = RadioButtons(
            label='Classifier Type',
            value='ObjectSegmenter',
            choices=['ObjectSegmenter', 'PixelClassifier'],
            tooltip='Object Segmenter is used for detecting objects of one '
            'class, including connected components. '
            'Pixel Classifier is used to classify pixel-types.',
        )
        self._max_depth = SpinBox(
            label='Num. of Forests',
            value=2,
            max=20,
            step=1,
            tooltip='Increases training time for each forest',
        )
        self._num_trees = SpinBox(
            label='Num. of Trees',
            value=100,
            step=50,
            tooltip='Increases computational requirements.',
        )
        self._positive_class_id = SpinBox(
            label='Object Label ID',
            value=2,
            step=1,
            tooltip='Only used with ObjectSegmenter, otherwise ignored.',
        )

        self._PDFS = Enum(
            'PDFS', self.apoc.PredefinedFeatureSet._member_names_
        )
        self._predefined_features = ComboBox(
            label='Features',
            choices=self._PDFS,
            nullable=True,
            value=None,
            tooltip="All featuresets except 'custom' are premade",
        )
        self._feature_string = LineEdit(
            label='Feature String',
            tooltip=(
                "A string in the form of 'filter1=radius1 filter2=radius2'."
            ),
        )
        self._cl_container = Container(
            widgets=[
                self._classifier_file,
                self._continue_training,
                self._classifier_type,
                self._max_depth,
                self._num_trees,
                self._positive_class_id,
                self._predefined_features,
                self._feature_string,
            ]
        )

    def _initialize_batch_container(self):
        self._image_directory = FileEdit(label='Image Directory', mode='d')
        self._label_directory = FileEdit(label='Label Directory', mode='d')
        self._output_directory = FileEdit(label='Output Directory', mode='d')

        self._image_channels = Select(
            label='Image Channels',
            choices=[],
            tooltip=(
                'Channel order should be same for training and prediction.'
            ),
        )
        self._channel_order_label = Label(value='Select an Image Channel!')

        self._batch_train_button = PushButton(label='Train')
        self._batch_predict_button = PushButton(label='Predict')

        self._batch_train_container = Container(
            layout='horizontal',
            widgets=[
                self._label_directory,
                self._batch_train_button,
            ],
        )

        self._batch_predict_container = Container(
            layout='horizontal',
            widgets=[
                self._output_directory,
                self._batch_predict_button,
            ],
        )

        self._progress_bar = ProgressBar(label='Progress:')

        self._batch_container = Container(
            layout='vertical',
            label='Batch',
            widgets=[
                self._image_directory,
                self._image_channels,
                self._channel_order_label,
                self._batch_train_container,
                self._batch_predict_container,
                self._progress_bar,
            ],
        )

        self._init_batch_runner()

    def _init_batch_runner(self):
        """Initialize the BatchRunner for batch operations."""
        self._batch_runner = BatchRunner(
            on_item_complete=self._on_batch_item_complete,
            on_complete=self._on_batch_complete,
            on_error=self._on_batch_error,
        )
        # Track which operation is running for button state management
        self._current_batch_operation = None
        # Track error count for completion message
        self._batch_error_count = 0

    def _on_batch_item_complete(self, result, ctx):
        """Callback when a batch item completes."""
        self._progress_bar.value = ctx.index + 1
        self._progress_bar.label = result

    def _on_batch_complete(self):
        """Callback when the entire batch completes."""
        total = self._progress_bar.max
        errors = self._batch_error_count
        if self._current_batch_operation == 'train':
            if errors > 0:
                self._progress_bar.label = (
                    f'Trained on {total - errors} Images ({errors} Errors)'
                )
            else:
                self._progress_bar.label = f'Trained on {total} Images'
            self._set_train_button_state(running=False)
            # Update classifier statistics after training
            if hasattr(self, '_training_classifier'):
                self._classifier_statistics_table(self._training_classifier)
        elif self._current_batch_operation == 'predict':
            if errors > 0:
                self._progress_bar.label = (
                    f'Predicted {total - errors} Images ({errors} Errors)'
                )
            else:
                self._progress_bar.label = f'Predicted {total} Images'
            self._set_predict_button_state(running=False)
        self._current_batch_operation = None
        self._batch_error_count = 0

    def _on_batch_error(self, ctx, exception):
        """Callback when a batch item fails."""
        self._batch_error_count += 1
        file_name = ctx.item.name if hasattr(ctx.item, 'name') else str(ctx.item)
        self._progress_bar.label = f'Error on {file_name}: {str(exception)[:50]}'

    def _set_train_button_state(self, running: bool):
        """Update train button appearance based on running state."""
        if running:
            self._batch_train_button.text = 'Cancel'
            self._batch_train_button.tooltip = 'Cancel training.'
        else:
            self._batch_train_button.text = 'Train'
            self._batch_train_button.tooltip = 'Train classifier on images.'

    def _set_predict_button_state(self, running: bool):
        """Update predict button appearance based on running state."""
        if running:
            self._batch_predict_button.text = 'Cancel'
            self._batch_predict_button.tooltip = 'Cancel prediction.'
        else:
            self._batch_predict_button.text = 'Predict'
            self._batch_predict_button.tooltip = 'Predict labels on images.'


    def _initialize_viewer_container(self):
        self._image_layers = Select(
            choices=self._filter_layers(layers.Image),
            label='Image Layers',
        )
        self._label_layer = ComboBox(
            choices=self._filter_layers(layers.Labels),
            label='Label Layer',
        )
        self._train_image_button = PushButton(
            label='Train classifier on selected layers using label'
        )
        self._predict_image_layer = PushButton(
            label='Predict using classifier on selected layers'
        )
        self._single_result_label = LineEdit()

        self._viewer_container = Container(
            widgets=[
                self._image_layers,
                self._label_layer,
                self._train_image_button,
                self._predict_image_layer,
                self._single_result_label,
            ],
            layout='vertical',
            label='Viewer',
        )

    def _initialize_custom_apoc_container(self):
        from napari_ndev.widgets._apoc_feature_stack import ApocFeatureStack

        self._custom_apoc_container = ApocFeatureStack(viewer=self._viewer)
        self._custom_apoc_container.label = 'Custom Feature Set'

    def _setup_widget_layout(self):
        self.append(self._cl_container)
        self._tabs = TabbedContainer(
            widgets=[
                self._batch_container,
                self._viewer_container,
                self._custom_apoc_container,
            ],
            label=None,
            labels=None,
        )
        # self.append(self._tabs) # does not connect gui to native, but is scrollable
        # self._scroll = ScrollableContainer(widgets=[self._tabs])
        # from qtpy.QtCore import Qt
        # self._scroll._widget._layout.setAlignment(Qt.AlignTop) # does not work
        # self.append(self._scroll)
        # the only way for _label_layer and _image_layers to stay connected is to attach it to native, not sure why
        self.native.layout().addWidget(
            self._tabs.native
        )  # connects and is scrollable, internally, but not in the main window
        self.native.layout().addStretch()  # resets the layout to squish to top

    def _connect_events(self):
        self._image_directory.changed.connect(self._update_metadata_from_file)
        self._image_channels.changed.connect(self._update_channel_order)
        self._classifier_file.changed.connect(self._update_classifier_metadata)

        self._batch_train_button.clicked.connect(self._on_train_button_clicked)
        self._batch_predict_button.clicked.connect(
            self._on_predict_button_clicked
        )
        self._train_image_button.clicked.connect(self.image_train)
        self._predict_image_layer.clicked.connect(self.image_predict)

        self._custom_apoc_container._generate_string_button.clicked.connect(
            self.insert_custom_feature_string
        )
        self._predefined_features.changed.connect(self._get_feature_set)

        # when self._viewer.layers is updated, update the choices in the ComboBox
        if self._viewer is not None:
            self._viewer.layers.events.removed.connect(
                self._update_layer_choices
            )
            self._viewer.layers.events.inserted.connect(
                self._update_layer_choices
            )

    def _on_train_button_clicked(self):
        """Handle train button click - either start or cancel."""
        if self._batch_runner.is_running:
            self._batch_runner.cancel()
            self._set_train_button_state(running=False)
        else:
            self.batch_train()

    def _on_predict_button_clicked(self):
        """Handle predict button click - either start or cancel."""
        if self._batch_runner.is_running:
            self._batch_runner.cancel()
            self._set_predict_button_state(running=False)
        else:
            self.batch_predict()

    def _update_layer_choices(self):
        self._label_layer.choices = self._filter_layers(layers.Labels)
        self._image_layers.choices = self._filter_layers(layers.Image)

    def _update_metadata_from_file(self):
        from ndevio import nImage

        _, files = helpers.get_directory_and_files(self._image_directory.value)
        img = nImage(files[0])
        self._image_channels.choices = helpers.get_channel_names(img)

    def _update_channel_order(self):
        self._channel_order_label.value = 'Selected Channel Order: ' + str(
            self._image_channels.value
        )

    ##############################
    # Classifier Related Functions
    ##############################
    def _set_value_from_pattern(self, pattern, content):
        match = re.search(pattern, content)
        return match.group(1) if match else None

    def _process_classifier_metadata(self, content):
        self._classifier_type.value = self._set_value_from_pattern(
            r'classifier_class_name\s*=\s*([^\n]+)', content
        )
        self._max_depth.value = self._set_value_from_pattern(
            r'max_depth\s*=\s*(\d+)', content
        )
        self._num_trees.value = self._set_value_from_pattern(
            r'num_trees\s*=\s*(\d+)', content
        )
        self._positive_class_id.value = (
            self._set_value_from_pattern(
                r'positive_class_identifier\s*=\s*(\d+)', content
            )
            or 2
        )

    def _update_classifier_metadata(self):
        file_path = self._classifier_file.value

        # create file, if it doesn't exist
        file_path.touch(exist_ok=True)
        content = file_path.read_text()

        # Ignore rest of function if file contents are empty
        if not content.strip():
            return

        self._process_classifier_metadata(content)

        if self._classifier_type.value in self._classifier_type_mapping:
            classifier_class = self._classifier_type_mapping[
                self._classifier_type.value
            ]
            custom_classifier = classifier_class(
                opencl_filename=self._classifier_file.value
            )
        else:
            custom_classifier = None

        self._classifier_statistics_table(custom_classifier)

    def _classifier_statistics_table(self, custom_classifier):
        table, _ = custom_classifier.statistics()

        trans_table = {'filter_name': [], 'radius': []}

        for value in table:
            filter_name, radius = (
                value.split('=') if '=' in value else (value, 0)
            )
            trans_table['filter_name'].append(filter_name)
            trans_table['radius'].append(float(radius))

        for i in range(len(next(iter(table.values())))):
            trans_table[str(i)] = [round(table[key][i], 2) for key in table]

        table_df = pd.DataFrame.from_dict(trans_table)
        if self._viewer is not None:
            self._viewer.window.add_dock_widget(
                Table(value=table_df),
                name=os.path.basename(self._classifier_file.value),
            )

    def _get_feature_set(self):
        if self._predefined_features.value.value == 1:
            feature_set = ''
        else:
            feature_set = self.apoc.PredefinedFeatureSet[
                self._predefined_features.value.name
            ].value
        self._feature_string.value = feature_set
        self._custom_apoc_container._feature_string.value = (
            feature_set  # <- potentially deprecated in future
        )
        return feature_set

    def _get_training_classifier_instance(self):
        if self._classifier_type.value == 'PixelClassifier':
            return self.apoc.PixelClassifier(
                opencl_filename=self._classifier_file.value,
                max_depth=self._max_depth.value,
                num_ensembles=self._num_trees.value,
            )

        if self._classifier_type.value == 'ObjectSegmenter':
            return self.apoc.ObjectSegmenter(
                opencl_filename=self._classifier_file.value,
                positive_class_identifier=self._positive_class_id.value,
                max_depth=self._max_depth.value,
                num_ensembles=self._num_trees.value,
            )
        return None

    ##############################
    # Training and Prediction
    ##############################
    def batch_train(self):
        """Train classifier on batch of image-label pairs using BatchRunner."""
        from functools import partial

        from pyclesperanto import wait_for_kernel_to_finish

        image_directory, image_files = helpers.get_directory_and_files(
            self._image_directory.value
        )
        label_directory, _ = helpers.get_directory_and_files(
            self._label_directory.value
        )

        # https://github.com/clEsperanto/pyclesperanto_prototype/issues/163
        wait_for_kernel_to_finish()

        self._progress_bar.label = f'Training on {len(image_files)} Images'
        self._progress_bar.value = 0
        self._progress_bar.max = len(image_files)

        if not self._continue_training.value:
            self.apoc.erase_classifier(self._classifier_file.value)

        custom_classifier = self._get_training_classifier_instance()
        # Store classifier reference for statistics display after completion
        self._training_classifier = custom_classifier
        feature_set = self._feature_string.value

        channel_index_list = [
            self._image_channels.choices.index(channel)
            for channel in self._image_channels.value
        ]

        # Create partial function with fixed parameters
        train_func = partial(
            train_on_file,
            image_directory=image_directory,
            label_directory=label_directory,
            channel_index_list=channel_index_list,
            classifier=custom_classifier,
            feature_set=feature_set,
        )

        self._current_batch_operation = 'train'
        self._batch_error_count = 0
        self._set_train_button_state(running=True)

        self._batch_runner.run(
            func=train_func,
            items=image_files,
            threaded=True,
            log_file=self._classifier_file.value.with_suffix('.log.txt'),
        )

    def _get_prediction_classifier_instance(self):
        if self._classifier_type.value in self._classifier_type_mapping:
            classifier_class = self._classifier_type_mapping[
                self._classifier_type.value
            ]
            return classifier_class(
                opencl_filename=self._classifier_file.value
            )
        return None

    def batch_predict(self):
        """Predict labels on batch of images using BatchRunner."""
        from functools import partial

        from pyclesperanto import wait_for_kernel_to_finish

        _, image_files = helpers.get_directory_and_files(
            dir_path=self._image_directory.value,
        )

        # https://github.com/clEsperanto/pyclesperanto_prototype/issues/163
        wait_for_kernel_to_finish()

        self._progress_bar.label = f'Predicting {len(image_files)} Images'
        self._progress_bar.value = 0
        self._progress_bar.max = len(image_files)

        custom_classifier = self._get_prediction_classifier_instance()

        channel_index_list = [
            self._image_channels.choices.index(channel)
            for channel in self._image_channels.value
        ]

        # Create partial function with fixed parameters
        predict_func = partial(
            predict_on_file,
            output_directory=self._output_directory.value,
            channel_index_list=channel_index_list,
            classifier=custom_classifier,
        )

        self._current_batch_operation = 'predict'
        self._batch_error_count = 0
        self._set_predict_button_state(running=True)

        self._batch_runner.run(
            func=predict_func,
            items=image_files,
            threaded=True,
            log_file=self._output_directory.value / 'batch_predict.log.txt',
        )

    def image_train(self):
        from pyclesperanto import wait_for_kernel_to_finish

        image_names = [image.name for image in self._image_layers.value]
        label_name = self._label_layer.value.name
        self._single_result_label.value = (
            f'Training on {image_names} using {label_name}'
        )

        image_list = [image.data for image in self._image_layers.value]
        image_stack = np.stack(image_list, axis=0)
        label = self._label_layer.value.data

        # https://github.com/clEsperanto/pyclesperanto_prototype/issues/163
        wait_for_kernel_to_finish()

        if not self._continue_training:
            self.apoc.erase_classifier(self._classifier_file.value)

        custom_classifier = self._get_training_classifier_instance()
        feature_set = self._feature_string.value

        custom_classifier.train(
            features=feature_set,
            image=np.squeeze(image_stack),
            ground_truth=np.squeeze(label),
            continue_training=True,
        )

        self._single_result_label.value = (
            f'Trained on {image_names} using {label_name}'
        )

    def image_predict(self):
        from pyclesperanto import wait_for_kernel_to_finish

        # https://github.com/clEsperanto/pyclesperanto_prototype/issues/163
        wait_for_kernel_to_finish()

        image_names = [image.name for image in self._image_layers.value]
        self._single_result_label.value = f'Predicting {image_names}'
        image_list = [image.data for image in self._image_layers.value]
        image_stack = np.stack(image_list, axis=0)
        scale = self._image_layers.value[0].scale

        custom_classifier = self._get_prediction_classifier_instance()

        result = custom_classifier.predict(image=np.squeeze(image_stack))

        # sometimes, input layers may have shape with 1s, like (1,1,10,10)
        # however, we are squeezing the input, so the reuslt will have shape
        # (10,10), and therefore scale needs to accomodate dropped axes
        result_dims = result.ndim
        if len(scale) > result_dims:
            scale = scale[-result_dims:]

        self._viewer.add_labels(
            result,
            scale=scale,
            name=f'{self._classifier_file.value.stem} :: {image_names}',
        )

        self._single_result_label.value = f'Predicted {image_names}'

        return result

    def insert_custom_feature_string(self):
        self._feature_string.value = (
            self._custom_apoc_container._feature_string.value
        )
        return self._feature_string.value
