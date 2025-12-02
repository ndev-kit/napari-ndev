from __future__ import annotations

import ast
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from magicgui.widgets import (
    CheckBox,
    ComboBox,
    Container,
    FileEdit,
    LineEdit,
    ProgressBar,
    PushButton,
    Select,
    TextEdit,
    TupleEdit,
)
from qtpy.QtWidgets import QTabWidget

from napari_ndev import helpers

if TYPE_CHECKING:
    import pathlib

    import napari
    from bioio import BioImage


def group_and_save_measurements(
    measured_data_path: Path,
    grouping_cols: list[str],
    count_col: str | None,
    agg_cols: list[str],
    agg_funcs: list[str],
    pivot_wider: bool,
) -> Path:
    """Group measurements and save to CSV.

    Pure function for grouping measurements that can be run in a thread.

    Parameters
    ----------
    measured_data_path : Path
        Path to the measured data CSV file.
    grouping_cols : list[str]
        Columns to group by.
    count_col : str | None
        Column to count.
    agg_cols : list[str]
        Columns to aggregate.
    agg_funcs : list[str]
        Aggregation functions to apply.
    pivot_wider : bool
        Whether to pivot the data wider by label_name.

    Returns
    -------
    Path
        Path to the saved grouped measurements file.

    """
    from napari_ndev import measure as ndev_measure

    df = pd.read_csv(measured_data_path)

    grouped_df = ndev_measure.group_and_agg_measurements(
        df=df,
        grouping_cols=grouping_cols,
        count_col=count_col,
        agg_cols=agg_cols,
        agg_funcs=agg_funcs,
    )

    if pivot_wider and 'label_name' in grouping_cols:
        # get grouping cols without label name
        index_cols = [
            col for col in grouping_cols if col != 'label_name'
        ]

        # Only pivot if there are index columns to pivot on
        if index_cols:
            # pivot every values column that is not present in index or columns
            value_cols = [
                col
                for col in grouped_df.columns
                if col not in grouping_cols
            ]

            pivot_df = grouped_df.pivot(
                index=index_cols,
                columns='label_name',
                values=value_cols,
            )

            # reset index so that it is saved in the csv
            pivot_df.reset_index(inplace=True)

            grouped_df = pivot_df

    save_loc = (
        measured_data_path.parent
        / f'{measured_data_path.stem}_grouped.csv'
    )
    grouped_df.to_csv(save_loc, index=False)

    return save_loc


def measure_single_file(
    file: Path,
    label_dir: Path,
    image_dir: Path | None,
    region_dir: Path | None,
    label_channels: list[str],
    intensity_channels: list[str] | None,
    squeezed_dims: str,
    properties: list[str],
    props_scale: tuple,
    id_regex_dict: dict | None,
    tx_id: str | None,
    tx_dict: dict | None,
    tx_n_well: int | None,
) -> list[pd.DataFrame]:
    """Measure a single file across all its scenes.

    Parameters
    ----------
    file : Path
        The label file to process.
    label_dir : Path
        Directory containing label files.
    image_dir : Path | None
        Directory containing intensity images (optional).
    region_dir : Path | None
        Directory containing region images (optional).
    label_channels : list[str]
        List of label channel names to measure.
    intensity_channels : list[str] | None
        List of intensity channel names (with prefixes).
    squeezed_dims : str
        Dimension order string for image data extraction.
    properties : list[str]
        List of regionprops properties to measure.
    props_scale : tuple
        Physical pixel scale for measurements.
    id_regex_dict : dict | None
        Dictionary for ID regex parsing.
    tx_id : str | None
        Treatment ID column name.
    tx_dict : dict | None
        Treatment mapping dictionary.
    tx_n_well : int | None
        Number of wells for treatment mapping.

    Returns
    -------
    list[pd.DataFrame]
        List of measurement DataFrames, one per scene.
    """
    from ndevio import nImage

    from napari_ndev import measure as ndev_measure

    lbl = nImage(label_dir / file.name)
    id_string = helpers.create_id_string(lbl, file.stem)

    # Load optional images
    img = None
    reg = None

    if image_dir is not None:
        image_path = image_dir / file.name
        if not image_path.exists():
            raise FileNotFoundError(
                f'Image file {file.name} not found in intensity directory'
            )
        img = nImage(image_path)

    if region_dir is not None:
        region_path = region_dir / file.name
        if not region_path.exists():
            raise FileNotFoundError(
                f'Region file {file.name} not found in region directory'
            )
        reg = nImage(region_path)

    scene_results = []

    for scene_idx, scene in enumerate(lbl.scenes):
        lbl.set_scene(scene_idx)

        label_images = []
        label_names = []

        # iterate through each channel in the label image
        for label_chan in label_channels:
            # Remove 'Labels: ' prefix
            chan_name = label_chan[8:] if label_chan.startswith('Labels: ') else label_chan
            label_names.append(chan_name)

            lbl_C = lbl.channel_names.index(chan_name)
            label = lbl.get_image_data(squeezed_dims, C=lbl_C)
            label_images.append(label)

        intensity_images = []
        intensity_names = []

        # Get stack of intensity images if there are any selected
        if intensity_channels:
            for channel in intensity_channels:
                if channel.startswith('Labels: '):
                    chan = channel[8:]
                    lbl_C = lbl.channel_names.index(chan)
                    lbl.set_scene(scene_idx)
                    inten_img = lbl.get_image_data(squeezed_dims, C=lbl_C)
                elif channel.startswith('Intensity: '):
                    if img is None:
                        raise ValueError(
                            f"Intensity channel '{channel}' requested but no "
                            "intensity image directory was provided."
                        )
                    chan = channel[11:]
                    img_C = img.channel_names.index(chan)
                    img.set_scene(scene_idx)
                    inten_img = img.get_image_data(squeezed_dims, C=img_C)
                elif channel.startswith('Region: '):
                    if reg is None:
                        raise ValueError(
                            f"Region channel '{channel}' requested but no "
                            "region image directory was provided."
                        )
                    chan = channel[8:]
                    reg_C = reg.channel_names.index(chan)
                    reg.set_scene(scene_idx)
                    inten_img = reg.get_image_data(squeezed_dims, C=reg_C)
                else:
                    continue
                intensity_names.append(chan)
                intensity_images.append(inten_img)

            # the last dim is the multi-channel dim for regionprops
            intensity_stack = np.stack(intensity_images, axis=-1)
        else:
            intensity_stack = None
            intensity_names = None

        # Perform measurement
        measure_props_df = ndev_measure.measure_regionprops(
            label_images=label_images,
            label_names=label_names,
            intensity_images=intensity_stack,
            intensity_names=intensity_names,
            properties=properties,
            scale=props_scale,
            id_string=id_string,
            id_regex_dict=id_regex_dict,
            tx_id=tx_id,
            tx_dict=tx_dict,
            tx_n_well=tx_n_well,
            save_data_path=None,
        )

        scene_results.append(measure_props_df)

    return scene_results

class MeasureContainer(Container):
    """
    Widget to measure labels from folders.

    This class provides functionality to measure labels and compare them against intensity images, which can be microscopic images or other labels. It initializes various widgets and containers for user input and interaction, and connects events to handle user actions.

    Parameters
    ----------
    viewer : napari.viewer.Viewer
        The napari viewer instance. Optional.

    Attributes
    ----------
    viewer : napari.viewer.Viewer
        The napari viewer instance.

    _label_choices : list
        List of label choices.
    _intensity_choices : list
        List of intensity image choices.
    _p_sizes : None
        Placeholder for pixel sizes.
    _squeezed_dims : None
        Placeholder for squeezed dimensions.
    _prop : object
        Dynamic object to hold region properties checkboxes.
    _label_directory : FileEdit
        Widget for selecting label directory.
    _image_directory : FileEdit
        Widget for selecting image directory.
    _region_directory : FileEdit
        Widget for selecting region directory.
    _output_directory : FileEdit
        Widget for selecting output directory.
    _label_image : ComboBox
        Widget for selecting label image.
    _intensity_images : Select
        Widget for selecting intensity images.
    _scale_tuple : TupleEdit
        Widget for setting physical pixel sizes.
    _measure_button : PushButton
        Button to start measurement.
    _progress_bar : ProgressBar
        Progress bar to show measurement progress.
    _props_container : Container
        Container for region properties checkboxes.
    _sk_props : list
        List of region properties.
    _id_regex_container : Container
        Container for ID regex settings.
    _example_id_string : LineEdit
        Widget for example ID string.
    _id_regex_dict : TextEdit
        Widget for ID regex dictionary.
    _tx_map_container : Container
        Container for treatment map settings.
    _tx_id : LineEdit
        Widget for treatment ID.
    _tx_n_well : ComboBox
        Widget for number of wells.
    _tx_dict : TextEdit
        Widget for treatment dictionary.
    _grouping_container : Container
        Container for grouping settings.
    _create_grouped : CheckBox
        Checkbox to create grouped data.
    _group_by_sample_id : CheckBox
        Checkbox to group by sample ID.

    Methods
    -------
    _init_widgets()
        Initializes the widgets for user input.
    _init_regionprops_container()
        Initializes the container for region properties checkboxes.
    _init_id_regex_container()
        Initializes the container for ID regex settings.
    _init_tx_map_container()
        Initializes the container for treatment map settings.
    _init_grouping_container()
        Initializes the container for grouping settings.
    _init_layout()
        Initializes the layout of the container.
    _connect_events()
        Connects events to handle user actions.
    _get_0th_img_from_dir(directory)
        Gets the first image from a directory.
    _update_dim_and_scales(img)
        Updates the dimensions and scales based on the image.
    _update_choices(directory, prefix, update_label=False)
        Updates the choices for labels and intensity images.
    _update_image_choices()
        Updates the choices for intensity images.
    _update_label_choices()
        Updates the choices for label images.
    _update_region_choices()
        Updates the choices for region images.
    _safe_dict_eval(dict_string, dict_name=None)
        Safely evaluates a dictionary string.
    batch_measure()
        Performs batch measurement of labels and intensity images, and returns the measurement results as a DataFrame.

    """

    def __init__(
        self,
        viewer: napari.viewer.Viewer = None,
    ):
        """
        Initialize the MeasureContainer.

        Parameters
        ----------
        viewer : napari.viewer.Viewer
            The napari viewer instance. Optional.

        """
        super().__init__()

        self.viewer = viewer if viewer is not None else None
        self._label_choices = []
        self._intensity_choices = []
        self._p_sizes = None
        self._squeezed_dims = None
        self._prop = type('', (), {})()
        self._measure_results: list[pd.DataFrame] = []

        self._init_widgets()
        self._init_regionprops_container()
        self._init_id_regex_container()
        self._init_tx_map_container()
        self._init_grouping_container()
        self._init_layout()
        self._connect_events()
        self._init_batch_runner()

    def _init_widgets(self):
        """Initialize non-container widgets."""
        self._label_directory = FileEdit(label='Label directory', mode='d')
        self._image_directory = FileEdit(
            label='Image directory', mode='d', nullable=True
        )
        self._region_directory = FileEdit(
            label='Region directory', mode='d', nullable=True
        )
        self._output_directory = FileEdit(label='Output directory', mode='d')

        self._label_images = Select(
            label='Label image',
            choices=self._label_choices,
            allow_multiple=True,
            nullable=False,
            tooltip='Select label images to measure',
        )
        self._intensity_images = Select(
            label='Intensity images',
            choices=self._intensity_choices,
            allow_multiple=True,
            nullable=True,
            tooltip='Select intensity images to compare against labels',
        )
        self._scale_tuple = TupleEdit(
            value=(0.0000, 1.0000, 1.0000),
            label='Physical Pixel Sizes, ZYX',
            tooltip='Pixel size, usually in μm/px',
            options={'step': 0.0001},
        )
        self._measure_button = PushButton(label='Measure')

        self._progress_bar = ProgressBar(label='Progress:')

    def _init_regionprops_container(self):
        """Initialize the container for region properties checkboxes."""
        self._props_container = Container(layout='vertical')

        self._sk_props = [
            'label',
            'area',
            'area_convex',
            'bbox',
            'centroid',
            'eccentricity',
            'extent',
            'feret_diameter_max',
            'intensity_max',
            'intensity_mean',
            'intensity_min',
            'intensity_std',
            'orientation',
            'perimeter',
            'solidity',
        ]

        for feature in self._sk_props:
            setattr(self._prop, feature, CheckBox(label=feature))
            self._props_container.extend([getattr(self._prop, feature)])

        self._prop.label.value = True
        self._prop.area.value = True

    def _init_id_regex_container(self):
        """Initialize the container for ID regex settings."""
        self._id_regex_container = Container(layout='vertical')
        self._example_id_string = LineEdit(
            label='Example ID String',
            value=None,
            nullable=True,
        )
        self._id_regex_dict = TextEdit(
            label='ID Regex Dict',
            value='{\n\n}',
        )
        self._id_regex_container.extend(
            [self._example_id_string, self._id_regex_dict]
        )

    def _init_tx_map_container(self):
        """Initialize the container for treatment map settings."""
        self._tx_map_container = Container(layout='vertical')
        self._update_tx_id_choices_button = PushButton(
            label='Update Treatment ID Choices'
        )
        self._tx_id = ComboBox(
            label='Treatment ID',
            choices=['id'],
            value=None,
            nullable=True,
            tooltip='Usually, the treatment ID is the well ID or a unique identifier for each sample'
            "The treatment dict will be looked up against whatever this value is. If it is 'file', then will match against the filename",
        )
        self._tx_n_well = ComboBox(
            label='Number of Wells',
            value=None,
            choices=[6, 12, 24, 48, 96, 384],
            nullable=True,
            tooltip='By default, treatments must be verbosely defined for each condition and sample id '
            'If you have a known plate map, then selecting wells will allow a sparse treatment map to be passed to PlateMapper',
        )
        self._tx_dict = TextEdit(label='Treatment Dict', value='{\n\n}')
        # TODO: Add example treatment regex result widget when example id string or id regex dict is changed

        self._tx_map_container.extend(
            [
                self._update_tx_id_choices_button,
                self._tx_id,
                self._tx_n_well,
                self._tx_dict,
            ]
        )

    def _init_grouping_container(self):
        """Initialize the container for grouping settings."""
        self._grouping_container = Container(layout='vertical')

        self._measured_data_path = FileEdit(
            label='Measured Data Path',
            tooltip='Path to the measured data',
        )
        self._grouping_cols = Select(
            label='Grouping Columns',
            choices=[],
            allow_multiple=True,
            tooltip='Select columns to group the data by',
        )
        self._count_col = ComboBox(
            label='Count Column',
            choices=[],
            tooltip='Select column that will be counted',
        )
        self._agg_cols = Select(
            label='Aggregation Columns',
            choices=[],
            allow_multiple=True,
            nullable=True,
            value=None,
            tooltip='Select columns to aggregate with functions',
        )
        self._agg_funcs = Select(
            label='Aggregation Functions',
            choices=[
                'mean',
                'median',
                'std',
                'sem',
                'min',
                'max',
                'sum',
                'nunique',
            ],
            value=['mean'],
            allow_multiple=True,
            tooltip='Select functions performed on aggregation columns',
        )
        self._pivot_wider = CheckBox(label='Pivot Wider', value=True)
        self._group_measurements_button = PushButton(
            label='Group Measurements'
        )

        self._grouping_container.extend(
            [
                self._measured_data_path,
                self._grouping_cols,
                self._count_col,
                self._agg_cols,
                self._agg_funcs,
                self._pivot_wider,
                self._group_measurements_button,
            ]
        )

    def _init_layout(self):
        """Initialize the layout of the container."""
        self.extend(
            [
                self._label_directory,
                self._image_directory,
                self._region_directory,
                self._output_directory,
                self._label_images,
                self._intensity_images,
                self._scale_tuple,
                self._measure_button,
                self._progress_bar,
            ]
        )

        tabs = QTabWidget()
        tabs.addTab(self._props_container.native, 'Region Props')
        tabs.addTab(self._id_regex_container.native, 'ID Regex')
        tabs.addTab(self._tx_map_container.native, 'Tx Map')
        tabs.addTab(self._grouping_container.native, 'Grouping')
        self.native.layout().addWidget(tabs)

    def _connect_events(self):
        """Connect events to handle user actions."""
        self._image_directory.changed.connect(self._update_image_choices)
        self._label_directory.changed.connect(self._update_label_choices)
        self._region_directory.changed.connect(self._update_region_choices)
        self._update_tx_id_choices_button.clicked.connect(
            self._update_tx_id_choices
        )
        self._measure_button.clicked.connect(self._on_measure_button_clicked)
        self._measured_data_path.changed.connect(self._update_grouping_cols)
        self._group_measurements_button.clicked.connect(
            self.group_measurements
        )

    def _init_batch_runner(self) -> None:
        """Initialize the BatchRunner with callbacks."""
        from nbatch import BatchRunner

        self._batch_runner = BatchRunner(
            on_start=self._on_batch_start,
            on_item_complete=self._on_batch_item_complete,
            on_complete=self._on_batch_complete,
            on_error=self._on_batch_error,
        )

    def _on_batch_start(self, total: int) -> None:
        """Handle batch start to set up progress bar.

        Parameters
        ----------
        total : int
            Total number of items to process.

        """
        self._progress_bar.value = 0
        self._progress_bar.max = total
        self._progress_bar.label = f'Measuring {total} Images'

    def _on_batch_item_complete(
        self, result: list[pd.DataFrame], ctx
    ) -> None:
        """Handle completion of a single file measurement.

        Collects DataFrames from each file and increments progress.

        Parameters
        ----------
        result : list[pd.DataFrame]
            List of DataFrames (one per scene) from measuring the file.
        ctx : BatchContext
            Context containing item info and progress state.

        """
        if result:
            self._measure_results.extend(result)
        self._progress_bar.value = ctx.index + 1

    def _on_batch_complete(self) -> None:
        """Handle completion of all measurements.

        Concatenates all collected DataFrames, saves to CSV, and resets
        button state.

        """
        total = self._progress_bar.max
        errors = self._batch_runner.error_count
        try:
            if self._measure_results:
                # Concatenate all DataFrames
                measure_df = pd.concat(
                    self._measure_results, axis=0, ignore_index=True
                )

                # Get label names for filename
                label_names = [
                    (chan[8:] if chan.startswith('Labels: ') else chan)
                    for chan in self._label_images.value
                ]
                labels_string = '_'.join(label_names)

                # Save to CSV
                output_path = (
                    Path(self._output_directory.value)
                    / f'measure_props_{labels_string}.csv'
                )
                measure_df.to_csv(output_path, index=False)

            # Update progress bar label with completion status
            if errors > 0:
                self._progress_bar.label = (
                    f'Measured {total - errors} Images ({errors} Errors)'
                )
            else:
                self._progress_bar.label = f'Measured {total} Images'
        finally:
            self._set_measure_button_state(running=False)
            self._measure_results.clear()

    def _on_batch_error(self, ctx, error: Exception) -> None:
        """Handle error during measurement processing.

        Parameters
        ----------
        ctx : BatchContext
            Context containing item info and progress state.
        error : Exception
            The exception that occurred.

        """
        self._progress_bar.value = ctx.index + 1

    def _set_measure_button_state(self, running: bool) -> None:
        """Update measure button text and state.

        Parameters
        ----------
        running : bool
            Whether a batch measurement is currently running.

        """
        if running:
            self._measure_button.text = 'Cancel'
        else:
            self._measure_button.text = 'Measure'

    def _on_measure_button_clicked(self) -> None:
        """Handle measure button click for run/cancel toggling."""
        if self._batch_runner.is_running:
            self._batch_runner.cancel()
            self._set_measure_button_state(running=False)
        else:
            self.batch_measure()

    def _update_tx_id_choices(self):
        """Update the choices for treatment ID."""
        id_regex_dict = self._safe_dict_eval(self._id_regex_dict.value)
        if id_regex_dict is None:
            return
        # add the keys to a list which already contains 'id'
        regex_choices = list(id_regex_dict.keys())
        self._tx_id.choices = ['id'] + regex_choices

    def _update_grouping_cols(self):
        """Update the columns for grouping."""
        if self._measured_data_path.value is None:
            return

        df = pd.read_csv(self._measured_data_path.value)
        self._grouping_cols.choices = df.columns
        self._count_col.choices = df.columns
        self._agg_cols.choices = df.columns

        # set default value to label_name and id if exists
        grouping_cols = []
        if 'label_name' in df.columns:
            grouping_cols.append('label_name')
        if 'id' in df.columns:
            grouping_cols.append('id')
        self._grouping_cols.value = grouping_cols

        if 'label' in df.columns:
            self._count_col.value = 'label'

        return

    def _get_0th_img_from_dir(
        self, directory: str | None = None
    ) -> tuple[BioImage, pathlib.Path]:
        """Get the first image from a directory."""
        from ndevio import nImage

        _, files = helpers.get_directory_and_files(directory)
        return nImage(files[0]), files[0]

    def _update_dim_and_scales(self, img):
        """Update the dimensions and scales based on the image."""
        self._squeezed_dims = helpers.get_squeezed_dim_order(img)
        self._scale_tuple.value = (
            img.physical_pixel_sizes.Z or 1,
            img.physical_pixel_sizes.Y or 1,
            img.physical_pixel_sizes.X or 1,
        )

    def _update_choices(self, directory, prefix, update_label=False):
        """Update the choices for labels and intensity images."""
        img, _ = self._get_0th_img_from_dir(directory)
        img_channels = helpers.get_channel_names(img)
        img_channels = [f'{prefix}: {channel}' for channel in img_channels]

        if update_label:
            self._update_dim_and_scales(img)
            self._label_choices.extend(img_channels)
            self._label_images.choices = self._label_choices

        self._intensity_choices.extend(img_channels)
        self._intensity_images.choices = self._intensity_choices

    def _update_image_choices(self):
        """Update the choices for intensity images."""
        self._update_choices(self._image_directory.value, 'Intensity')

    def _update_label_choices(self):
        """Update the choices for label images."""
        self._update_choices(
            self._label_directory.value, 'Labels', update_label=True
        )
        img, file_id = self._get_0th_img_from_dir(self._label_directory.value)
        id_string = helpers.create_id_string(img, file_id.stem)
        self._example_id_string.value = id_string

    def _update_region_choices(self):
        """Update the choices for region images."""
        self._update_choices(self._region_directory.value, 'Region')

    def _safe_dict_eval(self, dict_string, dict_name=None):
        """Safely evaluate a string as a dictionary."""
        if dict_string is None:
            return None

        stripped_string = dict_string.strip()
        if stripped_string == '{}' or not stripped_string:
            return None
        try:
            return ast.literal_eval(stripped_string)
        except (ValueError, SyntaxError):
            return None

    def batch_measure(self) -> None:
        """
        Perform batch measurement of labels and intensity images.

        Use scikit-image's regionprops to measure properties of labels and
        intensity images. The measurements are saved to a CSV file in the
        output directory. Uses BatchRunner for threaded execution with
        progress tracking and cancellation support.

        """
        # get all the files in the label directory
        label_dir, label_files = helpers.get_directory_and_files(
            self._label_directory.value
        )
        image_dir, image_files = helpers.get_directory_and_files(
            self._image_directory.value
        )
        region_dir, region_files = helpers.get_directory_and_files(
            self._region_directory.value
        )

        # Validate files exist
        if not label_files:
            return

        # File count mismatch validation is handled by individual file errors
        # during batch processing - each missing file will raise FileNotFoundError

        # get the relevant spacing for regionprops, depending on length
        props_scale = self._scale_tuple.value
        props_scale = props_scale[-len(self._squeezed_dims) :]

        # get the properties list
        properties = [
            prop.label for prop in self._props_container if prop.value
        ]

        id_regex_dict = self._safe_dict_eval(
            self._id_regex_dict.value, 'ID Regex Dict'
        )
        tx_dict = self._safe_dict_eval(self._tx_dict.value, 'Tx Dict')

        # Get tx_n_well as int if provided
        tx_n_well = (
            int(self._tx_n_well.value) if self._tx_n_well.value else None
        )

        # Reset results collection
        self._measure_results = []

        # Setup logging
        log_file = self._output_directory.value / 'measure.log.txt'

        # Run BatchRunner with kwargs instead of partial - cleaner pattern
        self._set_measure_button_state(running=True)
        self._batch_runner.run(
            func=measure_single_file,
            items=label_files,
            label_dir=label_dir,
            image_dir=image_dir,
            region_dir=region_dir,
            label_channels=list(self._label_images.value),
            intensity_channels=(
                list(self._intensity_images.value)
                if self._intensity_images.value
                else None
            ),
            squeezed_dims=self._squeezed_dims,
            properties=properties,
            props_scale=props_scale,
            id_regex_dict=id_regex_dict,
            tx_id=self._tx_id.value,
            tx_dict=tx_dict,
            tx_n_well=tx_n_well,
            log_file=log_file,
            threaded=True,
        )

    def group_measurements(self):
        """Group measurements based on user input with threading.

        Uses the values in the Grouping Container of the Widget and passes them
        to the group_and_agg_measurements function in the measure module. The
        grouped measurements are saved to a CSV file in the same directory as
        the measured data with '_grouped' appended.

        Uses a background thread to avoid blocking the UI during processing.

        """
        from napari.qt import create_worker

        self._progress_bar.label = 'Grouping Measurements'
        self._progress_bar.value = 0
        self._progress_bar.max = 0  # indeterminate mode

        # Filter out None values from agg_cols
        agg_cols = [col for col in self._agg_cols.value if col is not None]

        self._group_worker = create_worker(
            group_and_save_measurements,
            measured_data_path=self._measured_data_path.value,
            grouping_cols=list(self._grouping_cols.value),
            count_col=self._count_col.value,
            agg_cols=agg_cols,
            agg_funcs=list(self._agg_funcs.value),
            pivot_wider=self._pivot_wider.value,
        )
        self._group_worker.returned.connect(self._on_group_complete)
        self._group_worker.errored.connect(self._on_group_error)
        self._group_worker.start()

    def _on_group_complete(self, save_loc: Path) -> None:
        """Handle completion of grouping measurements."""
        self._progress_bar.label = f'Grouped measurements saved to {save_loc.name}'
        self._progress_bar.max = 1
        self._progress_bar.value = 1

    def _on_group_error(self, exc: Exception) -> None:
        """Handle error during grouping measurements."""
        self._progress_bar.label = f'Error grouping: {exc}'
        self._progress_bar.max = 1
        self._progress_bar.value = 0
