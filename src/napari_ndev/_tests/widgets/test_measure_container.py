import pathlib

import numpy as np
from ndevio import nImage

from napari_ndev.widgets._measure_container import MeasureContainer


def test_widg_init_no_viewer():
    wdg = MeasureContainer()
    assert wdg._image_directory.label == 'Image directory'
    assert wdg._image_directory.value is None
    assert len(wdg._props_container) == len(wdg._sk_props)
    assert hasattr(wdg._prop, 'area')
    assert hasattr(wdg._prop, 'intensity_min')
    assert wdg.viewer is None


def test_widg_init_with_viewer(make_napari_viewer):
    viewer = make_napari_viewer()
    wdg = MeasureContainer(viewer)
    assert wdg.viewer == viewer


def test_get_0th_img_from_dir(tmp_path):
    image_directory = pathlib.Path(
        'src/napari_ndev/_tests/resources/Apoc/Images'
    )
    container = MeasureContainer()
    img, file_id = container._get_0th_img_from_dir(image_directory)

    assert img is not None
    assert isinstance(file_id, pathlib.Path)


def test_update_dim_and_scales():
    image_directory = pathlib.Path(
        'src/napari_ndev/_tests/resources/Apoc/Images'
    )
    file_name = 'SPF-4MM-22 slide 9-S6_Top Slide_TR2_p00_0_A01f00d0.tiff'
    container = MeasureContainer()
    img = nImage(image_directory / file_name)
    container._update_dim_and_scales(img)

    assert container._scale_tuple.value == (1.0, 0.2634, 0.2634)


def test_update_choices():
    container = MeasureContainer()
    image_directory = pathlib.Path(
        'src/napari_ndev/_tests/resources/Workflow/Images'
    )
    label_directory = pathlib.Path(
        'src/napari_ndev/_tests/resources/Workflow/Labels'
    )
    region_directory = pathlib.Path(
        'src/napari_ndev/_tests/resources/Workflow/ShapesAsLabels'
    )
    container._image_directory.value = image_directory
    container._label_directory.value = label_directory
    container._region_directory.value = region_directory

    container._update_choices(image_directory, 'Intensity')
    container._update_choices(label_directory, 'Labels', update_label=True)
    container._update_choices(region_directory, 'Region')

    # Check the choices in the label image ComboBox
    assert container._label_images.choices == ('Labels: Labels',)

    # Check the choices in the intensity images Select widget
    assert container._intensity_images.choices == (
        None,  # The default choice
        'Intensity: membrane',
        'Intensity: nuclei',
        'Labels: Labels',
        'Region: Shapes',
    )


def test_batch_measure_label_only(tmp_path):
    container = MeasureContainer()
    label_directory = pathlib.Path(
        'src/napari_ndev/_tests/resources/Workflow/Labels'
    )
    # make a dummy output folder
    output_folder = tmp_path / 'Output'
    output_folder.mkdir()

    container._label_directory.value = label_directory
    container._label_images.value = 'Labels: Labels'
    container._output_directory.value = output_folder
    df = container.batch_measure()

    assert output_folder.exists()
    assert (output_folder / 'measure_props_Labels.csv').exists()
    assert df is not None
    assert list(df.columns) == ['label_name', 'id', 'label', 'area']


# TODO: figure out why _intensity_images.value is not in index order, but alphabetical
def test_batch_measure_intensity(tmp_path):
    container = MeasureContainer()
    image_directory = pathlib.Path(
        'src/napari_ndev/_tests/resources/Workflow/Images'
    )
    label_directory = pathlib.Path(
        'src/napari_ndev/_tests/resources/Workflow/Labels'
    )
    region_directory = pathlib.Path(
        'src/napari_ndev/_tests/resources/Workflow/ShapesAsLabels'
    )
    # make a dummy output folder
    output_folder = tmp_path / 'Output'
    output_folder.mkdir()

    container._image_directory.value = image_directory
    container._label_directory.value = label_directory
    container._region_directory.value = region_directory
    container._scale_tuple.value = (3, 0.25, 0.25)
    container._prop.intensity_mean.value = True

    container._label_images.value = 'Labels: Labels'
    container._intensity_images.value = [
        'Region: Shapes',
        'Intensity: membrane',
        'Intensity: nuclei',
    ]
    container._output_directory.value = output_folder
    df = container.batch_measure()

    assert output_folder.exists()
    assert (output_folder / 'measure_props_Labels.csv').exists()
    assert df is not None
    assert list(df.columns) == [
        'label_name',
        'id',
        'label',
        'area',
        'intensity_mean-membrane',
        'intensity_mean-nuclei',
        'intensity_mean-Shapes',
    ]


def test_batch_measure_with_regex(tmp_path):
    container = MeasureContainer()
    label_directory = pathlib.Path(
        'src/napari_ndev/_tests/resources/Measure/Labels'
    )
    output_folder = tmp_path / 'Output'
    output_folder.mkdir()

    container._label_directory.value = label_directory
    container._label_images.value = 'Labels: DAPI'
    container._output_directory.value = output_folder
    container._id_regex_dict.value = r"""
        {
            'scene': r'(P\d{1,3}-\w+).ome',
            'well': r'-(\w+).ome',
            'HIC': r'(\d{1,3})HIC',
            'date': r'(\d{4}-\d{2}-\d{2})',
        }
    """
    container._update_tx_id_choices_button.clicked()
    assert container._tx_id.choices == (None, 'id','scene', 'well', 'HIC', 'date')

    container._tx_id.value = 'well'
    container._tx_n_well.value = 96
    container._tx_dict.value = r"""
        {
            'chelation':{
                'Control': ['B1:C12'],
                '50uM DFP': ['D1:E12'],
                '100uM DFP': ['F1:G12'],
                '100uM DFO': ['A1:A12', 'H1:H12'],
            },
        }
    """

    df = container.batch_measure()

    assert (output_folder / 'measure_props_DAPI.csv').exists()
    assert 'scene' in df.columns
    assert 'well' in df.columns
    assert 'chelation' in df.columns

def test_batch_measure_multiple_label_images(tmp_path):
    container = MeasureContainer()
    label_directory = pathlib.Path(
        'src/napari_ndev/_tests/resources/Measure/Labels'
    )
    output_folder = tmp_path / 'Output'
    output_folder.mkdir()

    container._label_directory.value = label_directory
    container._label_images.value = ['Labels: DAPI', 'Labels: Ferritin']
    container._output_directory.value = output_folder

    df = container.batch_measure()

    assert (output_folder / 'measure_props_DAPI_Ferritin.csv').exists()
    assert 'label_name' in df.columns
    assert np.array_equal(df['label_name'].unique(), ['DAPI', 'Ferritin'])

def test_group_measurements_no_agg_defaults():
    container = MeasureContainer()
    test_data_path = pathlib.Path(r'src/napari_ndev/_tests/resources/measure_props_Labels.csv')
    container._measured_data_path.value = test_data_path

    grouped_df = container.group_measurements()

    assert grouped_df is not None
    assert list(grouped_df.columns) == [('id', ''), ('label_count', 'label_1')]

def test_group_measurements_with_agg():
    container = MeasureContainer()
    test_data_path = pathlib.Path(r'src/napari_ndev/_tests/resources/measure_props_Labels.csv')
    container._measured_data_path.value = test_data_path
    container._grouping_cols.value = ['label_name', 'id', 'intensity_max-Labels']
    container._agg_cols.value = ['area']
    container._agg_funcs.value = ['mean']
    container._pivot_wider.value = False

    grouped_df = container.group_measurements()

    assert grouped_df is not None
    assert list(grouped_df.columns) == ['label_name', 'id', 'intensity_max-Labels', 'label_count', 'area_mean']
