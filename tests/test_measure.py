from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from bioio_base.types import ArrayLike

from napari_ndev.measure import (
    _convert_to_list,
    _extract_info_from_id_string,
    _generate_measure_dict,
    _map_tx_dict_to_df_id_col,
    _rename_intensity_columns,
    group_and_agg_measurements,
    measure_regionprops,
)


@pytest.mark.parametrize(
    ('arg', 'expected'),
    [
        (None, None),
        (1, [1]),
        ('string', ['string']),
        ([1, 2, 3], [1, 2, 3]),
        (np.array([1, 2, 3]), [np.array([1, 2, 3])]),
        (
            [np.array([1, 2]), np.array([3, 4])],
            [np.array([1, 2]), np.array([3, 4])],
        ),
    ],
)
def test_convert_to_list(
    arg: list | ArrayLike | str | None, expected: list | None
):
    result = _convert_to_list(arg)
    if isinstance(result, list):
        for res, exp in zip(result, expected):
            if isinstance(res, np.ndarray):
                assert np.array_equal(res, exp)
            else:
                assert res == exp
    else:
        assert result == expected


label1 = np.array([[1, 2], [3, 4]])
label2 = np.array([[5, 6], [7, 8]])
label_list = [label1, label2]
intensity1 = np.array([[5, 6], [7, 8]])
intensity2 = np.array([[9, 10], [11, 12]])
intensity_list = [intensity1, intensity2]


@pytest.mark.parametrize(
    (
        'label_images',
        'label_names',
        'intensity_images',
        'intensity_names',
        'expected',
    ),
    [
        (
            label1,
            None,
            None,
            None,
            {
                'label_images': label1,
                'label_names': ['label_0'],
                'intensity_images': None,
                'intensity_names': None,
            },
        ),
        (
            label1,
            None,
            intensity1,
            None,
            {
                'label_images': label1,
                'label_names': ['label_0'],
                'intensity_images': intensity1,
                'intensity_names': ['intensity_0'],
            },
        ),
        (
            label1,
            'custom_label',
            intensity1,
            'custom_intensity',
            {
                'label_images': label1,
                'label_names': ['custom_label'],
                'intensity_images': intensity1,
                'intensity_names': ['custom_intensity'],
            },
        ),
        (
            label1,
            'custom_label',
            None,
            None,
            {
                'label_images': label1,
                'label_names': ['custom_label'],
                'intensity_images': None,
                'intensity_names': None,
            },
        ),
        (
            label_list,
            ['label_c1', 'label_c2'],
            intensity_list,
            ['intensity_c1', 'intensity_c2'],
            {
                'label_images': label_list,
                'label_names': ['label_c1', 'label_c2'],
                'intensity_images': intensity_list,
                'intensity_names': ['intensity_c1', 'intensity_c2'],
            },
        ),
    ],
)
def test_generate_measure_dict(
    label_images, label_names, intensity_images, intensity_names, expected
):
    result = _generate_measure_dict(
        label_images, label_names, intensity_images, intensity_names
    )
    assert result['label_images'][0].all() == expected['label_images'][0].all()
    assert result['label_names'] == expected['label_names']
    assert isinstance(result['label_names'], list)
    # assert intensity images are the same, if they exist
    if result['intensity_images'] is not None:
        assert (
            result['intensity_images'][0].all()
            == expected['intensity_images'][0].all()
        )
        assert result['intensity_names'] == expected['intensity_names']
    else:
        assert result['intensity_images'] is None
        assert result['intensity_names'] is None


def test_extract_info_from_id_string():
    id_string = 'P14-A6__2024-07-16 25x 18HIC ncoa4 FT dapi obl 01'
    id_regex = {
        'scene': r'(P\d{1,3}-\w+)__',
        'well': r'-(\w+)__',
        'HIC': r'(\d{1,3})HIC',
        'exp': r'obl (\d{2,3})',
    }
    expected = {'scene': 'P14-A6', 'well': 'A6', 'HIC': '18', 'exp': '01'}
    result = _extract_info_from_id_string(id_string, id_regex)
    assert result == expected


def test_rename_intensity_columns():
    data = {
        'area': [100, 200],
        'intensity_mean-0': [0.5, 0.7],
        'intensity_mean-1': [0.8, 0.6],
    }
    df = pd.DataFrame(data)
    intensity_names = ['membrane', 'nuclei']
    result = _rename_intensity_columns(df, intensity_names)

    assert list(result.columns) == [
        'area',
        'intensity_mean-membrane',
        'intensity_mean-nuclei',
    ]


def test_map_tx_dict_to_df_id_col():
    tx = {
        'Treatment1': {'Condition1': ['A1', 'B2'], 'Condition2': ['C3']},
        'Treatment2': {'Condition3': ['D4:E5']},
    }
    tx_n_well = 96
    target_df = pd.DataFrame(
        {
            'well': ['A1', 'A2', 'A3', 'A4', 'A5', 'A6'],
            'value': [10, 20, 30, 40, 50, 60],
        }
    )
    id_column = 'well'
    result = _map_tx_dict_to_df_id_col(tx, tx_n_well, False, target_df, id_column)
    assert isinstance(result, pd.DataFrame)
    assert 'Treatment1' in result.columns
    assert 'Treatment2' in result.columns
    assert (
        result.loc[result['well'] == 'A1', 'Treatment1'].values[0]
        == 'Condition1'
    )


# Define test data
label_image_2d = np.array([[0, 0, 1], [0, 1, 1], [1, 1, 0]])
label_image_3d = np.array(
    [[[0, 0, 1], [0, 1, 1], [1, 1, 0]], [[0, 0, 1], [0, 1, 1], [1, 1, 0]]]
)
intensity_image_2d = np.array([[10, 10, 20], [10, 20, 20], [20, 20, 10]])
intensity_image_3d = np.array(
    [
        [[10, 10, 20], [10, 20, 20], [20, 20, 10]],
        [[10, 10, 20], [10, 20, 20], [20, 20, 10]],
    ]
)
id_string = 'P14-A6__2024-07-16 25x 18HIC ncoa4 FT dapi obl 01'
id_regex = {
    'well': r'-(\w+)__',
    'HIC': r'(\d{1,3})HIC',
    'exp': r'obl (\d{2,3})',
}


@pytest.mark.parametrize(
    (
        'label_images',
        'label_names',
        'intensity_images',
        'intensity_names',
        'properties',
        'scale',
        'id_string',
        'id_regex_dict',
        'save_data_path',
        'expected_columns',
    ),
    [
        # 2D label image, no intensity image
        (
            label_image_2d,
            None,
            None,
            None,
            ['area', 'eccentricity'],
            (1.0, 1.0),
            id_string,
            id_regex,
            None,
            ['id', 'well', 'HIC', 'exp', 'area', 'eccentricity'],
        ),
        # 2D label image, with intensity image
        (
            [label_image_2d],
            None,
            [intensity_image_2d],
            None,
            ['area', 'intensity_mean'],
            (1.0, 1.0),
            'test_id',
            None,
            None,
            ['id', 'area', 'intensity_mean'],
        ),
        # 2D label image, with 1 intensity images and names
        # TODO: Figure out why intensity_mean is not intensity_mean-0
        (
            [label_image_2d],
            None,
            [intensity_image_2d],
            ['test1'],
            ['area', 'intensity_mean'],
            (1.0, 1.0),
            'test_id',
            None,
            None,
            ['id', 'area', 'intensity_mean'],
        ),
        # 2D label image, with 2 intensity images and names
        (
            [label_image_2d],
            None,
            [intensity_image_2d, intensity_image_2d],
            ['test1', 'test2'],
            ['area', 'intensity_mean'],
            (1.0, 1.0),
            'test_id',
            None,
            None,
            ['id', 'area', 'intensity_mean-test1', 'intensity_mean-test2'],
        ),
        # 3D label image, no intensity image
        (
            [label_image_3d],
            None,
            None,
            None,
            ['area'],
            (1.0, 1.0, 1.0),
            'test_id',
            None,
            None,
            ['id', 'area'],
        ),
        # 3D label image, with intensity image
        (
            [label_image_3d],
            None,
            [intensity_image_3d],
            None,
            ['area', 'intensity_mean'],
            (1.0, 1.0, 1.0),
            'test_id',
            None,
            None,
            ['id', 'area', 'intensity_mean'],
        ),
    ],
)
def test_measure_regionprops(
    label_images,
    label_names,
    intensity_images,
    intensity_names,
    properties,
    scale,
    id_string,
    id_regex_dict,
    save_data_path,
    expected_columns,
):
    result_df = measure_regionprops(
        label_images=label_images,
        label_names=label_names,
        intensity_images=intensity_images,
        intensity_names=intensity_names,
        properties=properties,
        scale=scale,
        id_string=id_string,
        id_regex_dict=id_regex_dict,
        save_data_path=save_data_path,
    )
    # from skimage import measure
    # print(measure.regionprops_table(label_images[0], intensity_images[0], properties=properties))

    assert isinstance(result_df, pd.DataFrame)
    # Check if the DataFrame contains the expected columns
    assert all(column in result_df.columns for column in expected_columns)
    # Check if the id_string column contains the correct value
    assert (result_df['id'] == id_string).all()


def test_measure_regionprops_tx_dict():
    label_images = [label_image_2d]
    intensity_images = [intensity_image_2d]
    properties = ['area', 'intensity_mean']
    scale = (1.0, 1.0)
    id_string = 'P14-A6__2024-07-16 25x 18HIC ncoa4 FT dapi obl 01'
    id_regex_dict = {
        'well': r'-(\w+)__',
        'HIC': r'(\d{1,3})HIC',
        'exp': r'obl (\d{2,3})',
    }
    tx_id = 'well'
    tx_dict = {
        'Treatment1': {'Condition1': ['A1', 'B2'], 'Condition2': ['C3']},
        'Treatment2': {'Condition3': ['D4:E5']},
    }
    tx_n_well = 96
    result_df = measure_regionprops(
        label_images=label_images,
        intensity_images=intensity_images,
        properties=properties,
        scale=scale,
        id_string=id_string,
        id_regex_dict=id_regex_dict,
        tx_id=tx_id,
        tx_n_well=tx_n_well,
        tx_dict=tx_dict,
    )

    assert isinstance(result_df, pd.DataFrame)
    # Check if the DataFrame contains the expected columns
    assert all(
        column in result_df.columns
        for column in [
            'id',
            'area',
            'intensity_mean',
            'Treatment1',
            'Treatment2',
        ]
    )
    # Check if the id_string column contains the correct value
    assert (result_df['id'] == id_string).all()
    # Check if the treatments were assigned correctly
    assert (
        result_df.loc[result_df['well'] == 'A1', 'Treatment1'] == 'Condition1'
    ).all()
    assert (
        result_df.loc[result_df['well'] == 'B2', 'Treatment1'] == 'Condition1'
    ).all()
    assert (
        result_df.loc[result_df['well'] == 'C3', 'Treatment1'] == 'Condition2'
    ).all()
    assert (
        result_df.loc[result_df['well'] == 'D4', 'Treatment2'] == 'Condition3'
    ).all()
    assert (
        result_df.loc[result_df['well'] == 'E5', 'Treatment2'] == 'Condition3'
    ).all()
    assert (
        result_df.loc[result_df['well'] == 'E4', 'Treatment2'] == 'Condition3'
    ).all()


@pytest.fixture
def sample_data():
    return pd.DataFrame(
        {
            'id': ['id1', 'id1', 'id2', 'id2'],
            'label': [1, 2, 4, 5],
            'area': [100, 200, 300, 400],
            'intensity_mean': [0.5, 0.7, 0.8, 0.6],
        }
    )

def test_group_and_agg_measurements_sample_data(sample_data):

    result_df = group_and_agg_measurements(
        sample_data,
        grouping_cols=['id'],
        agg_cols=['area', 'intensity_mean'],
        agg_funcs=['mean', 'sum'],
    )

    assert isinstance(result_df, pd.DataFrame)
    assert all(
        column in result_df.columns
        for column in ['id', 'area_mean', 'area_sum', 'intensity_mean_mean', 'intensity_mean_sum']
    )
    assert result_df['area_mean'].tolist() == [150.0, 350.0]
    assert result_df['area_sum'].tolist() == [300, 700]
    assert result_df['intensity_mean_mean'].tolist() == [0.6, 0.7]
    assert result_df['intensity_mean_sum'].tolist() == [1.2, 1.4]

def test_group_and_agg_measurements_string_agg_func(sample_data):
    result_df = group_and_agg_measurements(
        sample_data,
        grouping_cols='id',
        agg_cols='area',
        agg_funcs='mean',
    )

    assert isinstance(result_df, pd.DataFrame)
    assert all(
        column in result_df.columns
        for column in ['id', 'area_mean']
    )

def test_group_and_agg_measurements_real_data():
    df = pd.read_csv('src/napari_ndev/_tests/resources/measure_props_Labels.csv')

    result_df = group_and_agg_measurements(
        df,
        grouping_cols=['id', 'intensity_max-Labels'],
        agg_cols=['area'],
        agg_funcs=['mean', 'std'],
    )

    assert isinstance(result_df, pd.DataFrame)
    assert all(
        column in result_df.columns
        for column in ['id', 'intensity_max-Labels', 'label_count', 'area_mean', 'area_std']
    )


def test_group_and_agg_measurements_no_agg(sample_data):
    result_df = group_and_agg_measurements(
        sample_data,
        grouping_cols=['id'],
        agg_cols=[],  # no aggregation
    )

    assert isinstance(result_df, pd.DataFrame)
    assert all(
        column in result_df.columns
        for column in ['id', 'label_count']
    )
