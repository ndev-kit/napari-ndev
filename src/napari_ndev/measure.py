"""
Functions for measuring properties of labels.

Measure properties of labels in images using sci-kit image's regionprops.
It includes utilities for handling label and intensity images,
extracting information from ID strings, renaming intensity columns,
and mapping treatment dictionaries to DataFrame ID columns.

Functions
---------
measure_regionprops : Measure properties of labels with sci-kit image regionprops.
group_and_agg_measurements : Count and aggregate measurements by grouping IDs from measurement results.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from napari_ndev._plate_mapper import PlateMapper

if TYPE_CHECKING:
    from bioio_base.types import ArrayLike, PathLike

__all__ = ['group_and_agg_measurements', 'measure_regionprops']


def measure_regionprops(
    label_images: list[ArrayLike] | ArrayLike,
    label_names: list[str] | str | None = None,
    intensity_images: list[ArrayLike] | ArrayLike | None = None,
    intensity_names: list[str] | str | None = None,
    properties: list[str] | None = None,
    scale: tuple[float, float] | tuple[float, float, float] = (1, 1),
    id_string: str | None = None,
    id_regex_dict: dict | None = None,
    tx_id: str | None = None,
    tx_dict: dict | None = None,
    tx_n_well: int | None = None,
    tx_leading_zeroes: bool = False,
    save_data_path: PathLike = None,
) -> pd.DataFrame:
    """
    Measure properties of labels with sci-kit image regionprops.

    Optionally give a list of intensity_images to measure intensity properties
    of labels (i.e. 'intensity_mean', 'intensity_min', 'intensity_max',
    'intensity_std'). If no label or intensity names are given, the names are
    automatically generated as a string of the input variable name.
    Choose from a list of properties to measure: [
            "label",
            "area",
            "area_convex",
            "bbox",
            "centroid",
            "eccentricity",
            "extent",
            "feret_diameter_max",
            "intensity_max",
            "intensity_mean",
            "intensity_min",
            "intensity_std",
            "num_pixels",
            "orientation",
            "perimeter",
            "solidity",
        ].

    Parameters
    ----------
    label_images : list of ArrayLike or ArrayLike
        The label images.
    label_names : list of str or str or None, optional
        The names of the label images.
    intensity_images : list of ArrayLike or ArrayLike or None, optional
        The intensity images.
    intensity_names : list of str or str or None, optional
        The names of the intensity images.
    properties : list of str or None, optional
        The properties to measure.
    scale : tuple of float, optional
        The scale for the measurements.
    id_string : str or None, optional
        The ID string.
    id_regex_dict : dict or None, optional
        The regex dictionary for extracting information from the ID string.
    tx_id : str or None, optional
        The treatment ID.
    tx_dict : dict or None, optional
        The treatment dictionary.
    tx_n_well : int or None, optional
        The number of wells in the plate.
    tx_leading_zeroes : bool, optional
        Whether to use leading zeroes in the plate map.
    save_data_path : PathLike or None, optional
        The path to save the data.

    Returns
    -------
    pd.DataFrame
        The DataFrame with measured properties.

    """
    from skimage import measure

    if properties is None:
        properties = ['area']
    measure_dict = _generate_measure_dict(
        label_images, label_names, intensity_images, intensity_names
    )

    if intensity_images is not None:
        if len(measure_dict['intensity_images']) == 1:
            intensity_stack = measure_dict['intensity_images'][0]
        else:
            intensity_stack = np.stack(
                measure_dict['intensity_images'], axis=-1
            )
    else:
        intensity_stack = None

    measure_df_list = []

    for label_idx, label_image in enumerate(measure_dict['label_images']):
        measure_props = measure.regionprops_table(
            label_image=label_image,
            intensity_image=intensity_stack,
            properties=properties,
            spacing=scale,
        )

        measure_df = pd.DataFrame(measure_props)
        measure_df.insert(
            0, 'label_name', measure_dict['label_names'][label_idx]
        )
        measure_df_list.append(measure_df)

    if len(measure_df_list) > 1:
        measure_df = pd.concat(measure_df_list, ignore_index=True)

    if intensity_names is not None:
        measure_df = _rename_intensity_columns(
            measure_df, measure_dict['intensity_names']
        )

    measure_df.insert(1, 'id', id_string)

    if id_regex_dict is not None:
        id_dict = _extract_info_from_id_string(id_string, id_regex_dict)
        for key, value in id_dict.items():
            measure_df.insert(2, key, value)

    if tx_id is not None and tx_dict is not None:
        _map_tx_dict_to_df_id_col(
            tx_dict, tx_n_well, tx_leading_zeroes, measure_df, tx_id
        )

    if save_data_path is not None:
        measure_df.to_csv(save_data_path, index=False)

    return measure_df


def group_and_agg_measurements(
    df: pd.DataFrame,
    grouping_cols: str | list[str] = 'id',
    count_col: str = 'label',
    agg_cols: str | list[str] | None = None,
    agg_funcs: str | list[str] = 'mean',
) -> pd.DataFrame:
    """
    Count and aggregate measurements by grouping IDs from measurement results.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame with measurement properties, usually from measure_regionprops.
    grouping_cols : str or list of str, optional
        The columns to group by. By default, just the image ID.
    count_col : str, optional
        The column to count. By default, just the 'label' column.
    agg_cols : list of str or None, optional
        The columns to aggregate. By default, None.
    agg_funcs : str or list of str, optional
        The aggregating functions. By default, just the mean.

    Returns
    -------
    pd.DataFrame
        The DataFrame with grouped and aggregated measurements.

    """
    grouping_cols = _convert_to_list(grouping_cols)
    agg_cols = _convert_to_list(agg_cols)
    agg_funcs = _convert_to_list(agg_funcs)

    # get count data
    df_count = (
        df.copy()
        .groupby(grouping_cols)
        .agg({count_col: 'count'})  # counts count_col
        .rename(columns={count_col: f'{count_col}_count'})
        .reset_index()
    )

    if agg_cols is None or agg_cols == []:
        return df_count

    # get aggregated data
    agg_cols = df[agg_cols]
    agg_dict = dict.fromkeys(agg_cols, agg_funcs)
    df_agg = (
        df.copy()
        .groupby(grouping_cols)  # sw
        .agg(agg_dict)
        .reset_index()
    )  # genereates a multi-index
    # collapse multi index and combine columns names with '_' sep
    df_agg.columns = [
        f'{col[0]}_{col[1]}' if col[1] else col[0] for col in df_agg.columns
    ]

    # insert label count column into df_agg after grouping columns
    insert_pos = 1 if isinstance(grouping_cols, str) else len(grouping_cols)
    df_agg.insert(insert_pos, 'label_count', df_count['label_count'])

    return df_agg


def _convert_to_list(arg: list | ArrayLike | str | None):
    """
    Convert any non-list arguments to lists.

    Parameters
    ----------
    arg : list or ArrayLike or str or None
        The argument to convert.

    Returns
    -------
    list or None
        The converted list or None if the input was None.

    """
    if arg is None:
        return None
    if not isinstance(arg, list):
        return [arg]
    return arg


def _generate_measure_dict(
    label_images: list[ArrayLike] | ArrayLike,
    label_names: list[str] | str | None = None,
    intensity_images: list[ArrayLike] | ArrayLike | None = None,
    intensity_names: list[str] | str | None = None,
) -> dict:
    """
    Generate a dictionary of label and intensity images with their names.

    Parameters
    ----------
    label_images : list of ArrayLike or ArrayLike
        The label images.
    label_names : list of str or str or None, optional
        The names of the label images.
    intensity_images : list of ArrayLike or ArrayLike or None, optional
        The intensity images.
    intensity_names : list of str or str or None, optional
        The names of the intensity images.

    Returns
    -------
    dict
        A dictionary containing label and intensity images with their names.

    """
    label_images = _convert_to_list(label_images)
    intensity_images = _convert_to_list(intensity_images)
    label_names = _convert_to_list(label_names)
    intensity_names = _convert_to_list(intensity_names)

    # automatically generate label and intensity names if not given
    if label_names is None:
        label_names = [f'label_{i}' for i in range(len(label_images))]
    if intensity_names is None and intensity_images is not None:
        intensity_names = [
            f'intensity_{i}' for i in range(len(intensity_images))
        ]

    return {
        'label_images': label_images,
        'label_names': label_names,
        'intensity_images': intensity_images,
        'intensity_names': intensity_names,
    }


def _extract_info_from_id_string(id_string: str, id_regex: dict) -> dict:
    r"""
    Extract information from an id string using a regex dictionary.

    Parameters
    ----------
    id_string : str
        The ID string to extract information from.
    id_regex : dict
        A dictionary where keys are the information to extract and values are
        the regex patterns to use for extraction.

    Returns
    -------
    dict
        A dictionary containing the extracted information.

    Examples
    --------
    >>> id_string = "P14-A6__2024-07-16 25x 18HIC ncoa4 FT dapi obl 01"
    >>> id_regex = {'well': '-(\w+)__', 'HIC': r'(\d{1,3})HIC', 'exp': r'obl (\d{2,3})'}
    >>> _extract_info_from_id_string(id_string, id_regex)
    {'well': 'A6', 'HIC': '18', 'exp': '01'}

    """
    id_dict = {}
    for key, regex in id_regex.items():
        match = re.search(regex, id_string)
        if match:
            id_dict[key] = match.group(1)
        else:
            id_dict[key] = None
    return id_dict


def _rename_intensity_columns(df: pd.DataFrame, intensity_names: list[str]):
    """
    Rename columns in the DataFrame to include the intensity names.

    The intensity names are appended to the end of the column name based
    on the index of the intensity_names list.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame with measurement properties.
    intensity_names : list of str
        The list of intensity names.

    Returns
    -------
    pd.DataFrame
        The DataFrame with renamed columns.

    """
    new_columns = []
    for col in df.columns:
        if any(col.endswith(f'-{idx}') for idx in range(len(intensity_names))):
            base_name, idx = col.rsplit('-', 1)
            new_columns.append(f'{base_name}-{intensity_names[int(idx)]}')
        else:
            new_columns.append(col)

    df.columns = new_columns

    return df


def _map_tx_dict_to_df_id_col(
    tx: dict | None = None,
    tx_n_well: int | None = None,
    tx_leading_zeroes: bool = False,
    df: pd.DataFrame = None,
    id_column: str | None = None,
):
    """
    Map a dictionary of treatments to a DataFrame's id_column.

    This should work on either a complete dataset, or as part of an iterative.

    Parameters
    ----------
    tx : dict or None, optional
        The dictionary of treatments.
    tx_n_well : int or None, optional
        The number of wells in the plate.
    tx_leading_zeroes : bool, optional
        Whether to use leading zeroes in the plate map. Default is False.
    df : pd.DataFrame
        The DataFrame to map treatments to.
    id_column : str or None, optional
        The column in the DataFrame that contains the IDs.

    Returns
    -------
    pd.DataFrame
        The DataFrame with treatments mapped to the id_column.

    """
    if isinstance(tx_n_well, int):
        plate = PlateMapper(tx_n_well, leading_zeroes=tx_leading_zeroes)
        plate.assign_treatments(tx)
        tx_map = plate.plate_map.set_index('well_id').to_dict(orient='index')
    else:
        tx_map = tx

    for identifier, txs in tx_map.items():
        for tx, condition in txs.items():
            if tx not in df.columns:
                df[tx] = None
            df.loc[df[id_column] == identifier, tx] = condition

    return df
