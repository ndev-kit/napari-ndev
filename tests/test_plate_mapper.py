from __future__ import annotations

import pandas as pd
import pytest

from napari_ndev._plate_mapper import PlateMapper


@pytest.fixture
def plate_mapper():
    return PlateMapper(96)





def test_plate_mapper_init_empty():
    pm = PlateMapper()
    plate_map = pm.plate_map
    assert isinstance(plate_map, pd.DataFrame)
    assert pm.pivoted_plate_map is None
    assert pm.styled_plate_map is None
    assert len(plate_map) == 96
    assert len(plate_map.columns) == 3
    assert 'row' in plate_map.columns
    assert 'A' in plate_map['row'].values
    assert 'column' in plate_map.columns
    assert 1 in plate_map['column'].values
    assert 'well_id' in plate_map.columns
    assert 'A1' in plate_map['well_id'].values

def test_plate_mapper_init_with_plate_size():
    pm = PlateMapper(384)
    plate_map = pm.plate_map
    assert len(plate_map) == 384
    assert len(plate_map.columns) == 3
    assert 'P' in plate_map['row'].values # 16th letter
    assert 24 in plate_map['column'].values

def test_plate_mapper_leading_zeroes():
    pm = PlateMapper(leading_zeroes=True)
    assert 'A' in pm.plate_map['row'].values
    assert '01' in pm.plate_map['column'].values
    assert '12' in pm.plate_map['column'].values
    assert 'A01' in pm.plate_map['well_id'].values
    assert 'H12' in pm.plate_map['well_id'].values

@pytest.fixture
def treatments():
    return {
        'Treatment1': {'Condition1': ['A1', 'B2'], 'Condition2': ['C3']},
        'Treatment2': {'Condition3': ['D4:E5']},
    }

def test_plate_mapper_init_with_treatments(treatments):
    pm = PlateMapper(96, treatments=treatments)
    plate_map = pm.plate_map
    pivoted_pm = pm.pivoted_plate_map

    assert isinstance(plate_map, pd.DataFrame)
    assert isinstance(pivoted_pm, pd.DataFrame)
    assert 'Treatment1' in plate_map.columns
    assert 'Treatment2' in plate_map.columns
    assert (
        plate_map.loc[plate_map['well_id'] == 'A1', 'Treatment1'].values[0]
        == 'Condition1'
    )
    assert (
        plate_map.loc[plate_map['well_id'] == 'B2', 'Treatment1'].values[0]
        == 'Condition1'
    )
    assert (
        plate_map.loc[plate_map['well_id'] == 'C3', 'Treatment1'].values[0]
        == 'Condition2'
    )
    assert (
        plate_map.loc[plate_map['well_id'] == 'D4', 'Treatment2'].values[0]
        == 'Condition3'
    )
    assert (
        plate_map.loc[plate_map['well_id'] == 'E5', 'Treatment2'].values[0]
        == 'Condition3'
    )
    assert (
        plate_map.loc[plate_map['well_id'] == 'E4', 'Treatment2'].values[0]
        == 'Condition3'
    )

def test_plate_mapper_init_with_treatments_and_leading_zeroes(treatments):
    pm = PlateMapper(96, treatments=treatments, leading_zeroes=True)
    plate_map = pm.plate_map

    assert (
        plate_map.loc[plate_map['well_id'] == 'A01', 'Treatment1'].values[0]
    )


def test_plate_mapper_get_pivoted_plate_map(
    plate_mapper: PlateMapper, treatments: dict[str, dict[str, list[str]]]
):
    plate_mapper.assign_treatments(treatments)
    plate_map_pivot = plate_mapper.get_pivoted_plate_map('Treatment1')

    assert isinstance(plate_map_pivot, pd.DataFrame)
    assert len(plate_map_pivot) == 8
    assert len(plate_map_pivot.columns) == 12
    assert 'A' in plate_map_pivot.index
    assert 'Condition1' in plate_map_pivot.values
    assert 'Condition2' in plate_map_pivot.values


def test_plate_mapper_get_styled_plate_map(
    plate_mapper: PlateMapper, treatments: dict[str, dict[str, list[str]]]
):
    plate_mapper.assign_treatments(treatments)
    plate_map_styled = plate_mapper.get_styled_plate_map('Treatment1')

    assert isinstance(plate_map_styled, pd.io.formats.style.Styler)
    assert plate_map_styled.caption == 'Treatment1 Plate Map'
