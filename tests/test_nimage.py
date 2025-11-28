from __future__ import annotations

from pathlib import Path
from unittest import mock

import pytest
from ndevio import nImage

RGB_TIFF = "RGB.tiff"  # has two scenes
CELLS3D2CH_OME_TIFF = "cells3d2ch.tiff"  # 2 channel, 3D OME-TIFF

def test_nImage_init(resources_dir: Path):
    img = nImage(resources_dir / RGB_TIFF)
    assert img.path == resources_dir / RGB_TIFF
    assert img.reader is not None
    assert img.data.shape == (1, 1, 1, 1440, 1920, 3)
    assert img.napari_data is None

def test_nImage_ome_reader(resources_dir: Path):
    """
    Test that the OME-TIFF reader is used for OME-TIFF files.

    This test is in response to https://github.com/bioio-devs/bioio/issues/79
    whereby images saved with bioio.writers.OmeTiffWriter are not being read with
    bioio_ome_tiff.Reader, but instead with bioio_tifffile.Reader.

    The example here was saved with aicsimageio.writers.OmeTiffWriter. nImage
    has an __init__ function that should override the reader determined by
    bioio.BioImage.determine_plugin() with bioio_ome_tiff if the image is an
    OME-TIFF.

    """
    import bioio
    import bioio_tifffile

    img_path = resources_dir / CELLS3D2CH_OME_TIFF
    fr = bioio.plugin_feasibility_report(img_path)
    assert fr['bioio-tifffile'].supported is True
    assert fr['bioio-ome-tiff'].supported is True

    nimg = nImage(img_path)
    assert nimg.settings.ndevio_Reader.preferred_reader == 'bioio-ome-tiff'
    # the below only exists if 'bioio-ome-tiff' is used
    assert hasattr(nimg, 'ome_metadata')
    assert nimg.channel_names == ['membrane', 'nuclei']

    nimg = nImage(img_path, reader=bioio_tifffile.Reader)

    # check that despite preferred reader, the reader is still bioio_tifffile
    # because there is no ome_metadata
    assert nimg.settings.ndevio_Reader.preferred_reader == 'bioio-ome-tiff'
    # check that calling nimg.ome_metadata raises NotImplementedError
    with pytest.raises(NotImplementedError):
        _ = nimg.ome_metadata

def test_nImage_save_read(resources_dir: Path, tmp_path: Path):
    """
    Test saving and reading an image with OmeTiffWriter and nImage.

    Confirm that the image is saved with the correct physical pixel sizes and
    channel names, and that it is read back with the same physical pixel sizes
    and channel names because it is an OME-TIFF. See the above test for
    the need of this and to ensure not being read by bioio_tifffile.Reader.

    """
    from bioio.writers import OmeTiffWriter
    from bioio_base.types import PhysicalPixelSizes

    img = nImage(resources_dir / CELLS3D2CH_OME_TIFF)
    assert img.physical_pixel_sizes.X == 1

    img_data = img.get_image_data('CZYX')
    OmeTiffWriter.save(
        img_data,
        tmp_path / "test_save_read.tiff",
        dim_order='CZYX',
        physical_pixel_sizes=PhysicalPixelSizes(1,2,3), # ZYX
        channel_names=['test1','test2']
    )
    assert (tmp_path / "test_save_read.tiff").exists()

    new_img = nImage(tmp_path / "test_save_read.tiff")

    # having the below features means it is properly read as OME-TIFF
    assert new_img.physical_pixel_sizes.Z == 1
    assert new_img.physical_pixel_sizes.Y == 2
    assert new_img.physical_pixel_sizes.X == 3
    assert new_img.channel_names == ['test1','test2']


def test_determine_in_memory(resources_dir: Path):
    img = nImage(resources_dir / RGB_TIFF)
    assert img._determine_in_memory() is True

def test_get_napari_image_data(resources_dir: Path):
    img = nImage(resources_dir / RGB_TIFF)
    img.get_napari_image_data()
    assert img.napari_data.shape == (1440, 1920, 3)
    assert img.napari_data.dims == ('Y', 'X', 'S')

def test_get_napari_image_data_not_in_memory(resources_dir: Path):
    import dask

    img = nImage(resources_dir / RGB_TIFF)
    img.get_napari_image_data(in_memory=False)
    assert img.napari_data is not None
    # check that the data is a dask array
    assert isinstance(img.napari_data.data, dask.array.core.Array)

def test_get_napari_metadata(resources_dir: Path):
    img = nImage(resources_dir / RGB_TIFF)
    img.get_napari_metadata(path = img.path)
    assert img.napari_metadata['name'] == '0 :: Image:0 :: RGB'
    assert img.napari_metadata['scale'] == (264.5833333333333, 264.5833333333333)
    assert img.napari_metadata['rgb'] is True

def test_nImage_determine_in_memory_large_file(resources_dir: Path):
    img = nImage(resources_dir / RGB_TIFF)
    with mock.patch(
        'psutil.virtual_memory',
        return_value=mock.Mock(available=1e9)
    ):
        with mock.patch(
            'bioio_base.io.pathlike_to_fs',
            return_value=(mock.Mock(size=lambda x: 5e9), '')
        ):
            assert img._determine_in_memory() is False


def test_get_napari_image_data_mosaic_tile_in_memory(resources_dir: Path):
    import xarray as xr
    from bioio_base.dimensions import DimensionNames

    with mock.patch.object(nImage, 'reader', create=True) as mock_reader:
        mock_reader.dims.order = [DimensionNames.MosaicTile]
        mock_reader.mosaic_xarray_data.squeeze.return_value = xr.DataArray([1, 2, 3])
        img = nImage(resources_dir / RGB_TIFF)
        data = img.get_napari_image_data(in_memory=True)
        assert data is not None
        assert data.shape == (3,)
        assert img.napari_data is not None

def test_get_napari_image_data_mosaic_tile_not_in_memory(resources_dir: Path):
    import xarray as xr
    from bioio_base.dimensions import DimensionNames

    with mock.patch.object(nImage, 'reader', create=True) as mock_reader:
        mock_reader.dims.order = [DimensionNames.MosaicTile]
        mock_reader.mosaic_xarray_dask_data.squeeze.return_value = xr.DataArray([1, 2, 3])
        img = nImage(resources_dir / RGB_TIFF)
        data = img.get_napari_image_data(in_memory=False)
        assert data is not None
        assert data.shape == (3,)
        assert img.napari_data is not None
