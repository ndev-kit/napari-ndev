import logging
import tempfile
from pathlib import Path

import numpy as np
import pytest
from bioio.writers import OmeTiffWriter
from ndevio import nImage

from napari_ndev.helpers import (
    check_for_missing_files,
    create_id_string,
    elide_string,
    get_channel_names,
    get_directory_and_files,
    get_squeezed_dim_order,
    setup_logger,
)


def test_check_for_missing_files_path(tmp_path):
    # Create a directory and some files
    directory = Path(tmp_path / 'test_dir')
    directory.mkdir()
    file1 = directory / 'file1.txt'
    file1.write_text('This is a test file.')
    file2 = directory / 'file2.txt'
    file2.write_text('This is another test file.')
    file3 = directory / 'file3.txt'

    # Check for missing files
    missing_files = check_for_missing_files([file1, file2], directory)
    assert missing_files == []

    # Check for missing files
    missing_files = check_for_missing_files([file1, file3], directory)
    assert missing_files == [('file3.txt', 'test_dir')]


def test_check_for_missing_file_str(tmp_path):
    # Create a directory and some files
    directory = tmp_path / 'test_dir'
    directory.mkdir()
    file1 = directory / 'file1.txt'
    file1.write_text('This is a test file.')
    file2 = directory / 'file2.txt'
    file2.write_text('This is another test file.')

    # Check for missing files
    missing_files = check_for_missing_files(
        ['file1.txt', 'file2.txt'], directory
    )
    assert missing_files == []

    # Check for missing files
    missing_files = check_for_missing_files(
        ['file1.txt', 'file3.txt'], directory
    )
    assert missing_files == [('file3.txt', 'test_dir')]


def test_create_id_string():
    img = nImage(np.random.random((2, 2)))
    identifier = 'test_id'
    id_string = create_id_string(img, identifier)
    assert id_string == 'test_id__0__Image:0'


def test_create_id_string_no_id():
    img = nImage(np.random.random((2, 2)))
    identifier = None
    id_string = create_id_string(img, identifier)
    assert id_string == 'None__0__Image:0'


def test_create_id_string_ome_metadata_no_name(resources_dir: Path):
    file = Path(
        # './tests/resources/Workflow/Images/cells3d2ch.tiff'
        resources_dir / 'Workflow/Images/cells3d2ch.tiff'
    )
    img = nImage(file)

    identifier = file.stem
    id_string = create_id_string(img, identifier)

    assert img.ome_metadata.images[0].name is None
    assert img.channel_names == ['membrane', 'nuclei']
    assert id_string == 'cells3d2ch__0__Image:0'


def test_create_id_string_ometiffwriter_name(tmp_path):
    OmeTiffWriter.save(
        data=np.random.random((2, 2)),
        uri=tmp_path / 'test.tiff',
        image_name='test_image',
    )

    img = nImage(tmp_path / 'test.tiff')
    identifier = 'test_id'

    id_string = create_id_string(img, identifier)
    assert img.current_scene == 'Image:0'
    assert id_string == 'test_id__0__test_image'


def test_get_channel_names_CYX():
    file = Path(
        r'./tests/resources/Workflow/Images/cells3d2ch.tiff'
    )
    img = nImage(file)
    assert get_channel_names(img) == img.channel_names


def test_get_channel_names_RGB():
    file = Path(r'./tests/resources/RGB.tiff')
    img = nImage(file)
    assert get_channel_names(img) == ['red', 'green', 'blue']


def test_get_directory_and_files_default_pattern():
    directory = Path(r'./tests/resources/test_czis')
    directory, files = get_directory_and_files(directory)
    assert directory == Path(r'./tests/resources/test_czis')
    assert files == list(directory.glob('*'))


def test_get_directory_and_files_custom_pattern():
    directory = Path(r'./tests/resources/test_czis')
    directory, files = get_directory_and_files(directory, pattern='.czi')
    assert directory == Path(r'./tests/resources/test_czis')
    assert files == list(directory.glob('*.czi'))


def test_get_directory_and_files_none_dir():
    directory, files = get_directory_and_files(None)
    assert directory is None
    assert files == []


def test_get_directory_and_files_dir_not_exists():
    directory = Path(
        r'./tests/resources/test_czis_not_exists'
    )
    with pytest.raises(FileNotFoundError):
        get_directory_and_files(directory)


def test_get_squeezed_dim_order_ZYX():
    file = Path(
        r'./tests/resources/Workflow/Images/cells3d2ch.tiff'
    )
    img = nImage(file)
    assert get_squeezed_dim_order(img) == 'ZYX'


def test_get_squeezed_dim_order_RGB():
    file = Path(r'./tests/resources/RGB.tiff')
    img = nImage(file)
    assert get_squeezed_dim_order(img) == 'YX'


def test_setup_logger():
    # Create a temporary file for logging
    with tempfile.NamedTemporaryFile(delete=False) as temp_log_file:
        log_path = Path(temp_log_file.name)

    try:
        # Set up the logger
        logger, handler = setup_logger(log_path)

        # Check that the logger is set up correctly
        assert isinstance(logger, logging.Logger)
        assert logger.level == logging.INFO
        assert len(logger.handlers) == 1
        assert logger.handlers[0] == handler

        # Check that the handler is set up correctly
        assert isinstance(handler, logging.FileHandler)
        assert handler.level == logging.INFO
        assert handler.baseFilename == str(log_path)

        # Check that the formatter is set up correctly
        formatter = handler.formatter
        assert isinstance(formatter, logging.Formatter)
        assert formatter._fmt == '%(asctime)s - %(message)s'

    finally:
        # remove the handler and close it
        logger.removeHandler(handler)
        handler.close()
        # Clean up the temporary log file
        log_path.unlink()

def test_elide_string():
    # Test cases where the string is shorter than the max length
    assert elide_string("short", 10) == "short"
    assert elide_string("short", 6) == "short"

    # Test cases where the string is exactly the max length
    assert elide_string("exactly15chars", 15) == "exactly15chars"

    # Test cases where the string is longer than the max length
    assert elide_string("thisisaverylongstring", 10) == "thi...ing"
    assert elide_string("thisisaverylongstring", 15) == "thisis...string"
    assert elide_string("thisisaverylongstring", 5) == "thisi"

    # Test cases for different elide locations
    assert elide_string("thisisaverylongstring", 10, "start") == "...gstring"
    assert elide_string("thisisaverylongstring", 10, "end") == "thisisa..."
    assert elide_string("thisisaverylongstring", 10, "middle") == "thi...ing"

    # Test cases for very small max_length
    assert elide_string("thisisaverylongstring", 3) == "thi"
    assert elide_string("thisisaverylongstring", 4) == "this"
    assert elide_string("thisisaverylongstring", 5) == "thisi"

    # Test cases for invalid location
    with pytest.raises(ValueError, match='Invalid location. Must be "start", "middle", or "end".'):
        elide_string("thisisaverylongstring", 10, "invalid")

    # Test cases for edge cases
    assert elide_string("", 10) == ""
    assert elide_string("a", 1) == "a"
    assert elide_string("ab", 1) == "a"
    assert elide_string("abc", 2) == "ab"
    assert elide_string("abcd", 3) == "abc"
