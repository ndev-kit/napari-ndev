from unittest.mock import patch

from napari_ndev import __main__


@patch('subprocess.run')
def test_main(mock_run):
    __main__.main()
    # Check if subprocess.run was called, indicating an attempt to open napari
    assert mock_run.called
    # Optionally, you can check the arguments to ensure it was called correctly
    mock_run.assert_called_with(["napari", "-w", "napari-ndev"])
