"""Integration tests for ndev-settings usage in napari-ndev.

These tests verify that napari-ndev correctly integrates with ndev-settings.
"""


def test_ndev_settings_import():
    """Test that ndev_settings can be imported."""
    from ndev_settings import get_settings

    settings = get_settings()
    assert settings is not None


def test_ndev_settings_has_ndevio_reader_group():
    """Test that ndevio_reader settings group is available."""
    from ndev_settings import get_settings

    settings = get_settings()
    assert hasattr(settings, 'ndevio_reader')

    # Test expected settings exist
    reader = settings.ndevio_reader
    assert hasattr(reader, 'preferred_reader')
    assert hasattr(reader, 'scene_handling')
    assert hasattr(reader, 'clear_layers_on_new_scene')
    assert hasattr(reader, 'unpack_channels_as_layers')


def test_ndev_settings_has_ndevio_export_group():
    """Test that ndevio_export settings group is available."""
    from ndev_settings import get_settings

    settings = get_settings()
    assert hasattr(settings, 'ndevio_export')

    # Test expected settings exist
    export = settings.ndevio_export
    assert hasattr(export, 'canvas_scale')
    assert hasattr(export, 'override_canvas_size')
    assert hasattr(export, 'canvas_size')


def test_ndev_settings_singleton():
    """Test that get_settings returns the same instance."""
    from ndev_settings import get_settings

    settings1 = get_settings()
    settings2 = get_settings()
    assert settings1 is settings2


def test_utilities_container_uses_ndev_settings():
    """Test that UtilitiesContainer correctly uses ndev-settings."""
    from ndev_settings import get_settings
    from ndevio.widgets import UtilitiesContainer

    settings = get_settings()
    container = UtilitiesContainer()

    # The container should have access to settings
    assert container._settings is settings
