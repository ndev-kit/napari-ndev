from napari_ndev._settings import get_settings
from napari_ndev.widgets import SettingsContainer


def test_settings_container():
    container = SettingsContainer()
    settings_singleton = get_settings()
    original_reader = settings_singleton.PREFERRED_READER

    assert container.settings is settings_singleton
    assert 'bioio-tifffile' in container._available_readers # check that available readers are loaded
    assert container._preferred_reader in container._available_readers # check that preferred reader is in available readers
    assert container._preferred_reader == settings_singleton.PREFERRED_READER # check that preferred reader is the same as the settings singleton

    assert container._preferred_reader_combo.value == settings_singleton.PREFERRED_READER
    assert container._scene_handling_combo.value == settings_singleton.SCENE_HANDLING
    assert container._clear_on_scene_select_checkbox.value == settings_singleton.CLEAR_LAYERS_ON_NEW_SCENE
    assert container._unpack_channels_as_layers_checkbox.value == settings_singleton.UNPACK_CHANNELS_AS_LAYERS

    # then, change a value and check that the settings singleton is updated
    container._preferred_reader_combo.value = 'bioio-imageio' # should also be in defaults
    assert settings_singleton.PREFERRED_READER == 'bioio-imageio'

    # now switch back to the original value, so to not mess up the users settings
    container._preferred_reader_combo.value = original_reader
