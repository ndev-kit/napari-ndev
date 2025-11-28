import yaml

from napari_ndev._settings import Settings, get_settings


def test_settings(tmp_path):

    # Write a temporary settings file
    settings_file = tmp_path / 'test_settings.yaml'
    settings_file.write_text(
        yaml.dump(
            {
                'PREFERRED_READER': 'test-reader',
                'SCENE_HANDLING': 'test-scene',
                'CLEAR_LAYERS_ON_NEW_SCENE': True,
                'UNPACK_CHANNELS_AS_LAYERS': False,
            }
        )
    )

    # Test that the settings are loaded correctly (init calls load_settings)
    settings = Settings(str(settings_file))
    assert settings.PREFERRED_READER == 'test-reader'
    assert settings.SCENE_HANDLING == 'test-scene'
    assert settings.CLEAR_LAYERS_ON_NEW_SCENE
    assert not settings.UNPACK_CHANNELS_AS_LAYERS

    # Update settings, and then test that save works
    settings.PREFERRED_READER = 'new-reader'
    settings.SCENE_HANDLING = 'new-scene'
    settings.CLEAR_LAYERS_ON_NEW_SCENE = False
    settings.UNPACK_CHANNELS_AS_LAYERS = True


    settings.save_settings()

    with open(settings_file) as file:
        saved_settings = yaml.safe_load(file)

    assert saved_settings['PREFERRED_READER'] == 'new-reader'
    assert saved_settings['SCENE_HANDLING'] == 'new-scene'
    assert not saved_settings['CLEAR_LAYERS_ON_NEW_SCENE']
    assert saved_settings['UNPACK_CHANNELS_AS_LAYERS']

def test_get_settings():
    # this test will look at the real settings file
    # so if there are updates, this may need changed
    settings1 = get_settings()
    settings2 = get_settings()

    assert settings1.PREFERRED_READER == 'bioio-ome-tiff'
    assert settings1 is settings2 # ensure is singleton

    settings1.PREFERRED_READER = 'test-reader'

    assert settings2.PREFERRED_READER == 'test-reader'

    settings1.PREFERRED_READER = 'bioio-ome-tiff' # reset for other tests

    assert settings1.SCENE_HANDLING == 'Open Scene Widget'
    assert settings1.CLEAR_LAYERS_ON_NEW_SCENE is False
    assert settings1.UNPACK_CHANNELS_AS_LAYERS is True
    assert settings1.CANVAS_SCALE == 1.0
    assert settings1.OVERRIDE_CANVAS_SIZE is False
    assert settings1.CANVAS_SIZE == [1024, 1024]
