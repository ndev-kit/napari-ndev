from pathlib import Path

import yaml


class Settings:
    """A class to manage settings for the nDev plugin."""

    def __init__(self, settings_file: str):
        """Initialize the settings manager with a file path."""
        self.settings_file = settings_file
        self.load_settings()

    def load_settings(self):
        """Load settings from the settings file."""
        with open(self.settings_file) as file:
            settings = yaml.safe_load(file)

            self.PREFERRED_READER = settings.get(
                'PREFERRED_READER', 'bioio-ome-tiff'
            )
            self.SCENE_HANDLING = settings.get(
                'SCENE_HANDLING', 'Open Scene Widget'
            )
            self.CLEAR_LAYERS_ON_NEW_SCENE = settings.get(
                'CLEAR_LAYERS_ON_NEW_SCENE', False
            )
            self.UNPACK_CHANNELS_AS_LAYERS = settings.get(
                'UNPACK_CHANNELS_AS_LAYERS', True
            )
            self.CANVAS_SCALE = settings.get('CANVAS_SCALE', 1.0)
            self.OVERRIDE_CANVAS_SIZE = settings.get(
                'OVERRIDE_CANVAS_SIZE', False
            )
            self.CANVAS_SIZE = settings.get('CANVAS_SIZE', (1024, 1024))

    def save_settings(self):
        """Save the current settings to the settings file."""
        settings = {
            'PREFERRED_READER': self.PREFERRED_READER,
            'SCENE_HANDLING': self.SCENE_HANDLING,
            'CLEAR_LAYERS_ON_NEW_SCENE': self.CLEAR_LAYERS_ON_NEW_SCENE,
            'UNPACK_CHANNELS_AS_LAYERS': self.UNPACK_CHANNELS_AS_LAYERS,
            'CANVAS_SCALE': self.CANVAS_SCALE,
            'OVERRIDE_CANVAS_SIZE': self.OVERRIDE_CANVAS_SIZE,
            'CANVAS_SIZE': self.CANVAS_SIZE,
        }

        with open(self.settings_file, 'w') as file:
            yaml.safe_dump(settings, file)


_settings_instance = None


def get_settings() -> Settings:
    """Get the singleton instance of the settings manager."""
    global _settings_instance
    if _settings_instance is None:
        _settings_instance = Settings(
            str(Path(__file__).parent / 'ndev_settings.yaml')
        )
    return _settings_instance
