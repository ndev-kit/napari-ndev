import sys

if sys.version_info >= (3, 10):
    from importlib.metadata import entry_points
else:
    from importlib_metadata import entry_points

from magicclass.widgets import GroupBoxContainer
from magicgui.widgets import (
    CheckBox,
    ComboBox,
    Container,
    FloatSpinBox,
    TupleEdit,
)

from napari_ndev._settings import get_settings


class SettingsContainer(Container):
    def __init__(self):
        super().__init__(labels=False)
        self.settings = get_settings()
        self._available_readers = [
            reader.name
            for reader in entry_points(
                group='bioio.readers'
            )  # use entry_points(group='bioio.readers') for py >= 3.10
        ]
        self._preferred_reader = (
            self.settings.PREFERRED_READER
            if self.settings.PREFERRED_READER in self._available_readers
            else 'bioio-ome-tiff'
        )

        self._init_widgets()
        self._connect_events()

    def _init_widgets(self):
        self._preferred_reader_combo = ComboBox(
            label='Preferred Reader',
            value=self._preferred_reader,
            choices=self._available_readers,
            tooltip='Preferred reader to use when opening images. \n'
            'If the reader is not available, it will attempt to fallback \n'
            'to the next available working reader.',
        )
        self._scene_handling_combo = ComboBox(
            label='Multi-Scene Handling',
            value=self.settings.SCENE_HANDLING,
            choices=[
                'Open Scene Widget',
                'View All Scenes',
                'View First Scene Only',
            ],
            tooltip='How to handle files with multiple-scenes, by default \n'
            'opens a widget to select the scenes to open. \n'
            'If "View All Scenes" then the scenes will be added as a slider \n'
            'dimension in the viewer.',
        )
        self._unpack_channels_as_layers_checkbox = CheckBox(
            value=self.settings.UNPACK_CHANNELS_AS_LAYERS,
            label='Unpack Channels as Layers',
            tooltip='Whether to unpack channels as layers.',
        )
        self._clear_on_scene_select_checkbox = CheckBox(
            value=self.settings.CLEAR_LAYERS_ON_NEW_SCENE,
            label='Clear All Layers On New Scene Selection',
            tooltip='Whether to clear the viewer when selecting a new scene.',
        )
        self._bioio_settings_container = GroupBoxContainer(
            name='Reader Settings',
            widgets=[
                self._preferred_reader_combo,
                self._scene_handling_combo,
                self._clear_on_scene_select_checkbox,
                self._unpack_channels_as_layers_checkbox,
            ],
            layout='vertical',
        )

        self._canvas_scale_slider = FloatSpinBox(
            label='Canvas Scale',
            value=self.settings.CANVAS_SCALE,
            min=0.01,
            max=100.0,
            step=1.0,
            tooltip='Scales exported figures and screenshots by this value.',
        )
        self._override_canvas_size_checkbox = CheckBox(
            value=self.settings.OVERRIDE_CANVAS_SIZE,
            label='Override Canvas Size',
            tooltip='Whether to override the canvas size when exporting canvas screenshot.',
        )
        self._canvas_size_tuple = TupleEdit(
            value=self.settings.CANVAS_SIZE,
            label='Canvas Size',
            tooltip='Height x width of the canvas when exporting a screenshot.'
            ' Only used if "Override Canvas Size" is checked.',
        )
        self._export_settings_container = GroupBoxContainer(
            name='Export Settings',
            widgets=[
                self._canvas_scale_slider,
                self._override_canvas_size_checkbox,
                self._canvas_size_tuple,
            ],
            layout='vertical',
        )

        self.extend(
            [self._bioio_settings_container, self._export_settings_container]
        )
        self.native.layout().addStretch()

    def _connect_events(self):
        self._preferred_reader_combo.changed.connect(self._update_settings)
        self._scene_handling_combo.changed.connect(self._update_settings)
        self._clear_on_scene_select_checkbox.changed.connect(
            self._update_settings
        )
        self._unpack_channels_as_layers_checkbox.changed.connect(
            self._update_settings
        )
        self._canvas_scale_slider.changed.connect(self._update_settings)
        self._override_canvas_size_checkbox.changed.connect(
            self._update_settings
        )
        self._canvas_size_tuple.changed.connect(self._update_settings)

    def _update_settings(self):
        self._preferred_reader = self._preferred_reader_combo.value
        widget_to_setting = {
            'PREFERRED_READER': self._preferred_reader_combo,
            'SCENE_HANDLING': self._scene_handling_combo,
            'CLEAR_LAYERS_ON_NEW_SCENE': self._clear_on_scene_select_checkbox,
            'UNPACK_CHANNELS_AS_LAYERS': self._unpack_channels_as_layers_checkbox,
            'CANVAS_SCALE': self._canvas_scale_slider,
            'OVERRIDE_CANVAS_SIZE': self._override_canvas_size_checkbox,
            'CANVAS_SIZE': self._canvas_size_tuple,
        }

        # the below updates all the settings of the singleton after something changes
        for setting_name, widget in widget_to_setting.items():
            setattr(self.settings, setting_name, widget.value)

        self.settings.save_settings()
