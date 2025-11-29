from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from magicclass.widgets import (
    ScrollableContainer,
    TabbedContainer,
)
from magicgui.widgets import (
    Container,
    Label,
    PushButton,
)

from napari_ndev import __version__

if TYPE_CHECKING:
    import napari


class nDevContainer(ScrollableContainer):
    """
    A widget container to container the primary ndev widgets.

    Parameters
    ----------
    viewer: napari.viewer.Viewer, optional
        The napari viewer instance.

    """

    def __init__(self, viewer: napari.viewer.Viewer = None):
        """
        Initialize the MainContainer.

        Parameters
        ----------
        viewer: napari.viewer.Viewer, optional
            The napari viewer instance.

        """
        super().__init__(labels=False)

        self.min_width = 600  # TODO: remove this hardcoded value
        self._viewer = viewer if viewer is not None else None

        _logo_path = (
            Path(__file__).parent.parent / 'resources' / 'nDev-logo-small.png'
        )
        self._logo_label = Label(
            value='<h1 style="text-align: center;">'
            f'<img src="{_logo_path}"/>'
            '</h1>'
        )
        self._version_label = Label(value=f'v{__version__}')

        self._docs_link_button = PushButton(
            text='Docs', icon='ic:round-menu-book'
        )
        self._bug_report_link_button = PushButton(
            text='Bug Report',
            icon='ic:outline-bug-report',
        )
        self._link_container = Container(
            widgets=[
                self._docs_link_button,
                self._bug_report_link_button,
                self._version_label,
            ],
            layout='vertical',
        )
        self._header_container = Container(
            widgets=[self._logo_label, self._link_container],
            layout='horizontal',
        )

        self._init_widget_containers()
        self._init_layout()
        self._init_callbacks()

    def _init_widget_containers(self):
        from napari_ndev.widgets._apoc_container import ApocContainer
        from napari_ndev.widgets._measure_container import MeasureContainer
        from napari_ndev.widgets._settings_container import SettingsContainer
        from napari_ndev.widgets._utilities_container import UtilitiesContainer
        from napari_ndev.widgets._workflow_container import WorkflowContainer

        """Initialize the widget containers."""
        self._apoc_container = ApocContainer(viewer=self._viewer)
        self._apoc_container.label = 'APOC'
        self._measure_container = MeasureContainer(viewer=self._viewer)
        self._measure_container.label = 'Measure'
        self._utilities_container = UtilitiesContainer(viewer=self._viewer)
        self._utilities_container.label = 'Utilities'
        self._workflow_container = WorkflowContainer(viewer=self._viewer)
        self._workflow_container.label = 'Workflow'
        self._settings_container = SettingsContainer()
        self._settings_container.label = 'Settings'

        self._tabbed_container = TabbedContainer(
            # labels=["Apoc", "Measure", "Utilities", "Workflow"],
            labels=False,
            layout='horizontal',
            widgets=[
                self._utilities_container,
                self._apoc_container,
                self._workflow_container,
                self._measure_container,
                self._settings_container,
            ],
        )

    def _init_layout(self):
        """Initialize the layout."""
        self.append(self._header_container)
        self.append(self._tabbed_container)
        # self.stretch(self._tabbed_container)

    def _init_callbacks(self):
        """Initialize the widget callbacks."""
        self._docs_link_button.clicked.connect(self._open_docs_link)
        self._bug_report_link_button.clicked.connect(
            self._open_bug_report_link
        )

    def _open_docs_link(self):
        """Open the documentation link."""
        import webbrowser

        webbrowser.open('https://ndev-kit.github.io/')

    def _open_bug_report_link(self):
        """Open the bug report link."""
        import webbrowser

        webbrowser.open('https://github.com/ndev-kit/napari-ndev/issues')
