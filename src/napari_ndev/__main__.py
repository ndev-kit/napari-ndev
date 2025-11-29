"""Open napari with the napari_ndev plugin using command line."""

import subprocess


def main():
    """Run napari with the napari_ndev plugin."""
    subprocess.run(['napari', '-w', 'napari-ndev'])
