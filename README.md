# napari-ndev

[![License BSD-3](https://img.shields.io/pypi/l/napari-ndev.svg?color=green)](https://github.com/ndev-kit/napari-ndev/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/napari-ndev.svg?color=green)](https://pypi.org/project/napari-ndev)
[![Python package index download statistics](https://img.shields.io/pypi/dm/napari-ndev.svg)](https://pypistats.org/packages/napari-ndev)
[![Python Version](https://img.shields.io/pypi/pyversions/napari-ndev.svg?color=green)](https://python.org)
[![tests](https://github.com/ndev-kit/napari-ndev/workflows/tests/badge.svg)](https://github.com/ndev-kit/napari-ndev/actions)
[![codecov](https://codecov.io/gh/ndev-kit/napari-ndev/branch/main/graph/badge.svg)](https://codecov.io/gh/ndev-kit/napari-ndev)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/napari-ndev)](https://napari-hub.org/plugins/napari-ndev)
![Static Badge](https://img.shields.io/badge/plugin-npe2-brightgreen?style=flat-square&label=plugin)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14787853.svg)](https://doi.org/10.5281/zenodo.14787853)

<img src="https://github.com/ndev-kit/napari-ndev/raw/main/resources/nDev-logo-large.png" alt="logo" width="400" style="display: block; margin: auto;">

A collection of widgets intended to serve any person seeking to process microscopy images from start to finish, *with no coding necessary*. `napari-ndev` was designed to address the **gap between the napari viewer and batch python scripting**.

* Accepts **diverse image formats**, dimensionality, file size, and maintains key metadata.
* Allows **advanced, arbitrary image processing** workflows to be used by novices.
* **User-friendly** sparse annotation and batch training of **machine learning classifiers**.
* Flexible label measurements, parsing of metadata, and summarization for **easily readable datasets**.
* Designed for ease of use, modification, and reproducibility.

## [Check out the Docs to learn more!](https://ndev-kit.github.io)

### See the [poster presented at BINA 2024](https://ndev-kit.github.io/BINA_poster/) for an overview of the plugins in action

### Try out the [Virtual I2K 2024 Workshop](https://ndev-kit.github.io/tutorial/00_setup/) for an interactive tutorial

## Installation

**See the [full installation guide](https://ndev-kit.github.io/installation/) for more options including uv and Pixi.**

**napari-ndev** is a pure Python package and can be installed with [pip]:

```bash
pip install napari-ndev
```

The easiest way to get started is to install the opinionated optional dependencies, which includes napari and PyQt6 (the Qt backend) and additional napari plugins:

```bash
pip install napari-ndev[all]
```

After installation, launch napari with the nDev App widget open:

```bash
napari-ndev
```

This is equivalent to running `napari -w napari-ndev "nDev App"`.

----------------------------------

### Optional Dependencies

**napari-ndev** provides several optional dependency groups:

* **`[qtpy-backend]`** - Includes `napari[pyqt6]` for the Qt GUI
* **`[extras]`** - Additional napari plugins like napari-assistant for enhanced workflows
* **`[all]`** - Everything above plus sample data and themes (recommended)

### Additional Image Format Support

**napari-ndev** uses [ndevio](https://github.com/ndev-kit/ndevio) for image I/O, which relies on [bioio](https://github.com/bioio-devs/bioio) readers. Basic formats (TIFF, OME-TIFF, OME-Zarr, PNG, etc.) work out of the box.

For additional formats, install the appropriate bioio reader:

```bash
# CZI files (GPL-3 licensed)
pip install bioio-czi

# LIF files (GPL-3 licensed)
pip install bioio-lif

# Bio-Formats for many proprietary formats
pip install bioio-bioformats
```

**Note:** Some bioio readers are GPL-3 licensed. If you install and use these, you must comply with GPL-3 license terms.

### Development

For development, clone the repository and install with the dev dependency group:

```bash
git clone https://github.com/ndev-kit/napari-ndev.git
cd napari-ndev
pip install -e . --group dev
```

This includes pytest, pytest-cov, pytest-qt, ruff, pre-commit, and all optional dependencies.

Run tests:

```bash
pytest -v --cov=napari_ndev
```

## Pixi Usage

You may locally clone this repo and use [Pixi](https://pixi.sh) to create a reproducible environment:

```bash
git clone https://github.com/ndev-kit/napari-ndev.git
cd napari-ndev
```

Then launch napari with the nDev plugin:

```bash
pixi run napari-ndev
```

Or install the package in editable/development mode and activate the local environment:

```bash
pixi install           # Default environment with qtpy-backend
pixi install -e dev    # Development environment with testing tools
pixi shell             # Activate the environment
napari                 # Run napari or any command
```

To run tests: `pixi run -e dev test`

----------------------------------

The wide breadth of this plugin's scope is only made possible by the amazing libraries and plugins from the python and napari community, especially [Robert Haase](https://github.com/haesleinhuepf).

This [napari] plugin was generated with [Cookiecutter] using [napari]'s [cookiecutter-napari-plugin] template.

## Contributing

Contributions are very welcome. Tests can be run with [tox], please ensure
the coverage at least stays the same before you submit a pull request.

## License

Distributed under the terms of the [BSD-3] license,
"napari-ndev" is free and open source software.

Some optional libraries can be installed to add functionality to `napari-ndev`, including some that may be more restrictive than this package's BSD-3-Clause.

## Issues

If you encounter any problems, please [file an issue] along with a detailed description.

[napari]: https://github.com/napari
[Cookiecutter]: https://github.com/audreyr/cookiecutter
[BSD-3]: http://opensource.org/licenses/BSD-3-Clause
[cookiecutter-napari-plugin]: https://github.com/napari/cookiecutter-napari-plugin

[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/pip/
