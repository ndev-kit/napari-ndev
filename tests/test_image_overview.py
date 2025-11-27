import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib_scalebar.scalebar import ScaleBar

from napari_ndev.image_overview import (
    ImageOverview,
    ImageSet,
    _add_scalebar,
    _convert_dict_to_ImageSet,
    image_overview,
)


def test_ImageSet():
    # create a random numpy array of size 100 x 100
    data = np.random.rand(100, 100)
    # create a dictionary with the image data
    image_set = ImageSet(
        image=[data],
        title=['Image 1'],
        labels=None,
        min_display_intensity=[0],
        max_display_intensity=[1],
    )

    assert isinstance(image_set, ImageSet)
    assert isinstance(image_set.image, list)
    assert isinstance(image_set.image[0], np.ndarray)
    assert isinstance(image_set.title, list)
    assert isinstance(image_set.title[0], str)
    assert isinstance(image_set.labels, list)
    assert isinstance(image_set.labels[0], bool)
    assert image_set.labels[0] is False
    assert isinstance(image_set.colormap, list)
    assert image_set.colormap[0] is None
    assert isinstance(image_set.min_display_intensity, list)
    assert isinstance(image_set.min_display_intensity[0], (int, float))
    assert isinstance(image_set.max_display_intensity, list)
    assert isinstance(image_set.max_display_intensity[0], (int, float))

def test_image_overview_wrap():
    # create a random numpy array of size 100 x 100
    data = np.random.rand(100, 100)
    # create a dictionary with the image data
    five_image_set = ImageSet(
        image=[data, data, data, data, data],
        title=['Image 1', 'Image 2', 'Image 3', 'Image 4', 'Image 5'],
    )

    fig = image_overview(five_image_set)
    assert isinstance(fig, plt.Figure)
    assert np.array_equal(
        fig.get_size_inches(), np.array([9, 6]) # 3 columns * 3 width, 2 rows * 3 height
    )
    assert len(fig.axes) == 6
    assert fig.axes[0].get_title() == 'Image 1'
    assert not fig.axes[5].get_title()
    assert not fig.axes[5].get_images()

def test_image_overview_nowrap():
    # create a random numpy array of size 100 x 100
    data = np.random.rand(100, 100)
    # create a dictionary with the image data
    three_image_set = ImageSet(
        image=[data, data, data],
        title=['Image 1', 'Image 2', 'Image 3'],
    )

    fig = image_overview(three_image_set)
    assert isinstance(fig, plt.Figure)
    assert np.array_equal(
        fig.get_size_inches(), np.array([9, 3]) # 3 columns * 3 width, 2 rows * 3 height
    )
    assert len(fig.axes) == 3
    assert fig.axes[0].get_title() == 'Image 1'

@pytest.fixture
def image_and_label_sets_dict():
    # create an img with shape 2 x 100 x 100
    image_data = np.random.rand(2, 100, 100)
    image_set1 = {
        'image': [image_data[0], image_data[1]],
        'colormap': ['PiYG', 'viridis'],
        'title': ['Image 1', 'Image 2'],
        'labels': [False, False],
        'min_display_intensity': [0, 0],
        'max_display_intensity': [1, 1],
    }

    # create a label image with only 1s and 0s with shape 100 x 100
    label_data = np.random.randint(0, 2, (100, 100))
    label_set = {
        'image': [None, label_data],
        'colormap': [None, 'Labels'], # tests the cmap.lower() component, this actually helped because it detected a lack of list
        'title': [None, 'Labels'],
    }

    return image_set1, label_set

@pytest.fixture
def image_and_label_sets_ImageSet():

    # create an img with shape 2 x 100 x 100
    image_data = np.random.rand(2, 100, 100)
    image_set1 = ImageSet(
        image=[image_data[0], image_data[1]],
        colormap=['PiYG', 'viridis'],
        title=['Image 1', 'Image 2'],
        labels=[False, False],
        min_display_intensity=[0, 0],
        max_display_intensity=[1, 1],
    )

    # create a label image with only 1s and 0s with shape 100 x 100
    label_data = np.random.randint(0, 2, (100, 100))
    label_set = ImageSet(
        image=[None, label_data],
        colormap=[None, 'Labels'], # tests the cmap.lower() component, this actually helped because it detected a lack of list
        title=[None, 'Labels'],
    )

    return image_set1, label_set

# test both the ImageSet and dictionary input
def test_image_overview_dict(image_and_label_sets_dict):
    fig = image_overview(image_and_label_sets_dict)

    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) == 4

    # Check the properties of each subplot
    assert fig.axes[0].get_title() == 'Image 1'
    assert fig.axes[0].get_images()[0].get_cmap().name == 'PiYG'

    assert fig.axes[1].get_title() == 'Image 2'
    assert fig.axes[1].get_images()[0].get_cmap().name == 'viridis'

    # fig. axes[2] should be empty
    assert not fig.axes[2].get_title()
    assert not fig.axes[2].get_images()

    assert fig.axes[3].get_title() == 'Labels'

def test_image_overview(image_and_label_sets_ImageSet):
    fig = image_overview(image_and_label_sets_ImageSet)

    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) == 4

    # Check the properties of each subplot
    assert fig.axes[0].get_title() == 'Image 1'
    assert fig.axes[0].get_images()[0].get_cmap().name == 'PiYG'

    assert fig.axes[1].get_title() == 'Image 2'
    assert fig.axes[1].get_images()[0].get_cmap().name == 'viridis'

    # fig. axes[2] should be empty
    assert not fig.axes[2].get_title()
    assert not fig.axes[2].get_images()

    assert fig.axes[3].get_title() == 'Labels'

def test_image_overview_plot_scale(image_and_label_sets_ImageSet):
    fig = image_overview(image_and_label_sets_ImageSet, fig_scale=(5, 6))

    assert isinstance(fig, plt.Figure)
    assert np.array_equal(
        fig.get_size_inches(), np.array([10, 12])
    ) # 2 columns * 5 width, 2 rows * 6 height

def test_image_overview_plot_title(image_and_label_sets_ImageSet):
    test_title = 'Test title'
    fig = image_overview(image_and_label_sets_ImageSet, fig_title=test_title)

    assert isinstance(fig, plt.Figure)
    assert fig._suptitle.get_text() == test_title

def test_add_scalebar_float(image_and_label_sets_ImageSet):
    fig = image_overview(image_and_label_sets_ImageSet)
    _add_scalebar(fig.axes[0], 0.5)

    assert isinstance(fig, plt.Figure)
    scalebar = [
        child for child in fig.axes[0].get_children()
        if isinstance(child, ScaleBar)
    ]
    assert len(scalebar) == 1

def test_add_scalebar_dict(image_and_label_sets_ImageSet):
    fig = image_overview(image_and_label_sets_ImageSet)
    scalebar_dict = {
        'dx': 0.25,
        'units': 'mm',
        'location': 'upper right',
        'badkey': 'badvalue',
    }
    _add_scalebar(fig.axes[0], scalebar_dict)

    assert isinstance(fig, plt.Figure)
    scalebar = [
        child for child in fig.axes[0].get_children()
        if isinstance(child, ScaleBar)
    ]
    assert len(scalebar) == 1

def test_image_overview_scalebar_float(image_and_label_sets_ImageSet):
    fig = image_overview(image_and_label_sets_ImageSet, scalebar=0.5)

    assert isinstance(fig, plt.Figure)

    scalebar_list = []
    for ax in fig.axes:
        scalebar = [
            child for child in ax.get_children()
            if isinstance(child, ScaleBar)
        ]
        scalebar_list.append(scalebar)

    assert len(scalebar_list) == 4
    assert len(scalebar_list[0]) == 1
    assert len(scalebar_list[1]) == 1
    assert len(scalebar_list[2]) == 0 # has no ax to add scalebar to
    assert len(scalebar_list[3]) == 1

def test_image_overview_scalebar_dict(image_and_label_sets_ImageSet):
    scalebar_dict = {
        'dx': 0.5,
        'units': 'mm',
        'location': 'upper right',
        'badkey': 'badvalue',
    }
    fig = image_overview(image_and_label_sets_ImageSet, scalebar=scalebar_dict)

    assert isinstance(fig, plt.Figure)

    # get scalebar from the first axis, multi-axes tested in float
    scalebar = [
        child for child in fig.axes[0].get_children()
        if isinstance(child, ScaleBar)
    ]
    assert len(scalebar) == 1


def test_imageoverview_init(image_and_label_sets_ImageSet):
    im = ImageOverview(image_and_label_sets_ImageSet, show=False)
    assert isinstance(im.fig, plt.Figure)


def test_imageoverview_save(image_and_label_sets_ImageSet):
    im = ImageOverview(image_and_label_sets_ImageSet, show=False)
    # test that the figure can be saved with .save()
    save_path = pathlib.Path(r'src\napari_ndev\_tests\resources')
    save_file_path = save_path / 'image_overview.png'

    im.save(str(save_path), 'image_overview.png')
    # assert that it was saved
    assert save_file_path.exists()
    # remove the saved file
    save_file_path.unlink()

def test_convert_dict_to_ImageSet():
    image_data = np.random.rand(2, 100, 100)
    image_set1 = {
        'image': [image_data[0], image_data[1]],
        'colormap': ['PiYG', 'viridis'],
        'title': ['Image 1', 'Image 2'],
        'labels': [False, False],
        'min_display_intensity': [0, 0],
        'max_display_intensity': [1, 1],
    }

    image_set = _convert_dict_to_ImageSet([image_set1])
    assert isinstance(image_set, list)
    assert isinstance(image_set[0], ImageSet)
    assert isinstance(image_set[0].image, list)
