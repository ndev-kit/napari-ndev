import numpy as np
import pytest

from napari_ndev import morphology


def test_convert_float_to_int():
    """Test the convert_float_to_int function."""
    float_image = np.array([[0.123, 0.234], [0.345, 0.456]])
    int_image = morphology.convert_float_to_int(float_image)

    assert np.issubdtype(float_image.dtype, np.floating)
    assert np.issubdtype(int_image.dtype, np.integer)
    assert int_image.dtype == np.uint32


label_2d = np.asarray([[0, 1, 1, 1], [2, 0, 1, 1], [2, 2, 0, 1], [2, 2, 1, 1]])
skeleton_label_2d = np.asarray([[0, 1, 0, 0], [2, 0, 1, 0], [0, 2, 0, 1], [0, 0, 1, 0]])
connected_label_2d = np.asarray([[0, 1, 1, 1], [1, 0, 1, 1], [1, 1, 0, 1], [1, 1, 1, 1]])

def test_skeletonize_labels():
    """Test the skeletonize_labels function."""
    skeleton = morphology.skeletonize_labels(label_2d)

    assert isinstance(skeleton, np.ndarray)
    assert skeleton.shape == label_2d.shape
    assert skeleton.dtype == np.uint16
    assert np.all(skeleton == skeleton_label_2d)

@pytest.mark.skip_ci
def test_connect_breaks_between_labels():
    """Test the connect_breaks_between_labels function."""
    connect_distance = 1.5
    connected_labels = morphology.connect_breaks_between_labels(label_2d, connect_distance)

    assert connected_labels.shape == label_2d.shape
    assert connected_labels.dtype == np.uint16
    assert np.all(connected_labels == connected_label_2d)

@pytest.mark.skip_ci
def test_label_voronoi_based_on_intensity():
    import pyclesperanto as cle
    """Test the label_voronoi_based_on_intensity function."""
    image = np.array([[10, 0, 1, 1], [0, 0, 1, 1], [1, 2, 1, 1], [2, 2, 10, 1]])
    labels = np.array([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]])
    voronoi_labels_exp = np.array([[1,1,1,2],[1,1,2,2,],[2,2,2,2],[2,2,2,2]])

    voronoi_labels = morphology.label_voronoi_based_on_intensity(
        label=labels, intensity_image=image
    )

    assert voronoi_labels.shape == labels.shape
    assert np.all(cle.pull(voronoi_labels) == voronoi_labels_exp)
