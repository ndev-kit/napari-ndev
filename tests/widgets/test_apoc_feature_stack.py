import pytest

from napari_ndev.widgets._apoc_feature_stack import ApocFeatureStack


# Test CustomAPOC
def test_generate_feature_string_button(make_napari_viewer):
    wdg = ApocFeatureStack(make_napari_viewer())
    wdg._original.value = True
    wdg._generate_string_button.clicked.emit()
    assert wdg._feature_string.value != ''


def test_generate_features_none(make_napari_viewer):
    wdg = ApocFeatureStack(make_napari_viewer())
    wdg.generate_feature_string()
    assert wdg._feature_string.value == ''


def test_generate_feature_string_original(make_napari_viewer):
    wdg = ApocFeatureStack(make_napari_viewer())
    wdg._original.value = True
    wdg.generate_feature_string()
    assert 'original' in wdg._feature_string.value


@pytest.mark.parametrize(
    ('input_value', 'expected_output'),
    [
        ('3', 'gaussian_blur=3'),
        ('3,4,5', 'gaussian_blur=3 gaussian_blur=4 gaussian_blur=5'),
        ('3, 4,   5', 'gaussian_blur=3 gaussian_blur=4 gaussian_blur=5'),
    ],
)
def test_generate_feature_string_gaussian_blur(
    make_napari_viewer, input_value, expected_output
):
    wdg = ApocFeatureStack(make_napari_viewer())
    wdg._gaussian_blur.value = input_value
    wdg.generate_feature_string()
    assert wdg._feature_string.value == expected_output
