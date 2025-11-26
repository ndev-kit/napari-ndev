import pathlib

import numpy as np
from ndevio import nImage

from napari_ndev.widgets._workflow_container import WorkflowContainer

# from napari_workflows._io_yaml_v1 import load_workflow

class MockWorkflow:
    def roots(self):
        return ['root1', 'root2']

    def leafs(self):
        return ['leaf1', 'leaf2']

    def set(self, name, func_or_data):
        pass

    def get(self, name):
        pass

def test_workflow_container_init_no_viewer():
    container = WorkflowContainer()

    assert container._viewer is None
    assert container._channel_names == []
    assert container._img_dims == ''

def test_workflow_container_init_with_viewer(make_napari_viewer):
    viewer = make_napari_viewer()
    container = WorkflowContainer(viewer)

    assert container._viewer == viewer
    assert container._channel_names == []
    assert container._img_dims == ''


def test_workflow_container_update_roots(make_napari_viewer):
    viewer = make_napari_viewer()
    container = WorkflowContainer(viewer)

    container.workflow = MockWorkflow()
    container._channel_names = ['red', 'green', 'blue']

    container._update_roots()
    # check that there are two widgets created
    assert len(container._batch_roots_container) == 2
    assert len(container._viewer_roots_container) == 2

    for idx, root in enumerate(container._batch_roots_container):
        assert root.label == f'Root {idx}: {container.workflow.roots()[idx]}'
        assert root.choices == (None, 'red', 'green', 'blue')
        assert root._nullable is True
        assert root.value is None

    for idx, root in enumerate(container._viewer_roots_container):
        assert root.label == f'Root {idx}: {container.workflow.roots()[idx]}'
        assert root.choices == (None,)
        assert root._nullable is True
        assert root.value is None

    # test that _update_layer_choices is called when the viewer is updated
    container._viewer.open_sample('napari', 'cells3d')
    container._viewer.add_labels(np.random.randint(0, 2, (10, 10, 10)))

    for root in container._viewer_roots_container:
        assert root.choices == (
            None, viewer.layers[0], viewer.layers[1], viewer.layers[2]
        )


def test_workflow_container_get_workflow_info():
    container = WorkflowContainer()
    wf_path = pathlib.Path(
        'src/napari_ndev/_tests/resources/Workflow/workflows/'
        'cpu_workflow-2roots-2leafs.yaml'
    )
    container.workflow_file.value = wf_path

    assert container._workflow_roots.value == str(container.workflow.roots())
    assert len(container._batch_roots_container) == len(container.workflow.roots())
    assert len(container._viewer_roots_container) == len(container.workflow.roots())
    assert container._tasks_select.value == list(container.workflow.leafs())
    assert list(container._tasks_select.choices) == list(container.workflow._tasks.keys())

def test_update_progress_bar():
    container = WorkflowContainer()
    container._progress_bar.value = 0
    container._progress_bar.max = 10
    container._update_progress_bar(9)
    assert container._progress_bar.value == 9

def test_batch_workflow_not_threaded(tmp_path):
    container = WorkflowContainer()
    wf_path = pathlib.Path(
        'src/napari_ndev/_tests/resources/Workflow/workflows/'
        'cpu_workflow-2roots-2leafs.yaml'
    )
    container.workflow_file.value = wf_path

    container.image_directory.value = pathlib.Path(
        'src/napari_ndev/_tests/resources/Workflow/Images'
    )

    output_folder = tmp_path / 'Output'
    output_folder.mkdir()
    container.result_directory.value = output_folder

    container._batch_roots_container[0].value = 'membrane'
    container._batch_roots_container[1].value = 'nuclei'

    # test the _batch_workflow_threaded generator method
    generator = container.batch_workflow()

    for _ in generator:
        pass

    assert output_folder.exists()
    assert (output_folder / 'cells3d2ch.tiff').exists()
    assert (output_folder / 'workflow.log.txt').exists()

def test_batch_workflow_leaf_tasks(tmp_path, qtbot):
    container = WorkflowContainer()
    wf_path = pathlib.Path(
        'src/napari_ndev/_tests/resources/Workflow/workflows/'
        'cpu_workflow-2roots-2leafs.yaml'
    )
    container.workflow_file.value = wf_path

    container.image_directory.value = pathlib.Path(
        'src/napari_ndev/_tests/resources/Workflow/Images'
    )

    output_folder = tmp_path / 'Output'
    output_folder.mkdir(exist_ok=True)
    container.result_directory.value = output_folder

    container._batch_roots_container[0].value = 'membrane'
    container._batch_roots_container[1].value = 'nuclei'

    container.batch_workflow_threaded()

    # wait for multithreading to complete
    with qtbot.waitSignal(container._batch_worker.finished, timeout=10000):
        pass

    # confirm a value was yielded by batch_workflow
    assert container._progress_bar.value == 1
    # output folder does exist
    assert output_folder.exists()
    assert (output_folder / 'cells3d2ch.tiff').exists()

    img = nImage(output_folder / 'cells3d2ch.tiff')
    assert len(img.channel_names) == 2

def test_batch_workflow_keep_original_images(tmp_path, qtbot):
    container = WorkflowContainer()
    wf_path = pathlib.Path(
        'src/napari_ndev/_tests/resources/Workflow/workflows/'
        'cpu_workflow-2roots-2leafs.yaml'
    )
    container.workflow_file.value = wf_path

    container.image_directory.value = pathlib.Path(
        'src/napari_ndev/_tests/resources/Workflow/Images'
    )

    output_folder = tmp_path / 'Output'
    output_folder.mkdir()
    container.result_directory.value = output_folder

    container._batch_roots_container[0].value = 'membrane'
    container._batch_roots_container[1].value = 'nuclei'

    container._keep_original_images.value = True
    container.batch_button.clicked()

    # wait for multithreading to complete
    with qtbot.waitSignal(container._batch_worker.finished, timeout=10000):
        pass

    assert output_folder.exists()
    assert (output_folder / 'cells3d2ch.tiff').exists()

    img = nImage(output_folder / 'cells3d2ch.tiff')
    assert len(img.channel_names) == 4
    assert img.channel_names == [
        'membrane', 'nuclei', 'membrane-label', 'nucleus-label'
    ]

def test_batch_workflow_all_tasks(tmp_path, qtbot):
    container = WorkflowContainer()
    wf_path = pathlib.Path(
        'src/napari_ndev/_tests/resources/Workflow/workflows/'
        'cpu_workflow-2roots-2leafs.yaml'
    )
    container.workflow_file.value = wf_path

    container.image_directory.value = pathlib.Path(
        'src/napari_ndev/_tests/resources/Workflow/Images'
    )

    output_folder = tmp_path / 'Output'
    output_folder.mkdir()
    container.result_directory.value = output_folder

    container._batch_roots_container[0].value = 'membrane'
    container._batch_roots_container[1].value = 'nuclei'

    container._tasks_select.value = list(container.workflow._tasks.keys())

    container.batch_workflow_threaded()

    # wait for multithreading to complete
    with qtbot.waitSignal(container._batch_worker.finished, timeout=10000):
        pass

    assert output_folder.exists()
    assert (output_folder / 'cells3d2ch.tiff').exists()

    img = nImage(output_folder / 'cells3d2ch.tiff')
    assert len(img.channel_names) == 6

def test_viewer_workflow(make_napari_viewer):
    viewer = make_napari_viewer()
    container = WorkflowContainer(viewer)
    wf_path = pathlib.Path(
        'src/napari_ndev/_tests/resources/Workflow/workflows/'
        'cpu_workflow-2roots-2leafs.yaml'
    )
    container.workflow_file.value = wf_path

    container._viewer.open_sample('napari', 'cells3d')
    container._viewer_roots_container[0].value = viewer.layers['membrane']
    container._viewer_roots_container[1].value = viewer.layers['nuclei']

    generator = container.viewer_workflow()

    expected_results = [
        (0, 'membrane-label'),
        (1, 'nucleus-label'),
    ]
    # check that the generator yields a value
    for idx, (task_idx, task, result) in enumerate(generator):
        assert task_idx == expected_results[idx][0]
        assert task == expected_results[idx][1]
        assert isinstance(result, np.ndarray)

def test_viewer_workflow_yielded(make_napari_viewer):
    viewer = make_napari_viewer()
    container = WorkflowContainer(viewer)
    data = np.random.randint(0, 2, (10, 10, 10))

    value = (1, 'test-name', data)
    container._viewer_workflow_yielded(value)

    assert container._progress_bar.value == 2 # idx + 1
    assert container._viewer.layers[0].name == 'test-name'
    assert container._viewer.layers[0].data.shape == data.shape
    assert np.array_equal(
        container._viewer.layers[0].scale, (1, 1, 1)
    )

def test_viewer_workflow_threaded(make_napari_viewer, qtbot):
    viewer = make_napari_viewer()
    container = WorkflowContainer(viewer)
    wf_path = pathlib.Path(
        'src/napari_ndev/_tests/resources/Workflow/workflows/'
        'cpu_workflow-2roots-2leafs.yaml'
    )
    container.workflow_file.value = wf_path

    container._viewer.open_sample('napari', 'cells3d')
    container._viewer_roots_container[0].value = viewer.layers['membrane']
    container._viewer_roots_container[1].value = viewer.layers['nuclei']

    container.viewer_workflow_threaded()

    # wait for multithreading to complete
    with qtbot.waitSignal(container._viewer_worker.finished, timeout=10000):
        pass

    assert container._progress_bar.value == 2
    assert container._viewer.layers[2].name == 'membrane-label'
    assert container._viewer.layers[3].name == 'nucleus-label'
