from unittest.mock import patch

import pytest
from sagemaker_mlflow_container import const
from sagemaker_mlflow_container._utils import _conda_info, _copy_environ_and_prepend_path, \
    _extract_conda_file_name, _find_mlproject_file_path, _get_conda_env_bin_path, _mapping_to_mlflow_hyper_params, \
    _mapping_to_mlflow_run_params, _split_run_params


def test_conda_info():

    class StubProcess:
        stdout = '{"result": "ok"}'

    env_name = 'training'

    with patch('subprocess.run') as run_mock:

        run_mock.return_value = StubProcess

        info = _conda_info(env_name)

        args, _ = run_mock.call_args

        cmd, *_ = args

        assert cmd == ['conda', 'run', '-n', env_name, 'conda', 'info', '--json']

        assert info == {'result': 'ok'}


def test_get_conda_env_bin_path():

    stub_path = '/stub/path/to/env'

    with patch('sagemaker_mlflow_container._utils._conda_info') as info_mock:

        info_mock.return_value = {
            const.CONDA_INFO_ENV_PATH_KEY: stub_path
        }

        bin_path = _get_conda_env_bin_path('env_name')

    assert bin_path == f'{stub_path}/bin'


def test_copy_environ_and_prepend_path():

    stub_path = '/usr/bin:/usr/local/bin'

    def stub_environ():
        return {'PATH': stub_path}

    new_path = '/home/user/bin'

    with patch('os.environ', stub_environ()):
        overridden_env = _copy_environ_and_prepend_path(new_path)

    assert overridden_env == {'PATH': f'{new_path}:{stub_path}'}


def test_find_mlproject_file_path_ok(tmpdir):
    p = tmpdir.join('MLproject')
    p.write("")
    fp = _find_mlproject_file_path(tmpdir)
    assert fp == p


def test_find_mlproject_file_path_error(tmpdir):
    with pytest.raises(ValueError):
        _find_mlproject_file_path(tmpdir)


def test_extract_conda_file_name(tmpdir):
    with_conda_spec = tmpdir / 'with_conda_spec.yaml'
    without_conda_spec = tmpdir / 'without_conda_spec.yaml'

    with_conda_spec.write_text('version: 1\nconda_env: deps/my_conda.yaml\n', encoding='utf-8')
    assert _extract_conda_file_name(str(with_conda_spec)) == 'deps/my_conda.yaml'

    without_conda_spec.write_text('version: 1\n', encoding='utf-8')
    assert _extract_conda_file_name(str(without_conda_spec)) == "conda.yaml"


def test_split_run_params():
    actual = _split_run_params({"sagemaker_mlflow_run_experiment-id": 2, "another_param": 3})
    expected = {"experiment-id": 2}
    assert actual == expected


def test_mapping_to_mlflow_run_params():
    actual = _mapping_to_mlflow_run_params({"experiment-id": 2, "no-conda": None})
    expected = ["--experiment-id", "2", "--no-conda"]
    assert actual == expected


def test_mapping_to_mlflow_hyper_params():
    actual = _mapping_to_mlflow_hyper_params({"alpha": 1.0, 'epochs': 10})
    expected = ["-P", "alpha=1.0", "-P", "epochs=10"]
    assert actual == expected
