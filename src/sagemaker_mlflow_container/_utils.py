#
#    Copyright 2020 EPAM Systems
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
#
import json
import logging
import os
import subprocess
from os.path import join
from typing import Any, List, Mapping, MutableMapping

import yaml
from sagemaker_mlflow_container import const

MLPROJECT_FILE_NAME = "mlproject"
DEFAULT_CONDA_FILE_NAME = "conda.yaml"

logger = logging.getLogger(__name__)


def _conda_info(codna_env: str) -> Mapping:
    """
    Return json response of `conda run -n $conda_env info` command
    :param codna_env:
    :return:
    """
    process = subprocess.run(
        ['conda', 'run', '-n', codna_env, 'conda', 'info', '--json'],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True
    )
    info = json.loads(process.stdout)
    return info


def _get_conda_env_bin_path(conda_env: str) -> str:
    """
    Return absolute path to codna environment bin
    :param conda_env: Name of conda env the bin folder is looking for
    :return:
    """
    info = _conda_info(conda_env)
    return os.path.join(info[const.CONDA_INFO_ENV_PATH_KEY], 'bin')


def _copy_environ_and_prepend_path(new_path: str) -> Mapping:
    """
    Copy os.environ() and prepend `PATH` variable with `new_path`
    :param new_path: new path to prepend
    :return:
    """
    overridden_enc = os.environ.copy()
    overridden_enc['PATH'] = f'{new_path}:{overridden_enc["PATH"]}'
    return overridden_enc


def _find_mlproject_file_path(ml_project_dir) -> str:
    """
    Looks for file where MLFlow project meta-information is set
    :param ml_project_dir:
    :return:
    """
    filenames = os.listdir(ml_project_dir)
    for filename in filenames:
        if filename.lower() == MLPROJECT_FILE_NAME:
            return join(ml_project_dir, filename)

    raise ValueError(f"Can't find MLProject file in the '{ml_project_dir}' dir")


def _extract_conda_file_name(mlproject_file_path: str) -> str:
    """
    Extract conda dependencies file name from MLFlow project file
    :param mlproject_file_path: MLFlow MLProject file path
    :return: conda file name
    """
    with open(mlproject_file_path) as f:
        ml_project = yaml.safe_load(f)

        return ml_project.get("conda_env", DEFAULT_CONDA_FILE_NAME)


def _split_run_params(mp: Mapping[str, Any]) -> MutableMapping[str, Any]:
    """
    Split from mapping `sagemaker_mlflow_run_*` prefixed hyperparameters

    >>> hps = {"sagemaker_mlflow_run_experiment-id": 2, "another_param": 3}
    >>> _split_run_params(hps)
    {"experiment-id": 2}
    :param mp:
    :return:
    """
    run_params = {}
    for k, v in mp.items():
        if k.startswith(const.MLFLOW_RUN_PARAMS_PREFIX):
            _, param = k.split(const.MLFLOW_RUN_PARAMS_PREFIX, maxsplit=1)
            run_params[param] = v
    return run_params


def _mapping_to_mlflow_run_params(mp: Mapping) -> List[str]:
    """
    Transform parameters from Mapping to list for passing to cmd
    Use --key1 value1 --key2 value2 format

    If value = None than parameter will be included w/o value (Interpreted as a flag)

    >>> _mapping_to_mlflow_run_params({"experiment-id": 2, "no-conda": None})
    ["--experiment-id", "2", "--no-conda"]
    :param mp:
    :return:
    """
    param_list = []
    for k, v in mp.items():
        param_list.append(f'--{k}')
        if v is not None:  # otherwise it is flag
            param_list.append(str(v))
    return param_list


def _mapping_to_mlflow_hyper_params(mp: Mapping) -> List[str]:
    """
    Transform mapping to param-list arguments for `mlflow run ...` command
    Used to pass hyper-parameters to mlflow entry point (See MLFlow reference for more information)
    All mapping values will be converted to str(`value`)

    >>> _mapping_to_mlflow_hyper_params({"alpha": 1.0, 'epochs': 10})
    ["-P", "alpha=1.0", "-P", "epochs=10"]

    >>> result = _mapping_to_mlflow_hyper_params({"alpha": 1.0, 'epochs': 10})
    >>> assert isinstance(result, List)

    :param mp:
    :return:
    """
    param_list = []
    for k, v in mp.items():
        param_list.append('-P')
        param_list.append(f'{k}={v}')

    return param_list
