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
import logging
import subprocess
from os.path import join
from typing import Mapping, MutableMapping


import mlflow
from sagemaker_containers import _env, _files, training_env
from sagemaker_containers._env import TrainingEnv
from sagemaker_containers._errors import _CalledProcessError
from sagemaker_containers._process import check_error

from sagemaker_mlflow_container import const
from sagemaker_mlflow_container._checkers import _check_env
from sagemaker_mlflow_container._mlflow import _copy_mlflow_results_to_dir
from sagemaker_mlflow_container._utils import _copy_environ_and_prepend_path, _get_conda_env_bin_path, \
    _mapping_to_mlflow_run_params, \
    _extract_conda_file_name, \
    _find_mlproject_file_path, _mapping_to_mlflow_hyper_params, _split_run_params

logger = logging.getLogger(__name__)


def _update_codna_env(ml_project_dir: str):
    ml_project_file = _find_mlproject_file_path(ml_project_dir)
    conda_fp = _extract_conda_file_name(ml_project_file)
    abs_conda_fp = join(ml_project_dir, conda_fp)
    logger.info(f'Found MLproject conda file with dependencies: {abs_conda_fp}')
    logger.info(f'Start to update {const.CONDA_TRAINING_ENV} conda env using {conda_fp} file')

    check_error([
        'conda', 'env', 'update', '-n', const.CONDA_TRAINING_ENV, '-f', abs_conda_fp
    ], _CalledProcessError, capture_error=True)


def _run_training(ml_project_dir: str, hyper_params: Mapping, run_parameters: MutableMapping) -> str:
    """
    Run MLFlow training in separate `training` environment
    :param ml_project_dir: Path to MLFlow project directory where MLFlow code is located
    :param hyper_params: model hyper parameters that will be passed as MLFLow parameters to training script
    :param run_parameters: run parameters that will be passed to `mlflow run ...` as arguments
    :return: MLFlow run_id
    """

    cmd = [
        'conda', 'run', '-n', const.CONDA_TRAINING_ENV,
        'mlflow', 'run'
    ]

    if run_parameters:
        cmd += _mapping_to_mlflow_run_params(run_parameters)

    if hyper_params:
        cmd += _mapping_to_mlflow_hyper_params(hyper_params)

    # Because conda run -n $codna_name ... – not overrides PATH search priority correctly
    # we override it manually to ensure that appropriate python executable will be selected
    # to run the training
    bin_path = _get_conda_env_bin_path(const.CONDA_TRAINING_ENV)
    new_env = _copy_environ_and_prepend_path(bin_path)

    run_id = run_parameters.pop(const.MLFLOW_RUN_ID_PARAM, None)
    with mlflow.start_run(run_id) as run:
        cmd += ['--run-id', run.info.run_id, ml_project_dir]
        subprocess.run(cmd, check=True, env=new_env, stderr=subprocess.STDOUT)

    return run.info.run_id


def _save_results(run_id: str, output_dir: str):
    _copy_mlflow_results_to_dir(run_id, output_dir)


def train(train_env: TrainingEnv):
    """

    :param train_env:
    :return:
    """

    code_dir = _env.code_dir

    logger.info('Download code')
    _files.download_and_extract(train_env.module_dir, code_dir)

    logger.info('Checking environment')
    _check_env()

    logger.info(f'Update {const.CONDA_TRAINING_ENV} conda env using MLProject dependencies')
    _update_codna_env(code_dir)

    logger.info('Run training')
    run_params = _split_run_params(train_env.additional_framework_parameters)
    run_id = _run_training(code_dir, train_env.hyperparameters, run_params)

    logger.info('Save results')
    _save_results(run_id, train_env.model_dir)


def main():
    train(training_env())


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
