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

from sagemaker_containers._process import check_error
from sagemaker_mlflow_container import const
from sagemaker_mlflow_container.errors import CondaIsNotInstalled, CondaTrainingEnvIsNotCreated, \
    MLFlowIsNotInstalledInConda

logger = logging.getLogger(__name__)


def _check_training_env():
    check_error([
        'conda', 'run', '-n', const.CONDA_TRAINING_ENV, 'conda', 'info'
    ], CondaTrainingEnvIsNotCreated, capture_error=True)


def _check_conda():

    check_error(['conda', '--version'], CondaIsNotInstalled, capture_error=True)


def _check_mlflow():
    """
    Conda
    :return:
    """
    check_error([
        'conda', 'run', '-n', const.CONDA_TRAINING_ENV, 'mlflow', '--version'
    ], MLFlowIsNotInstalledInConda, capture_error=True)


def _check_env():
    """
    To ensure correct behavior of package we should check os env where it is launched
    :return:
    """
    _check_conda()
    logger.info('OK – Conda binary found')

    _check_training_env()
    logger.info(f'OK – Conda env to run mlflow training found, {const.CONDA_TRAINING_ENV}')

    _check_mlflow()
    logger.info(f'OK – mlflow binary found for {const.CONDA_TRAINING_ENV}')
