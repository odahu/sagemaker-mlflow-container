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

import urllib3
import shutil
from os.path import join

import mlflow

from sagemaker_mlflow_container import const

logger = logging.getLogger(__name__)


def _copy_mlflow_results_to_dir(run_id: str, dir_: str):
    """
    Copy MLFlow run artifacts to directory
    :param run_id:
    :param dir_:
    :return:
    """
    artifact_uri: str = mlflow.get_run(run_id).info.artifact_uri
    url = urllib3.util.parse_url(artifact_uri)
    if url.scheme != 'file':
        raise NotImplementedError('Only local artifact storage is supported')

    result_dir = join(dir_, const.SAGEMAKER_MODEL_SUBDIR)
    shutil.copytree(url.path, join(dir_, result_dir))
    logger.info(f'MLFlow run: {run_id} artifacts were copied to {result_dir}')
