import json
import os
import tarfile

from sagemaker.estimator import Estimator, DIR_PARAM_NAME

from sagemaker_mlflow_container.const import SAGEMAKER_MODEL_SUBDIR


def test_local_run(resources_folder, sagemaker_role, image, tmpdir):

    mlflow_project_dir = 'file://' + os.path.join(resources_folder, 'ml/code')
    output_dir = 'file://' + str(tmpdir)

    estimator = Estimator(image_name=image,
                          hyperparameters={
                              DIR_PARAM_NAME: json.dumps(mlflow_project_dir),
                              'alpha': '1.0',
                              'sagemaker_mlflow_experiment_id': '2.0',
                          },
                          role=sagemaker_role,
                          output_path=output_dir,
                          train_instance_count=1,
                          train_instance_type='local')

    estimator.fit()

    _assert_files_exist_in_tar(output_dir, [
        f'{SAGEMAKER_MODEL_SUBDIR}/model/MLmodel',
        f'{SAGEMAKER_MODEL_SUBDIR}/model/model.pkl',
    ])


def _assert_files_exist_in_tar(output_path, files):
    if output_path.startswith('file://'):
        output_path = output_path[7:]
    model_file = os.path.join(output_path, 'model.tar.gz')
    with tarfile.open(model_file) as tar:
        for f in files:
            tar.getmember(f)
