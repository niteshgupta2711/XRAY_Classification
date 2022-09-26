import mlflow
from pathlib import Path
from mlflow import MlflowClient
# def print_auto_logged_info(r):
#     tags = {k: v for k, v in r.data.tags.items() if not k.startswith("mlflow.")}
#     artifacts = [f.path for f in MlflowClient().list_artifacts(r.info.run_id, "model")]
#     print("run_id: {}".format(r.info.run_id))
#     print("artifacts: {}".format(artifacts))
#     print("params: {}".format(r.data.params))
#     print("metrics: {}".format(r.data.metrics))
#     print("tags: {}".format(tags))



if __name__=='__main__':
    mlflow.set_tracking_uri('127.0.0.1:5000',)
    print(mlflow.get_tracking_uri())
    