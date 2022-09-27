import mlflow
import argparse
def print_auto_logged_info(r):
    tags = {k: v for k, v in r.data.tags.items() if not k.startswith("mlflow.")}
    artifacts = [f.path for f in MlflowClient().list_artifacts(r.info.run_id, "model")]
    print("run_id: {}".format(r.info.run_id))
    print("artifacts: {}".format(artifacts))
    print("params: {}".format(r.data.params))
    print("metrics: {}".format(r.data.metrics))
    print("tags: {}".format(tags))


def main():
    mlflow.set_tracking_uri("http://127.0.0.1:5000",)
    mlflow.set_experiment("AutoPytorch",)
    #mlflow.create_experiment('AutoPytorch',artifact_location='./artifacts')
    current_experiment=dict(mlflow.get_experiment_by_name("AutoPytorch"))
    experiment_id=current_experiment['experiment_id']
    

    
    # import os
    # import pandas as pd
    # from mlflow.pipelines import Pipeline
    # from mlflow.pyfunc import PyFuncModel

    # regression_pipeline = Pipeline(profile="local")
    # regression_pipeline.run(step='ingest_data',)
    # regression_pipeline.inspect()
    # train_df: pd.DataFrame = regression_pipeline.get_artifact("models")
    # trained_model: PyFuncModel = regression_pipeline.get_artifact("model")
    # client = mlflow.MlflowClient()
    # data = client.get_run(mlflow.active_run().info.run_id).data
    # print(data)
    
    # run = mlflow.active_run()
    # print("Active run_id: {}".format(run.info.run_id))
    # mlflow.end_run()
    
    mlflow.run(".", "ingest_data", use_conda=False,experiment_id=experiment_id,run_name='making' )
    
    mlflow.run(".", "train", use_conda=False,experiment_id=experiment_id,run_name='making')
    mlflow.run(".", "validate", use_conda=False,experiment_id=experiment_id,run_name='making')
        #run=mlflow.ac
    #print(mlflow.get_artifact_uri(artifact_path='./artifacts'))
    #print_auto_logged_info(mlflow.get_run(run_id=run.info.run_id))

        #mlflow.run(".", "stage_04", use_conda=False)

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--config','-c',default='config/config.yaml')
    args=parser.parse_args()
    main()