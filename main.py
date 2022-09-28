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
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("AutoPytorch")
    
    mlflow.run(".", "ingest_data", use_conda=False,)
    
    mlflow.run(".", "train", use_conda=False,)
    ##mlflow.run(".", "validate", use_conda=False,experiment_id=experiment_id,run_name='making')
    

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--config','-c',default='config/config.yaml')
    args=parser.parse_args()
    main()