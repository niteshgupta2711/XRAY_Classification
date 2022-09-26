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
    run=mlflow.set_tracking_uri("http://127.0.0.1:5000")
    print(run)

    #run=mlflow.active_run()
        # mlflow.run(".", "ingest_data", use_conda=False)
        # mlflow.run(".", "train", use_conda=False)
        # mlflow.run(".", "validate", use_conda=False)
        #run=mlflow.ac
    #print(mlflow.get_artifact_uri(artifact_path='./artifacts'))
    #print_auto_logged_info(mlflow.get_run(run_id=run.info.run_id))

        #mlflow.run(".", "stage_04", use_conda=False)

if __name__ == '__main__':
    main()