import mlflow
from pathlib import Path
from mlflow import MlflowClient
import tensorflow as tf
from common import read_yaml
import os
import argparse
import numpy as np
from common.get_data import DataIngestion
from sklearn.metrics import accuracy_score,confusion_matrix
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# def print_auto_logged_info(r):
#     tags = {k: v for k, v in r.data.tags.items() if not k.startswith("mlflow.")}
#     artifacts = [f.path for f in MlflowClient().list_artifacts(r.info.run_id, "model")]
#     print("run_id: {}".format(r.info.run_id))
#     print("artifacts: {}".format(artifacts))
#     print("params: {}".format(r.data.params))
#     print("metrics: {}".format(r.data.metrics))
#     print("tags: {}".format(tags))



if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--config','-c',default='config/config.yaml')
    args=parser.parse_args()
    c=read_yaml(args.config)
    test_data=DataIngestion(args.config).get_test_data(c.DATA_INGESTION.BATCH,c.DATA_INGESTION.REPEAT,c.DATA_INGESTION.SHUFFLE)


    mlflow.set_tracking_uri('http://127.0.0.1:5000')


    # mlflow.log_param('stage', 'validate')
    # run=mlflow.start_run()
    # print(run.data)
    # client = mlflow.MlflowClient()
    # data = client.get_run(mlflow.active_run().info.run_id).data
    # #mlflow.create_experiment('AutoPytorch',artifact_location='./artifacts')
    # mlflow.set_experiment("AutoPytorch",)
    model_path=mlflow.get_artifact_uri('models')
    print(model_path)
    print(os.path.join(model_path,'/data/model/'))
    mlflow.log_param('model_path',model_path)
    mlflow.log_params({'batch':c.DATA_INGESTION.BATCH,'epoch':c.DATA_INGESTION.REPEAT,'shuffle':c.DATA_INGESTION.SHUFFLE})
    model=mlflow.keras.load_model(f'{model_path}/data/model/',)
    print('______________________ model has been loaded____________________')
    #model(test_data)
    yTrue,ypred=[],[]
    for batch in test_data:
        yTrue+=batch[1].numpy().reshape(-1).tolist()
        ypred+=np.argmax(model(batch[0].numpy()),axis=1).reshape(-1).tolist()
    
    accuracy=accuracy_score(y_true=yTrue,y_pred=ypred)
    mlflow.log_metric('Accuracy',accuracy)
    cm=confusion_matrix(y_true=yTrue,y_pred=ypred)
    cm_df = pd.DataFrame(cm,
                     index = ['Covid-19','Lung_opacity','Normal','Viral_Pneumonia','Tuberculosis'], 
                     columns = ['Covid-19','Lung_opacity','Normal','Viral_Pneumonia','Tuberculosis'])
    

    plt.figure(figsize=(5,4))
    sns.heatmap(cm_df, annot=True)
    plt.title('Confusion Matrix')
    plt.ylabel('Actal Values')
    plt.xlabel('Predicted Values')
    plt.savefig('./confusion_matrix.png')
    mlflow.log_artifact('./confusion_matrix.png',artifact_path='validation_plots')





    
    


    #print(mlflow.get_artifact_uri(artifact_path='plots'))
    