from common.get_data import DataIngestion
from common.get_train import TrainData
import mlflow
from common import read_yaml
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from mlflow import MlflowClient
import tensorflow as tf
import os
import numpy as np
from sklearn.metrics import accuracy_score,confusion_matrix
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def train(config_path):
    tr=TrainData(config_path)
    mlflow.set_tracking_uri("http://127.0.0.1:5000",)
    c=read_yaml(config_path)
    mlflow.log_params({'batch':c.DATA_INGESTION.BATCH,'shuffle':c.DATA_INGESTION.SHUFFLE,'epochs':c.DATA_INGESTION.REPEAT})


    model=tr._get_model_()
    history=tr.train(model)
    plt.plot(pd.DataFrame(history.history))
    plt.savefig('./train_and_validation_curve.png')
    mlflow.log_artifact('./train_and_validation_curve.png',artifact_path='plots')
        
    mlflow.keras.log_model(model,artifact_path='models')
    print('-------------------------------------------------------------------------')
    
    
    test_data=DataIngestion(args.config).get_test_data(c.DATA_INGESTION.BATCH,c.DATA_INGESTION.REPEAT,c.DATA_INGESTION.SHUFFLE)


    #mlflow.set_tracking_uri('http://127.0.0.1:5000')


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
    #mlflow.log_params({'batch':c.DATA_INGESTION.BATCH,'epoch':c.DATA_INGESTION.REPEAT,'shuffle':c.DATA_INGESTION.SHUFFLE})
    model=mlflow.keras.load_model(model_path)
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

    

    

    

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--config','-c',default='config/config.yaml')
    args=parser.parse_args()
    train(args.config)