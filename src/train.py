from common.get_data import DataIngestion
from common.get_train import TrainData
import mlflow
from common import read_yaml
import argparse
import pandas as pd
import matplotlib.pyplot as plt

def train(config_path):
    tr=TrainData(config_path)
    mlflow.set_tracking_uri("http://127.0.0.1:5000",)
    c=read_yaml(config_path)
    mlflow.log_params({'batch':c.DATA_INGESTION.BATCH,'shuffle':c.DATA_INGESTION.SHUFFLE,'epochs':c.DATA_INGESTION.REPEAT})


    model=tr._get_model_()
    history=tr.train()
    plt.plot(pd.DataFrame(history.history))
    plt.savefig('./train_and_validation_curve.png')
    mlflow.log_artifact('./train_and_validation_curve.png',artifact_path='plots')
        
    mlflow.keras.log_model(model,artifact_path='models')
    

    

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--config','-c',default='config/config.yaml')
    args=parser.parse_args()
    train(args.config)