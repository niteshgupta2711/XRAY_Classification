from common import loggger, read_yaml
import argparse
import mlflow
import os
import tensorflow as tf
from common import DataIngestion


"""
The following module provides you with a tf.data.Dataset object that allows incremental load with tfrecord dataset
 tfrecords are way huge on disk
 given these are stored as raw bytes
 tfrecords are light weight and highly optimized for streaming large datasets
 These tfrecords are main format tensorflow distributed training strategies


"""
def main(path):
    mlflow.set_tracking_uri("http://127.0.0.1:5000",)
    loggger.info('tracking uri has been set ')
    data_config=read_yaml(path)
    mlflow.log_params({'data_config':path,
    'data_root':data_config.DATA_ROOT, 'train_record':data_config.TRAIN_DATA,'test_records':data_config.TEST_DATA 
    })











if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--config','-c',type=str,default='config/config.yaml')
    args=parser.parse_args()
    main(args.config)
    
