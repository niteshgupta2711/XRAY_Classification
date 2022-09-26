
import tensorflow as tf
from tensorflow.core.config.flags import config
from common.get_data import DataIngestion
from common import logger,read_yaml


class TrainData:
    def __init__(self,config:str):
        self.get_Data=DataIngestion(config)
        self.log=logger
        self.config=config

    
    def _get_model_(self,loss_func=tf.keras.losses.SparseCategoricalCrossentropy(),metrics=[tf.keras.metrics.Accuracy()],optim='adam'):
        self.conv1_input=tf.keras.Input(shape=(224,224,3),name='input_layer')
        self.conv2=tf.keras.layers.Conv2D(32,3,1,name='first_conv_layer',activation='relu')(self.conv1_input,)
        self.max_pool1=tf.keras.layers.MaxPool2D(2)(self.conv2)
        self.batch_n1=tf.keras.layers.BatchNormalization()(self.max_pool1)
        self.flatten=tf.keras.layers.Flatten()(self.batch_n1)
        self.dense_1=tf.keras.layers.Dense(256,activation='relu')(self.flatten)
        self.output=tf.keras.layers.Dense(5,activation='softmax')(self.dense_1)

        self.model=tf.keras.Model(self.conv1_input,self.output)
        self.model.compile(loss=loss_func,metrics=metrics,optimizer=optim,)

        return self.model
    
    def train(self):
        self.c=read_yaml(self.config)
        self.batch=self.c.DATA_INGESTION.BATCH
        self.epoch=self.c.DATA_INGESTION.REPEAT
        self.shuffle=self.c.DATA_INGESTION.SHUFFLE
        self.model=self._get_model_()
        self.log.info(f'model has been loaded {self.model.summary()}')
        self.validation_data=self.get_Data.get_val_data(self.batch,self.epoch,self.shuffle)
        self.history=self.model.fit(self.get_Data.get_train_data(self.batch,self.epoch,self.shuffle),epochs=self.epoch,batch_size=self.batch,validation_data=self.validation_data)
        return self.history
       





        
