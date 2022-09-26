import tensorflow as tf
from common import read_yaml,logger
import os






class DataIngestion:
    def __init__(self,config_path):
        self.log=logger
        self.config=read_yaml(config_path)
        self.train_data=os.path.join(self.config.DATA_ROOT,self.config.TRAIN_DATA)
        self.test_data=os.path.join(self.config.DATA_ROOT,self.config.TEST_DATA)
        self.val_data=os.path.join(self.config.DATA_ROOT,self.config.VAL_DATA)
        self.log.info('DataIngestion Initialized')
    
    def _parse_records(self,example_proto):
        feature_description={
        'image':tf.io.FixedLenFeature([],tf.string),
        'label':tf.io.FixedLenFeature([],tf.string)}
        feature=tf.io.parse_single_example(example_proto,feature_description)
        image=tf.reshape(tf.io.decode_raw(feature['image'],tf.float32),[224,224,3])
        label=tf.cast(tf.io.decode_raw(feature['label'],tf.float64),tf.int64)
    
    #label=tf.cast(feature['label'],tf.int32)
        return image,label

    def get_train_data(self,batch,repeat,shuffle):
        self.dataset=tf.data.TFRecordDataset(self.train_data,num_parallel_reads=4)
        self.log.info('train dataset has been made')
        return self.dataset.map(self._parse_records).shuffle(shuffle).repeat(repeat).batch(batch).prefetch(1)
    
    def get_test_data(self,batch,repeat,shuffle):
        self.dataset=tf.data.TFRecordDataset(self.test_data,num_parallel_reads=4)
        self.log.info('test dataset has been made')
        return self.dataset.map(self._parse_records).shuffle(shuffle).repeat(repeat).batch(batch).prefetch(1)

    def get_val_data(self,batch,repeat,shuffle):
        self.dataset=tf.data.TFRecordDataset(self.val_data,num_parallel_reads=4)
        self.log.info('val dataset has been made')
        return self.dataset.map(self._parse_records).shuffle(shuffle).repeat(repeat).batch(batch).prefetch(1)