# 파일명: iris_classification_preprocess_tensorflow.py

from iris_classification_preprocess_tensorflow_sub import *
import pickle
import logging


def process_for_train(pm):

    logging.info('[hunmin log] the start line of the function [process_for_train]')
    exec_process(pm)
    logging.info('[hunmin log] the end line of the function [process_for_train]')
    


def init_svc(im, rule):

    meta_path = im.meta_path   
    return {"meta_path": meta_path, "rule": rule}



def transform(df, params, batch_id):

    logging.info('[hunmin log] df : {}'.format(df))
    logging.info('[hunmin log] df.shape : {}'.format(df.shape))
    logging.info('[hunmin log] type(df) : {}'.format(type(df)))   
    logging.info('[hunmin log] the end line of the function [transform]')
    
    return df


