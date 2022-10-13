# 파일명: iris_classification_train_tc.py

# import
from iris_classification_train_tc_sub import exec_train, exec_init_svc, exec_inference
import tensorflow as tf
import logging

def train(tm):
    
    exec_train(tm)
    
    logging.info('[hunmin log] the end line of the function [train]')



def init_svc(im):

    model = exec_init_svc(im)
    
    logging.info('[hunmin log] the end line of the function [init_svc]')
    
    return {"model": model}



def inference(df, params, batch_id):

    result = exec_inference(df, params, batch_id)
    
    logging.info('[hunmin log] the end line of the function [inference]')

    return result


