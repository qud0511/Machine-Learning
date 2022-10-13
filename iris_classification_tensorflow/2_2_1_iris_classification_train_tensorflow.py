# 파일명: iris_classification_train_tensorflow.py
import pickle
import logging
from iris_classification_train_tensorflow_sub import *


#tf.logging.set_verbosity('INFO')
#tf.compat.v1.logging.set_verbosity('INFO')

def train(tm):

    logging.info('[hunmin log] the start line of the function [train]')

    exec_train(tm)
    
    logging.info('[hunmin log] the end line of the function [train]')



def init_svc(im):

    logging.info('[hunmin log] the start line of the function [init_svc]')

    params = exec_init_svc(im)
    
    logging.info('[hunmin log] the end line of the function [init_svc]')
    
    return params



def inference(df, params, batch_id):

    logging.info('[hunmin log] the start line of the function [inference]')

    result = exec_inference(df, params, batch_id)
    
    logging.info('[hunmin log] the end line of the function [inference]')

    return {"data": result}


