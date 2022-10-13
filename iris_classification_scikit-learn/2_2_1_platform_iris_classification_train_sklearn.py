# 파일명 : iris_classification_train_sk.py

#Imports
from iris_classification_train_sk_sub import exec_train, exec_inference
from sklearn import metrics
from sklearn.metrics import accuracy_score
import logging
import pickle
import os

def train(tm):
    logging.info('[hunmin log] the start line of the function [train]')  
    exec_train(tm)
    logging.info('[hunmin log] the finish line of the function [train]')

def init_svc(im):
    model = pickle.load(open(os.path.join(im.model_path, 'model.p'), 'rb'))
    return {'model':model}
    
def inference(df, params, batch_id):
    logging.info('[hunmin log] the start line of the function [inference]')
    result = exec_inference(df, params, batch_id)
    logging.info('[hunmin log] the end line of the function [inference]')
    return result