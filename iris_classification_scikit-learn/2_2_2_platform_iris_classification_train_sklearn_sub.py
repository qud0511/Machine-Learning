# 파일명: iris_classification_train_sk_sub.py

#Imports
import tensorflow as tf
import os
import pandas as pd
import sklearn.svm as svm
import pickle
import logging
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def exec_train(tm):
    ###########################################################################
    ## 1. 데이터 세트 준비(Data Setup)
    ###########################################################################     
    logging.info("[hunmin log] train_data_path = {}".format(tm.train_data_path))    
    logging.info("[hunmin log] model_path = {}".format(tm.model_path))
    df = pd.read_csv(os.path.join(tm.train_data_path, os.listdir(tm.train_data_path)[-1]))
    
    ###########################################################################
    ## 2. 데이터 전처리(Data Preprocessing)
    ###########################################################################
    x_train, x_test, y_train, y_test = train_test_split(df.iloc[:,:-1], df.iloc[:,-1], test_size=0.2, random_state=777)
    
    ###########################################################################
    ## 3. 학습 모델 훈련(Train Model)
    ###########################################################################
    model = svm.SVC(kernel='linear').fit(x_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(x_test))
    logging.info("[hunmin log] model = {}".format(model))
    logging.info("[hunmin log] accuracy = {}".format(accuracy))
    logging.info('[hunmin log] model_build')
    
    ###########################################################################
    ## 학습 모델 저장
    ###########################################################################
    pickle.dump(model, open(os.path.join(tm.model_path, 'model.p'), 'wb'))
    
    display_result(tm, model, x_test, y_test)

def exec_inference(df, params, batch_id):
    ###########################################################################
    ## 4. 추론(Inference)
    ###########################################################################
    # 학습 모델을 준비한다.
    model = params['model']
    logging.info('[hunmin log] model : {}'.format(model))
    results = {}    
    # 결과를 예측한다.
    scores = model.predict(df)
    logging.info('[hunmin log] scores : {}'.format(scores))
    results['inference'] = scores.tolist()
    return results

# exec_train(tm)에서 사용하는 시각화 함수
def display_result(tm, model, x_test, y_test):
    ###########################################################################
    ## 플랫폼 시각화
    ###########################################################################
     # Accuracy, Loss, Confusion Matrix, Precision/Recall/F1-score
    val_accuracy = accuracy_score(y_test, model.predict(x_test))
    val_loss = 0

    eval_results={}
    predict_y = model.predict(x_test).tolist()
    actual_y = y_test.tolist()
    logging.info('[hunmin log] predict_y : {}'.format(predict_y))
    logging.info('[hunmin log] actual_y : {}'.format(actual_y))
    
    eval_results['predict_y'] = predict_y
    eval_results['actual_y'] = actual_y
    eval_results['loss']= val_loss
    eval_results['accuracy'] = val_accuracy

    # confusion_matrix 계산(eval_results)
    eval_results['confusion_matrix'] = metrics.confusion_matrix(actual_y, predict_y).tolist()
    tm.save_result_metrics(eval_results)