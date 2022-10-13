# 파일명: iris_classification_train_tensorflow_sub.py


#Imports
import tensorflow as tf
import logging
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.python.client import device_lib
import pickle
from tensorflow.keras.models import load_model
from sklearn import metrics


def exec_train(tm):
    
    # tm.train_data_path            전처리 완료된 데이터 저장 경로
    # tm.model_path 		        학습 수행 후 학습모델을 저장하는 경로
    
    logging.info('[hunmin log]  the start line of the function [exec_train]')
    logging.info('[hunmin log] tm.train_data_path : {}'.format(tm.train_data_path))
    
    # 저장 파일 확인
    list_files_directories(tm.train_data_path)
      
    ###########################################################################
    ## 1. 데이터 세트 준비(Data Setup)
    ###########################################################################
    train_path = tm.train_data_path
    df = pd.read_csv(os.path.join(train_path, os.listdir(train_path)[-1]))
    
    ###########################################################################
    ## 2. 데이터 전처리(Data Preprocessing)
    ## LabelEncoder를 이용해서 'Iris-setosa', 'Iris-versicolor', 'Iris-virginica'를 각각 0, 1, 2의 숫자형 레이블로 변환
    ###########################################################################
    X=df.iloc[:,:-1]
    y=df.iloc[:,-1:]
    label_encoder = LabelEncoder()
    y=label_encoder.fit_transform(y)  
    with open(os.path.join(tm.model_path, 'label_encoder.p'), 'wb') as f:
        pickle.dump(label_encoder, f)  
    X_train, X_test, Y_train, Y_test = train_test_split(X,y,test_size=0.2,random_state=0)
    
    ###########################################################################
    ## 3. 학습 모델 훈련(Train Model)
    ## Sequential을 불러와 1개 레이어를 사용하여 모델을 만들고, 입력층 1개 + 중간층 2개와# input_dim = 특성의 개수
    ## 출력 계층은 softmax 활성화 기능이 있는 3개 유닛(각 클래스에 1개)의 Dense Layer입니다.
    ###########################################################################
    model =tf.keras.models.Sequential(    [
      tf.keras.layers.Dense(16,input_dim=4, activation='relu'),  
      tf.keras.layers.Dense(32, activation='relu'),  
      tf.keras.layers.Dense(10, activation='relu'),
      tf.keras.layers.Dense(3, activation='softmax')
    ])

    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
    history = model.fit(X_train, Y_train, validation_data=(X_test,Y_test), epochs=100)
    logging.info("[hunmin] model = {}".format(model))
    logging.info('[hunmin] model_build')
 

    ###########################################################################
    ## 학습 모델 저장
    ###########################################################################
    
    logging.info('[hunmin log] tm.model_path : {}'.format(tm.model_path))
    model.save(os.path.join(tm.model_path, 'iris_model.h5'))

    # 저장 파일 확인
    list_files_directories(tm.model_path)
    ###########################################################################
    ## 학습 모델 display
    ###########################################################################
    display_result(tm, history, model, X_test, Y_test)
    
    
def exec_init_svc(im):

    logging.info('[hunmin log] im.model_path : {}'.format(im.model_path))

    # load the model
    model = load_model(os.path.join(im.model_path, 'iris_model.h5'))

    # loda the labelencoder data
    with open(os.path.join(im.model_path,'label_encoder.p'), 'rb') as f:
        label_encoder = pickle.load(f)
    
    return {"model":model,"label_encoder":label_encoder}

def exec_inference(df, params, batch_id):
    
    ###########################################################################
    ## 4. 추론(Inference)
    ###########################################################################

    logging.info('[hunmin log]  the start line of the function [exec_inference]')
    
    ## 학습 모델 준비
    model = params['model']
    label_encoder = params['label_encoder']
    logging.info('[hunmin] model : {}'.format(model))
    results = {} 
    logging.info('[hunmin] preresults : {}'.format(results))
    scores = model.predict(df)
    decoded_data=label_encoder.inverse_transform(np.argmax(scores, axis=1))
    logging.info('[hunmin] scores : {}'.format(scores))
    results['inference'] = decoded_data.tolist()
    logging.info('[hunmin] after results : {}'.format(results))

    logging.info('[hunmin log]  the finish line of the function [exec_inference]')
    return results
    
# 저장 파일 확인,exec_train에서 불러쓰는 함수
def list_files_directories(path):
    # Get the list of all files and directories in current working directory
    dir_list = os.listdir(path)
    logging.info('[hunmin log] Files and directories in {} :'.format(path))
    logging.info('[hunmin log] dir_list : {}'.format(dir_list))

#exec_train에서 불러쓰는 함수
def display_result(tm, history, model, x_test, y_test):
    for i in range(len(history.history['loss'])):#num_epochs
        metric={}
        metric['loss'] = history.history['loss'][i]
        metric['accuracy'] = history.history['accuracy'][i]
        metric['step'] = i
        tm.save_stat_metrics(metric)

    eval_results={}

    predict_y = model.predict(x_test).argmax(axis=1).tolist()
    actual_y = y_test.tolist()

    eval_results['predict_y'] = predict_y
    eval_results['actual_y'] = actual_y
    eval_results['loss']= history.history['val_loss'][-1]
    eval_results['accuracy'] = history.history['val_accuracy'][-1]

    # calculate_confusion_matrix(eval_results)
    eval_results['confusion_matrix'] = metrics.confusion_matrix(actual_y, predict_y).tolist()
    tm.save_result_metrics(eval_results)
    logging.info('[user log] display show')






