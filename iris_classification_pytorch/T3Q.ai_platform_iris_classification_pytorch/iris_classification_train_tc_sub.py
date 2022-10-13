# 파일명: iris_classification_train_pytorch_sub.py


#Imports
import numpy as np
import pandas as pd
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import metrics

import tensorflow as tf
import logging

# (사용자 정의) 신경망(Neural Network) 생성
class Net(nn.Module):
    # define nn
    def __init__(self):
        super(Net, self).__init__()
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(4, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 3)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, X):
        X = F.relu(self.fc1(X))
        X = self.fc2(X)
        X = self.fc3(X)
        X = self.softmax(X)
    
        return X

def exec_train(tm):
    
    # tm.train_data_path            전처리 완료된 데이터 저장 경로
    # tm.model_path 		        학습 수행 후 학습모델을 저장하는 경로
      
    logging.info('[hunmin log]  the start line of the function [exec_train]')
    
    # 저장 파일 확인
    list_files_directories(tm.train_data_path)

    ###########################################################################
    ## 1. 데이터 세트 준비(Data Setup)
    ###########################################################################    
    train_path = tm.train_data_path
    # Load data
    dataset = pd.read_csv(os.path.join(train_path, os.listdir(train_path)[-1]))
    logging.info('[hunmin log] dataset : {}'.format(dataset))
    

    ###########################################################################
    ## 2. 데이터 전처리(Data Preprocessing)
    ###########################################################################
    
    # species을 numerics로 변환한다.
    # Iris-setosa = 0 / Iris-versicolor = 1 / Iris-virginica = 2
    dataset.loc[dataset.iloc[:,-1]=='Iris-setosa', dataset.columns[-1]] = 0
    dataset.loc[dataset.iloc[:,-1]=='Iris-versicolor', dataset.columns[-1]] = 1
    dataset.loc[dataset.iloc[:,-1]=='Iris-virginica', dataset.columns[-1]] = 2

    train_X, test_X, train_y, test_y = train_test_split(dataset[dataset.columns[0:4]].values, dataset.iloc[:,-1].values, test_size=0.2)

    # Pytorch에서 수학적 연산을 가속화하기 위해 tensor data를 사용한다.
    train_X = Variable(torch.Tensor(train_X).float())
    test_X = Variable(torch.Tensor(test_X).float())

    train_y = np.array(train_y, dtype=np.int64)
    test_y = np.array(test_y, dtype=np.int64)
    train_y = torch.from_numpy(train_y)
    test_y = torch.from_numpy(test_y)


    ###########################################################################
    ## 3. 학습 모델 훈련(Train Model)
    ###########################################################################
    
    # Net 클래스를 인스턴스화하고 model 객체를 생성합니다.
    model = Net()
    logging.info('[hunmin log] model : {}'.format(model))
    criterion = nn.CrossEntropyLoss()# cross entropy loss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # 
    loss_values = []
    accuracy_values = []
    
    correct = 0
    epochs = 100
    for epoch in range(1, epochs+1):
        running_loss = 0.0
        
        optimizer.zero_grad()
        output = model(train_X)
        loss = criterion(output, train_y)
        
        _, predict_y = torch.max(output, 1)

        # loss graph
        running_loss += loss.item()
        loss_values.append(running_loss)
    
        # #Accuracy
        correct = (predict_y == train_y).float().sum().detach().numpy()
        accuracy = correct/len(train_X)
        accuracy_values.append(accuracy)
        
        loss.backward()
        optimizer.step()
    
        if epoch % 10 == 0:
            logging.info('[hunmin log] Epoch {}/{}, Loss: {:.3f}, Accuracy: {:.3f}'.format(epoch,epochs, loss.data, accuracy))
    
    history = {"loss_values":loss_values,"accuracy_values":accuracy_values}
    
    ###########################################################################
    ## 학습 모델 저장
    ###########################################################################
    # 경로 지정
    model_path = os.path.join(tm.model_path, "iris_model.pth")
    logging.info('[hunmin log] model_path : {}'.format(model_path))
    # 저장하기
    torch.save(model.state_dict(), model_path)
    
    ###########################################################################
    ## 플랫폼 시각화
    ###########################################################################    
    display_result(tm, history, model, test_X, test_y)


def exec_init_svc(im):
    model = Net()
    model_path = im.model_path
    list_files_directories(model_path)
    model.load_state_dict(torch.load(os.path.join(model_path, 'iris_model.pth')))
    model.eval()
    logging.info('[hunmin log] model : {}'.format(model))
    
    return model


def exec_inference(df, params, batch_id):    
    ###########################################################################
    ## 4. 추론(Inference)
    ###########################################################################  
    logging.info('[hunmin log] inference start')
    logging.info('[hunmin log] df : {}'.format(df))
    logging.info('[hunmin log] params : {}'.format(params))
    model = params['model']
    logging.info('[hunmin log] model : {}'.format(model))
    result = {}
    logging.info('[hunmin log] result: {}'.format(result))
    df = np.array(df, dtype=np.float64)
    test_X = torch.from_numpy(df).float()
    predict_out = model(test_X)
    predict_y = np.argmax(predict_out.detach().numpy(), axis = 1)
    logging.info('[hunmin log] test_X: {}'.format(test_X))
    logging.info('[hunmin log] predict_out: {}'.format(predict_out))
    logging.info('[hunmin log] predict_y : {}'.format(predict_y))
    
    prediction = []
    for data in predict_y:
        if data == 0:
            prediction.append("Iris-setosa")
        elif data == 1:
            prediction.append("Iris-versicolor")
        elif data == 2:
            prediction.append("Iris-virginica")
    result['inference'] = prediction
    logging.info('[hunmin log] result: {}'.format(result))
    logging.info('[hunmin log] inference finish')
    return result     


# 저장 파일 확인
def list_files_directories(path):
    # Get the list of all files and directories in current working directory
    dir_list = os.listdir(path)
    logging.info('[hunmin log] Files and directories in {} :'.format(path))
    logging.info('[hunmin log] dir_list : {}'.format(dir_list))


###########################################################################
## exec_train(tm) 호출 함수 
###########################################################################
# 그래프 확인
def display_result(tm, history, model, x_test, y_test):
    for i in range(len(history['loss_values'])): #num_epochs
        metrix={}
        metrix['loss'] = history['loss_values'][i]
        metrix['accuracy'] = history['accuracy_values'][i]
        metrix['step'] = i
        tm.save_stat_metrics(metrix)

    predict_out = model(x_test)
    _, predict_y = torch.max(predict_out, 1)

    val_accuracy = 0
    val_loss = 0

    eval_results={}
    predict_y = predict_y.numpy().tolist()
    actual_y = y_test.numpy().tolist()
    eval_results['predict_y'] = predict_y
    eval_results['actual_y'] = actual_y
    eval_results['loss']= val_loss
    eval_results['accuracy'] = history['accuracy_values'][-1]

    # calculate_confusion_matrix(eval_results)
    eval_results['confusion_matrix'] = metrics.confusion_matrix(actual_y, predict_y).tolist()
    tm.save_result_metrics(eval_results)