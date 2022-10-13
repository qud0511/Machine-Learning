# 파일명: iris_classification_preprocess_pytorch_sub.py

import os
import numpy as np
import pandas as pd
import logging


def exec_process(pm):    
    ###########################################################################
    # 1_1_local_platform_iris_classfication_preprocess_train.ipynb 파일의 
    # 1.데이터 세트 준비(Data Setup) 부분을 여기에 작성한다.
    ###########################################################################
    
    logging.info('[hunmin log]  the start line of the function [exec_process]')

    # 데이터 파일이 저장된 경로.
    list_files_directories(pm.source_path)

    # 데이터 파일을 가져옴.
    #df = pd.read_csv(os.path.join(pm.source_path, os.listdir(pm.source_path)[-1]))
    #df = pd.read_csv(os.path.join(pm.source_path, "/*.csv")) -> X
    df = pd.read_csv(os.path.join(pm.source_path, "iris.csv"))

    # 가져온 데이터 파일을 저장.
    df.to_csv(pm.target_path+"/iris.csv", index = False)

    # 데이터 파일을 저장할 경로.
    list_files_directories(pm.target_path)

# 저장 파일 확인
def list_files_directories(path):
    # Get the list of all files and directories in current working directory
    dir_list = os.listdir(path)
    logging.info('[hunmin log] Files and directories in {} :'.format(path))
    logging.info('[hunmin log] dir_list : {}'.format(dir_list))
    logging.info('[hunmin log] finish exec_process')