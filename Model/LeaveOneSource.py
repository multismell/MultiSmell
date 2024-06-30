import torch
import numpy as np
import pandas as pd
import os
from transformers import XLNetTokenizer
from imblearn.over_sampling import SMOTE,ADASYN
from imblearn.under_sampling import NearMiss,EditedNearestNeighbours
import utils
from sklearn.utils import shuffle as reset
from torch.utils.data import DataLoader
import train
import K_foldTest
from collections import Counter
import models
from stat import ST_MTIME
import random


def LOS(config,path,model,train_size):
    """
        method
    """
    data_list = loop_dir(path)
    print(data_list)
    for i in range(0,len(data_list),2):
    # for i in range(10,len(data_list)):
        print('#'*20,'第',i+1,'项目','#'*20)
        print('数据集加载')
        print(data_list[i],data_list[i+1])
        trainData, testData = pd.read_excel(data_list[i+1]), pd.read_excel(data_list[i])
        trainData = train_test_split(trainData,train_size=0.1)
        model_method = models.MultiSmell(config).to(config.device)
        print('数据处理结束')
        print('训练数据总数：{} \t 测试数据总数：{}'.format(len(trainData),len(testData)))

        traindata, testdata = reset(trainData, random_state=42), reset(testData, random_state=42)
        traindataset,testdataset = utils.MyDataset(traindata,config),utils.MyDataset(testdata,config)
        trainloader, testloader = DataLoader(traindataset, batch_size=config.batch_size, shuffle=False, num_workers=0,
                                             drop_last=True), DataLoader(testdataset, batch_size=config.batch_size,
                                                                         shuffle=False, num_workers=0, drop_last=True)
        method_train_acc_list, method_train_loss_list = train.train_test(i,config,model_method,trainloader)
        train.test(config, model_method, testloader, type=config.types)

    '''
        class
    '''
    # data_list = os.listdir(path)
    #print(data_list)
    #model_class = models.MultiSmell(config).to(config.device)
    #print('数据集加载')
    #trainData, testData = pd.read_excel(os.path.join(path, data_list[1])), pd.read_excel(os.path.join(path, data_list[0]))
    #trainData = train_test_split(trainData, train_size=train_size)
    #print('数据处理结束')
    #print('训练数据总数：{} \t 测试数据总数：{}'.format(len(trainData),len(testData)))

    #traindata, testdata = reset(trainData, random_state=42), reset(testData, random_state=42)
    #traindataset,testdataset = utils.MyDataset(traindata, config),utils.MyDataset(testdata, config)
    #trainloader, testloader = DataLoader(traindataset, batch_size=config.batch_size, shuffle=False, num_workers=0,
     #                                    drop_last=True), DataLoader(testdataset, batch_size=config.batch_size,
     #                                                                shuffle=False, num_workers=0, drop_last=True)
    #method_train_acc_list, method_train_loss_list = train.train_test(0,config,model_class,trainloader)
    #train.test(config, model_class, testloader, type=config.types)


def get_mtime(filename):
    file_path = os.path.join('E:\MultiSmell\data\padding_data_2',filename)
    return os.stat(file_path).st_mtime

def loop_dir(path,start=True):
    result = []
    if start:
        # data_list = sorted(os.listdir(path), key=get_mtime)
        data_list = os.listdir(path)
    else:
        data_list = os.listdir(path)

    for i, file_name in enumerate(data_list):
        sub_path = os.path.join(path, file_name)
        if os.path.isdir(sub_path):
            result.extend(loop_dir(sub_path,start=False))
        else:
            result.append(sub_path)
    return result

def train_test_split(data, train_size=0.7):
    data_size = len(data)
    seed = 1024
    random.seed(seed)
    np.random.seed(seed)
    k = int(data_size * train_size)
    sj = random.sample(range(0,data_size),k)

    resultData = pd.DataFrame([data.iloc[0, :]])
    for item in sj:
        temp_data = pd.DataFrame([data.iloc[item, :]])
        resultData = pd.concat([resultData, temp_data])
    resultData = resultData.iloc[1:, :]
    resultData = resultData.reset_index(drop=True)

    return resultData


