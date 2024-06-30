import torch
import numpy as np
import utils
import time
import train
import models
import warnings
import sys
import K_foldTest,LeaveOneSource


if __name__ == '__main__':

    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    torch.backends.cudnn.deterministic = True

    start_train = time.time()

    # 方法级
    model_name_method = "FE_LPL_LM"
    path_method = 'G:/MultiSmell/data/FE_LPL_LM.json'
    types_method = 'method'

    print('path_method:', path_method)
    config_method = models.config(model_name_method, types_method)
    print(config_method.device)

    print('###############################################################')
    method_model = models.MultiSmell(config_method).to(config_method.device)
    print('method_model:', method_model)

    path = 'E:\MultiSmell\data\padding_data_2'
    LeaveOneSource.LOS(config_method, path, method_model)


    # 类级
    # model_name_class = "GC_CC_MC"
    # path_class = 'E:\MultiSmell\data\GC_CC_MC.xlsx'
    # types_class = 'class'
    # print('path_class:',path_class)
    # config_class = models.config(model_name_class, types_class)
    # print('###############################################################')
    # class_model = models.MultiSmell(config_class).to(config_class.device)
    # print('class_model:',class_model)

    # path = 'E:\MultiSmell\data\padding_class_data'
    # LeaveOneSource.LOS(config_class,path_class,class_model,train_size=1)
