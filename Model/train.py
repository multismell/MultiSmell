import numpy as np
import torch.nn.functional as F
from sklearn import metrics
from torch import optim
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, hamming_loss,jaccard_score,roc_auc_score,multilabel_confusion_matrix
from transformers.optimization import *
import sys
from tqdm import tqdm

class Logger(object):
    def __init__(self, filename, stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'w')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        return True

def train_val_test(config, model, train_iter,vaild_iter):
    model.train()
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm', 'LayerNorm.weight']
    optimizer_growped_paramters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = optim.Adam(params=optimizer_growped_paramters,
                           lr=config.learning_rate)
    flag = False  # 记录是否很久没有提升效果

    model.train()
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    # 训练/验证
    train_acc_list = np.array([], dtype=int)
    train_loss_list = np.array([], dtype=int)
    val_acc_list = np.array([], dtype=int)
    val_loss_list = np.array([], dtype=int)
    for epoch in range(config.num_epochs):
        print('Epoch{}/{}'.format(epoch + 1, config.num_epochs))
        epoch_train_acc_list_1 = np.array([],dtype=int)
        epoch_train_loss_list = np.array([],dtype=int)

        for i, (x1, mask, x2, labels) in enumerate(tqdm(train_iter)):
            x1 = x1.to(config.device)
            mask = mask.to(config.device)
            x2 = x2.to(config.device)
            labels = labels.to(config.device)
            outputs = model(x1, mask, x2, labels)
            model.zero_grad()
            loss = F.binary_cross_entropy(outputs, labels)
            loss.backward()
            loss = loss.detach().cpu().numpy()
            optimizer.step()
            predict = (outputs >= 0.5).long()  # 将输出阈值设置为0.5 [0.49,0.58]
            labels = labels.data.cpu().numpy()
            predict = predict.cpu().numpy()

            if i != 0:
                j = len(predict_all)
                labels_all = np.insert(labels_all,j,labels,axis=0)
                predict_all = np.insert(predict_all,j,predict,axis=0)
            else:
                j = len(predict)
                labels_all = labels
                predict_all = predict
            acc = metrics.jaccard_score(labels_all[:, ], predict_all[:, ],average='macro')
            epoch_train_acc_list_1 = np.append(epoch_train_acc_list_1, acc)
            epoch_train_loss_list = np.append(epoch_train_loss_list, loss.item())

        print("accuracy macro:{} \t loss:{} \t ".format(np.mean(epoch_train_acc_list_1),np.mean(epoch_train_loss_list)))

        val_acc, val_loss = evaluate(config, model, vaild_iter, test=False, type='method') # 验证

        train_acc_list = np.append(train_acc_list,np.mean(epoch_train_acc_list_1))
        train_loss_list = np.append(train_loss_list,np.mean(epoch_train_loss_list))
        val_acc_list = np.append(val_acc_list,val_acc)
        val_loss_list = np.append(val_loss_list,val_loss)


        if flag:
            break
    # torch.save(model.state_dict(), config.save_path)
    return train_acc_list,train_loss_list,val_acc_list,val_loss_list

def train_test(t, config, model, train_iter):
    model.train()
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm', 'LayerNorm.weight']
    optimizer_growped_paramters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = optim.Adam(params=optimizer_growped_paramters,
                           lr=config.learning_rate)
    flag = False  # 记录是否很久没有提升效果

    model.train()
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    # 训练/验证
    train_acc_list = np.array([], dtype=int)
    train_loss_list = np.array([], dtype=int)
    val_acc_list = np.array([], dtype=int)
    val_loss_list = np.array([], dtype=int)
    for epoch in range(config.num_epochs):
        print('Epoch{}/{}'.format(epoch + 1, config.num_epochs))
        epoch_train_acc_list_1 = np.array([],dtype=int)
        epoch_train_loss_list = np.array([],dtype=int)

        for i, (x1, mask, x2, labels) in enumerate(tqdm(train_iter)):
            x1 = x1.to(config.device)
            mask = mask.to(config.device)
            x2 = x2.to(config.device)
            labels = labels.to(config.device)
            outputs = model(x1, mask, x2, labels)
            model.zero_grad()
            loss = F.binary_cross_entropy(outputs, labels)
            loss.backward()
            loss = loss.detach().cpu().numpy()
            optimizer.step()
            predict = (outputs >= 0.5).long()  # 将输出阈值设置为0.5 [0.49,0.58]
            labels = labels.data.cpu().numpy()
            predict = predict.cpu().numpy()

            if i != 0:
                j = len(predict_all)
                labels_all = np.insert(labels_all,j,labels,axis=0)
                predict_all = np.insert(predict_all,j,predict,axis=0)
            else:
                j = len(predict)
                labels_all = labels
                predict_all = predict
            acc = metrics.jaccard_score(labels_all[:, ], predict_all[:, ],average='macro')
            epoch_train_acc_list_1 = np.append(epoch_train_acc_list_1, acc)
            epoch_train_loss_list = np.append(epoch_train_loss_list, loss.item())

        print("accuracy macro:{} \t loss:{} \t ".format(np.mean(epoch_train_acc_list_1),np.mean(epoch_train_loss_list)))

        train_acc_list = np.append(train_acc_list,np.mean(epoch_train_acc_list_1))
        train_loss_list = np.append(train_loss_list,np.mean(epoch_train_loss_list))


        if flag:
            break
    # torch.save(model.state_dict(), 'E:/save_dict/'+ config.model_name + '_' + str(t+1) + '.ckpt')
    return train_acc_list,train_loss_list

def evaluate(config, model, vaild_iter,test=False,type='method'):
    if test:
        model.eval()
    dev_acc_all = np.array([], dtype=int)
    dev_loss_all = np.array([],dtype=int)
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    dev_acc = np.array([],dtype=int)
    dev_loss = np.array([],dtype=int)
    with torch.no_grad():
        for i, (x1, mask, x2, labels) in enumerate(tqdm(vaild_iter)):
            x1 = x1.to(config.device)
            mask = mask.to(config.device)
            x2 = x2.to(config.device)
            labels = labels.to(config.device)
            outputs = model(x1, mask, x2, labels)
            predict = (outputs >= 0.5).long()
            epoch_dev_loss = F.binary_cross_entropy(outputs, labels)
            labels = labels.data.cpu().numpy()
            predict = predict.cpu().numpy()

            if i != 0:
                j = len(predict_all)
                labels_all = np.insert(labels_all,j,labels,axis=0)
                predict_all = np.insert(predict_all,j,predict,axis=0)
            else:
                labels_all = labels
                predict_all = predict
            acc = metrics.jaccard_score(labels_all[:, ], predict_all[:, ], average='macro')
            epoch_dev_loss = epoch_dev_loss.detach().cpu().numpy()
            dev_acc_all = np.append(dev_acc_all, acc)
            dev_loss_all = np.append(dev_loss_all,epoch_dev_loss)

    if type == 'method':
        label_name = ['FE','LPL','LM']
    else:
        label_name = ['GC','CC','MC']
    if test:
        hammingloss,accuracy_2,subsetAccuracy,macroPrecision,macroRecall,macroF1,microPrecision,microRecall,microF1 = metric(labels_all,predict_all)
        # hammingloss,accuracy_1,accuracy_2,subsetAccuracy,macroPrecision,macroRecall,macroF1,microPrecision,microRecall,microF1,macroAuc,microAuc = metric(labels_all,predict_all)
        report = metrics.classification_report(labels_all, predict_all, target_names=label_name, digits=4)
        # report = metrics.classification_report(labels_all, predict_all, digits=4)
        multilabel_confusion = metrics.multilabel_confusion_matrix(labels_all,predict_all)
        return hammingloss,accuracy_2,subsetAccuracy,macroPrecision,macroRecall,macroF1,microPrecision,microRecall,microF1,report,multilabel_confusion
        # return hammingloss,accuracy_1,accuracy_2,subsetAccuracy,macroPrecision,macroRecall,macroF1,microPrecision,microRecall,microF1,macroAuc,microAuc,report,multilabel_confusion

    dev_acc = np.append(dev_acc,np.mean(dev_acc_all))
    dev_loss = np.append(dev_loss,np.mean(dev_loss_all))

    return dev_acc, dev_loss


def test(config, model, test_iter,type):
    # model.load_state_dict(torch.load(config.save_path))
    # model.eval()
    hammingloss,accuracy_2,subsetAccuracy,macroPrecision,macroRecall,macroF1,microPrecision,microRecall,microF1,report,multilabel_confusion = evaluate(config, model, test_iter, True,type)
    # hammingloss,accuracy_1,accuracy_2,subsetAccuracy,macroPrecision,macroRecall,macroF1,microPrecision,microRecall,microF1,macroAuc,microAuc,report,multilabel_confusion = evaluate(config, model, test_iter, True,type)
    print("#" * 20, '测试结果', "#" * 20)
    print("test_hamming_loss:", hammingloss)
    # print("test_accuracy samples:", accuracy_1)
    print("test_ExactmatchRatio:", subsetAccuracy)  # subsetAccuracy == ExactmatchRatio
    print("test_JaccardAccuracy:", accuracy_2) # accuracy_2 == Jaccard
    print("test_MacroPrecision:", macroPrecision)
    print("test_MacroRecall:", macroRecall)
    print("test_MacroF1:", macroF1)
    print("test_MicroPrecision:", microPrecision)
    print("test_MicroRecall:", microRecall)
    print("test_MicroF1:", microF1)
    # print("macroAUC:",macroAuc)
    # print("microAUC:",microAuc)
    print("test_report:")
    print(report)
    print("multilabel Confusion Maxtrix:")
    print(multilabel_confusion)


def metric(y_true,y_pre,sample_weight=None):
    def Hloss():
        hammingloss = hamming_loss(y_true,y_pre)
        return hammingloss
    def Accuracy_1():
        accuracy_1 = jaccard_score(y_true,y_pre,average='samples')
        return accuracy_1
    def jaccardAccuracy():
        accuracy_2 = jaccard_score(y_true, y_pre, average='macro')
        return accuracy_2
    def SubAccuracy():
        subsetAccuracy = accuracy_score(y_true,y_pre)
        return subsetAccuracy
    def MacPrecision():
        macroPrecision = precision_score(y_true,y_pre,average='macro')
        return macroPrecision
    def MacRecall():
        macroRecall = recall_score(y_true,y_pre,average='macro')
        return macroRecall
    def MacF1():
        macroF1 = f1_score(y_true,y_pre,average='macro')
        return macroF1
    def MicPrecision():
        microPrecision = precision_score(y_true,y_pre,average='micro')
        return microPrecision
    def MicRecall():
        microRecall = recall_score(y_true,y_pre,average='micro')
        return microRecall
    def MicF1():
        microF1 = f1_score(y_true,y_pre,average='micro')
        return microF1
    def macroAUC():
        macroAuc = roc_auc_score(y_true,y_pre,average='macro')
        return macroAuc
    def microAUC():
        microAuc = roc_auc_score(y_true,y_pre,average='micro')
        return microAuc

    return Hloss(),jaccardAccuracy(),SubAccuracy(),MacPrecision(),MacRecall(),MacF1(),MicPrecision(),MicRecall(),MicF1()
    # return Hloss(),Accuracy_1(),Accuracy_2(),SubAccuracy(),MacPrecision(),MacRecall(),MacF1(),MicPrecision(),MicRecall(),MicF1(),macroAUC(),microAUC()