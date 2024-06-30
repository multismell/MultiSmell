from datetime import timedelta
import torch
from sklearn.model_selection import train_test_split
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import time
from sklearn.utils import shuffle as reset
import json
from transformers import XLNetTokenizer
from imblearn.over_sampling import SMOTE,ADASYN
from collections import Counter

class MyDataset(Dataset):
    def __init__(self, data, config):
        self.data = data
        self.config = config
        self.padding_size = 20

    def clean_data(self):
        for i in range(len(self.data)):
            x = self.data.iloc[i, :].values
            x2 = x[-2:34]
            try:
                x2 = [float(x) for x in x2]
                x2 = torch.tensor(x2)
            except:
                print(i + 1)
                print(x2)

    def __getitem__(self, index):
        line = self.data.iloc[index].values.tolist()
        if self.config.types == 'method':
            x1 = line[:-23]
            x2 = line[-23:-3]
            label = line[-3:43]
        else:
            x1 = line[:-13]
            x2 = line[-13:-3]
            label = line[-3:33]
        try:
            mask = self.build_data(x1)
        except:
            print(x1)
        x1 = torch.tensor(x1).long()
        mask = torch.tensor(mask).long()

        x2 = [float(x) for x in x2]
        x2 = torch.tensor(x2)
        try:
            label = [float(x) for x in label]
        except:
            print(index)
        label = torch.tensor(label)

        return x1, mask, x2, label

    def __len__(self):
        return len(self.data)

    def collate(self, data):
        x1 = torch.stack([x[0] for x in data], dim=0)
        mask = torch.stack([x[1] for x in data], dim=0)
        x2 = torch.stack([x[2] for x in data], dim=0)
        all_label = []
        i = 0
        for l in data:
            l = l[3]
            if len(l) == 2:
                all_label.append(l)
            else:
                print(i)
                print(l)
            i += 1
        label = torch.stack(all_label, dim=0)
        return x1, mask, x2, label

    def build_data(self, data):
        data_str = [x for x in data]
        tonkens = 0
        for content in data_str:
            if content != 0:
                tonkens += 1
        pad_size = self.padding_size
        mask = [1] * tonkens
        mask_len = len(mask)
        if pad_size:
            if mask_len <= pad_size:
                mask = mask + [0] * (pad_size - mask_len)
            if mask_len > pad_size:
                mask = mask[:pad_size]
        return mask

def get_time_dif(start_time):
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def train_test_split( data, train_size=0.7, shuffle=True, random_state=None):
    if shuffle:
        data = reset(data, random_state=42)

    train = data[:int(len(data) * train_size)].reset_index(drop=True)
    test = data[int(len(data) * train_size):].reset_index(drop=True)
    return train, test
