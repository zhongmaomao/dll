import torch
from torch import nn
import random
import time
import pickle
import numpy as py

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

##############################################################
#                 数据加载和处理 cifar-10
#       数据分别存储在data_train，label_train，data_validation，label_validation
#             (device, (n,3072)int16, (n)int8)
##############################################################

file_path = './cifar-10-batches-py'

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def data_iter(batch_size, features, labels):
    num_data = len(labels)
    idx = list(range(num_data))
    random.shuffle(idx)   #随机抽样
    for i in range(0, num_data, batch_size):
        j = torch.LongTensor(idx[i: min(i + batch_size, num_data)])  # 最后一次可能不足一个batch
        j = j.to(device)
        yield features.index_select(0, j), labels.index_select(0, j)

data_train = torch.tensor([], dtype=torch.float, device=device)
label_train = torch.tensor([], dtype=torch.int8, device=device)

for i in range(1,6):
    file = file_path + '/data_batch_' + str(i)
    data_batch = unpickle(file)
    data_train = torch.cat([data_train, torch.tensor(data_batch[b'data'], dtype=torch.float, device=device)], dim=0)
    label_train = torch.cat([label_train, torch.tensor(data_batch[b'labels'], dtype=torch.long, device=device)], dim=0)

data_train = data_train.view(-1, 3, 32, 32)

file = file_path + "/test_batch"
data_batch = unpickle(file)
data_validation = torch.tensor(data_batch[b'data'], dtype=torch.float, device=device)
label_validation = torch.tensor(data_batch[b'labels'], dtype=torch.long, device=device)
data_validation = data_validation.view(-1, 3, 32, 32)