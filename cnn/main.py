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

file_path = '.\cifar-10-batches-py'

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
    file = file_path + '\data_batch_' + str(i)
    data_batch = unpickle(file)
    data_train = torch.cat([data_train, torch.tensor(data_batch[b'data'], dtype=torch.float, device=device)], dim=0)
    label_train = torch.cat([label_train, torch.tensor(data_batch[b'labels'], dtype=torch.long, device=device)], dim=0)

data_train = data_train.view(-1, 3, 32, 32)

file = file_path + "\\test_batch"
data_batch = unpickle(file)
data_validation = torch.tensor(data_batch[b'data'], dtype=torch.float, device=device)
label_validation = torch.tensor(data_batch[b'labels'], dtype=torch.long, device=device)
data_validation = data_validation.view(-1, 3, 32, 32)

####################################################################################
#                       模型LeNet
####################################################################################


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 6, 5), # in_channels, out_channels, kernel_size
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2), # kernel_size, stride
            nn.Conv2d(6, 16, 5),
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(25*4*4, 120),
            nn.Sigmoid(),
            nn.Dropout(p=0.2),
            nn.Linear(120, 84),
            nn.Sigmoid(),
            nn.Dropout(p=0.2),
            nn.Linear(84, 10)
        )

    def forward(self, img):
        feature = self.conv(img)
        output = self.fc(feature.view(img.shape[0], -1))
        return output

def evaluate_accuracy(data_iter, net, device=None):
    if device is None and isinstance(net, torch.nn.Module):
        # 如果没指定device就使用net的device
        device = list(net.parameters())[0].device
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(net, torch.nn.Module):
                net.eval() # 评估模式, 这会关闭dropout
                acc_sum += (net(X.to(device)).argmax(dim=1) == y.to(device)).float().sum().cpu().item()
                net.train() # 改回训练模式
            else: # 自定义的模型, 3.13节之后不会用到, 不考虑GPU
                if('is_training' in net.__code__.co_varnames): # 如果有is_training这个参数
                    # 将is_training设置成False
                    acc_sum += (net(X, is_training=False).argmax(dim=1) == y).float().sum().item()
                else:
                    acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
            n += y.shape[0]
    return acc_sum / n


def train(net, data_iter, batch_size, optimizer, device, num_epochs, data_train, label_train,
            data_validation, label_validation):
    net = net.to(device)
    print("training on ", device)
    loss = torch.nn.CrossEntropyLoss()
    start = time.time()
    for epoch in range(num_epochs):
        train_iter = data_iter(batch_size, data_train, label_train)
        test_iter = data_iter(batch_size, data_validation, label_validation)
        #print("epoch " + str(epoch+1) + ":")
        train_l_sum, train_acc_sum, n, batch_count = 0.0, 0.0, 0, 0
        for X, y in train_iter:
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
        if (epoch%10 == 0):
            test_acc = evaluate_accuracy(test_iter, net)
            print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'
              % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))
            start = time.time()


timer = time.time()

net = LeNet()
print(net)
lr, num_epochs = 0.001, 1000
batch_size = 256
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
train(net, data_iter, batch_size, optimizer, device, num_epochs, data_train, label_train,
            data_validation, label_validation)

print('total time: %.1f sec' % (time.time() - timer))