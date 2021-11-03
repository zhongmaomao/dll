import torch
from torch import nn
import random
import numpy
import time
import data_loader
from collections import OrderedDict
from torch.nn import init

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class FlattenLayer(nn.Module):
    def __int__(self):
        super(FlattenLayer, self).__init__()

    def forward(self, x):
        # x shape: (batch, *, *, ...)
        return x.view(x.shape[0], -1)


class Net(nn.Module):
    def __init__(self, bar):
        super(Net, self).__init__()
        self.fc = nn.Sequential(
            FlattenLayer(),
            nn.Linear(100*bar, 1024),
            nn.Sigmoid(),
            nn.Linear(1024, 32),
            nn.Sigmoid(),
            nn.Linear(32, 2)
            # nn.Linear(100 * bar, 2)
        )

    def forward(self, cus):
        output = self.fc(cus)
        return output


def evaluate_accuracy(data_iter, net, device=None):
    if device is None and isinstance(net, torch.nn.Module):
        # 如果没指定device就使用net的device
        device = list(net.parameters())[0].device
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for X, y in data_iter:
            y_hat = net(X)
            # print([y_hat[i] for i in range(3)], [y[i] for i in range(3)])
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

        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'
           % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, test_acc))




data = torch.load('train_data1.pt')
label = torch.load('train_label1.pt')
# data, v, label = data_loader.reshape()
bar = data.size(2)
Num = data.size(0)
print('features num & length: ' + str(Num) + ' ' + str(bar))
print(label)

data = torch.split(data, int(Num/10 * 9), dim=0)
data_train = data[0]
data_validation = data[1]
label = torch.split(label, int(Num/10 * 9), dim=0)
label_train = label[0]
label_validation = label[1]


# idx = list(range(Num))
# random.shuffle(idx)  # 随机抽样
# for i in idx[0:int(Num/10)]:
#     data_validation = torch.cat([data_validation, data[i, :].unsqueeze(0)], dim=0)
#
# for i in idx[int(Num/10):Num]:
#     data_train = torch.cat([data_train, data[i, :].unsqueeze(0)], dim=0)


net = Net(bar)
print(net)
lr, num_epochs = 0.05, 100
batch_size = 256
data_iter = data_loader.data_iter
optimizer = torch.optim.SGD(net.parameters(), lr=lr)
train(net, data_iter, batch_size, optimizer, device, num_epochs, data_train, label_train,
            data_validation, label_validation)

