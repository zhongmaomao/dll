import torch
from torch import nn
from torch.nn import init
import torch.optim as optim
import numpy as np
import random
import time

time_start = time.time()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = 'cpu'

def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)  # 随机
    for i in range(0, num_examples, batch_size):
        j = torch.LongTensor(indices[i: min(i + batch_size, num_examples)])  # 最后一次可能不足一个batch
        j = j.to(device)
        yield features.index_select(0, j), labels.index_select(0, j)


class FlattenLayer(nn.Module):
    def __int__(self):
        super(FlattenLayer, self).__init__()

    def forward(self, x):
        # x shape: (batch, *, *, ...)
        return x.view(x.shape[0], -1)


# test
num_inputs = 10000
num_examples = 1000
true_w = torch.tensor(np.random.normal(0, 1, [1, num_inputs]), device=device)
features = torch.tensor(np.random.normal(0, 1, [1000, num_inputs]), device=device)
labels = torch.zeros(num_examples, device=device)


for i in range(num_examples):
   labels[i] = torch.sum(true_w * features[i, :])
   if labels[i] < -1:
       labels[i] = 0
   elif labels[i] > 1:
       labels[i] = 2
   else:
       labels[i] = 1

num_hide = 256
net = nn.Sequential(
    nn.Linear(num_inputs, num_hide, device=device),
    nn.ReLU(),
    nn.Linear(num_hide, 3, device=device),
)

#for para in net.parameters():
#    print(para)

init.normal_(net[0].weight, mean=0, std=0.01)
init.constant_(net[0].bias, val=0)
init.normal_(net[2].weight, mean=0, std=0.01)
init.constant_(net[2].bias, val=0)
#print(net[0].weight)
loss = nn.CrossEntropyLoss()


optimizer = optim.SGD(net.parameters(), lr=0.03)

num_epochs = 200
count = 0
for epoch in range(num_epochs):
    for X, y in data_iter(100, features, labels):
        count += 1
        #print(X.size(), y.size())
        X = X.float()
        y = y.long()
        output = net(X)
        # print(output,y)
        l = loss(output, y)
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
    print('epoch %d, loss: %f' % (epoch, l.item()))
print(output[0:5, :], y[0:5], count)


time_end = time.time()
print('totally cost', time_end-time_start)
#for X, y in data_iter(1, features, labels):
#    X = X.float()
#    output = net(X)
#    print(output)
#    print(y)
