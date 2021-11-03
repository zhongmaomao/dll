import csv
import gensim
import torch
from torch import nn
import random
import numpy
import time

Path = "RetailCustomerSales2.csv"

# Path = "amb.csv"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def data_iter(batch_size, features, labels):
    num_data = len(labels)
    idx = list(range(num_data))
    random.shuffle(idx)   #随机抽样
    for i in range(0, num_data, batch_size):
        j = torch.LongTensor(idx[i: min(i + batch_size, num_data)])  # 最后一次可能不足一个batch
        j = j.to(device)
        yield features.index_select(0, j), labels.index_select(0, j)


def one_hot(x, n_class, dtype=torch.float16):
    # X shape: (batch), output shape: (batch, n_class)
    x = x.long()
    res = torch.zeros(x.shape[0], n_class, dtype=dtype, device=x.device)
    res.scatter_(1, x.view(-1, 1), 1)
    return res


# 本函数已保存在d2lzh_pytorch包中方便以后使用
def to_onehot(X, n_class, model, idx_to_str, ID=False):
    # X shape: (batch, seq_len), output: seq_len elements of (batch, n_class)
    if ID:
        data = X[:, 0].unsqueeze(1)
        data = torch.cat([data, torch.tensor([model.wv.get_vector(j) for j in [idx_to_str[i] for i in X[:, 1]]], device=device)], dim=1)
    else:
        data = torch.tensor([model.wv.get_vector(j) for j in [idx_to_str[i] for i in X[:, 1]]], device=device)
    for i in range(2, 10):
        data = torch.cat([data, one_hot(X[:, i], n_class[i])], dim=1)
    data = torch.cat([data, X[:, 10].unsqueeze(1) / 1000], dim=1)
    return data


class FlattenLayer(nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()
    def forward(self, x):
        # x shape: (batch, *, *, ...)
        return x.view(x.shape[0], -1)


def embedding(path, size=100, window=64, min_count=1):
    sentence = gensim.models.word2vec.LineSentence(path)
    model = gensim.models.Word2Vec(sentence, vector_size=size, window=window, min_count=min_count, epochs=50)
    model.save('IDembedding.mdl')
    return model


def reshape():
    data_train, data_validation, label_train = data_reader(ID=True)
    label_train = label_train.cpu().numpy()
    #reshape
    print('数据量:' + str(data_train.size(0)) +' ' + str(data_validation.size(0)))
    tmp, start = data_train[0, 0], 0
    bar = data_train.size(1)
    train_data = torch.tensor([], dtype=torch.float16, device=device)
    train_data = train_data.view(-1, 100, bar-1)
    train_label = []
    for i in range(data_train.size(0)):
        if tmp != data_train[i, 0]:
            if i - start < 101:
                train_data = torch.cat([train_data, torch.cat([data_train[start:i, 1:bar], torch.zeros((100-i+start, bar-1), device=device)], dim=0).unsqueeze(0)], dim=0)
            else:
                train_data = torch.cat([train_data, data_train[start:start+100, 1:bar].unsqueeze(0)], dim=0)
            train_label.append(label_train[start])
            tmp = data_train[i, 0]
            start = i
    i = data_train.size(0)
    if i - start < 101:
        train_data = torch.cat([train_data, torch.cat([data_train[start:i, 1:bar], torch.zeros((100-i+start, bar-1), device=device)], dim=0).unsqueeze(0)], dim=0)
    else:
        train_data = torch.cat([train_data, data_train[start:start+100, 1:bar].unsqueeze(0)], dim=0)
    train_label.append(label_train[start])
    train_label = torch.tensor(train_label, device=device).long()

    tmp, start = data_validation[0, 0], 0
    validation_data = torch.tensor([], dtype=torch.float16, device=device)
    validation_data = validation_data.view(-1, 100, bar-1)
    for i in range(data_validation.size(0)):
        if tmp != data_validation[i, 0]:
            if i - start < 101:
                validation_data = torch.cat([validation_data, torch.cat([data_validation[start:i, 1:bar], torch.zeros((100-i+start, bar-1), device=device)], dim=0).unsqueeze(0)], dim=0)
            else:
                validation_data = torch.cat([validation_data, data_validation[start:start+100, 1:bar].unsqueeze(0)], dim=0)
            tmp = data_validation[i, 0]
            start = i
    i = data_validation.size(0)
    if i - start < 101:
        validation_data = torch.cat([validation_data, torch.cat([data_validation[start:i, 1:bar], torch.zeros((100 - i + start, bar-1), device=device)], dim=0).unsqueeze(0)],dim=0)
    else:
        validation_data = torch.cat([validation_data, data_validation[start:start + 100, 1:bar].unsqueeze(0)], dim=0)
    return train_data, validation_data, train_label


def data_reader(ID=False):
    dater = []
    i = 0
    # readin dater
    with open(Path) as fp:
        reader = csv.reader(fp)
        next(reader)
        for row in reader:
            dater.append(row)
            i = i + 1

    # list dater to np data
    lens = len(dater)
    data = numpy.zeros((lens, 12), int)
    vocab_size = numpy.zeros(11, int)
    for col in range(11):
        cols = [row[col] for row in dater]
        idx_to_str = list(set(cols))
        str_to_idx = dict([(str, i) for i, str in enumerate(idx_to_str)])
        vocab_size[col] = len(str_to_idx)
        data[:, col] = [str_to_idx[str] for str in cols]

    # embedding
    cols = [row[1] for row in dater]
    idx_to_str = list(set(cols))
    with open('itemID.csv', 'w') as fp:
        writer = csv.writer(fp, delimiter=' ')
        tmp, start = data[0, 0], 0
        for i in range(lens):
            if tmp != data[i, 0]:
                writer.writerow(cols[start:i])
                tmp = data[i, 0]
                start = i
        writer.writerow(cols[start:i])
    # one-hot & embedding
    vocab_size = numpy.delete(vocab_size, [7])
    print('各列类别数' + str(vocab_size))
    data[:, 11] = [int(str) for str in [row[11] for row in dater]]
    data_train = []
    label_train = []
    data_validation = []
    cols = [row[7] for row in dater]
    for i in range(lens):
        if cols[i] == '':
            data_validation.append(list(data[i, 0:8]) + list(data[i, 9:12]))
        else:
            data_train.append(list(data[i, 0:8]) + list(data[i, 9:12]))  # max 1024 for 1 cus
            label_train.append(data[i, 7])
    data_validation = torch.tensor(data_validation, dtype=torch.int, device=device)
    data_train = torch.tensor(data_train, dtype=torch.int, device=device)
    print('训练embedding,以及onehot变换:')
    startime = time.time()
    model = embedding('itemID.csv', size=32)
    data_train = to_onehot(data_train, vocab_size, model, idx_to_str, ID=ID)
    data_validation = to_onehot(data_validation, vocab_size, model, idx_to_str, ID=ID)
    print('  train data:' + str(data_train.size(0)))
    print('  validation data:' + str(data_validation.size(0)))
    label_train = torch.tensor(label_train, device=device).long() - 1

    return data_train, data_validation, label_train





# train_data, validation_data, train_label = data_reader()
# torch.save(train_data, 'train_data.pt')
# torch.save(train_label, 'train_label.pt')
# torch.save(validation_data, 'validation_data.pt')
#
# train_data, validation_data, train_label = reshape()
# torch.save(train_data, 'train_data1.pt')
# torch.save(train_label, 'train_label1.pt')
# torch.save(validation_data, 'validation_data1.pt')
