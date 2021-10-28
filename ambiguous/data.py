import torch
import numpy as py
import random

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def data_iter(batch_size, corpus, labels, device=None):
    if device is None:
        device = corpus.device

    num_corpus = len(labels)
    idx = list(range(num_corpus))
    random.shuffle(idx)

    for i in range(0, num_corpus, batch_size):
        j = torch.LongTensor(idx[i:min(i+batch_size-1, num_corpus)], device=device)
        yield corpus.indexselect(0, j), labels.indexselect(0, j)


def read_txt(file_path, is_test=False):
    #read txt to py struct
    corpus = []
    with open(file_path, encoding='utf-8') as fp:
        data = fp.readlines()
        for row in data:
            r = row.split('\t')
            if not is_test:
                corpus.append([r[0], r[1], int(r[2])])
            else:
                corpus.append([r[0], r[1]])
    return corpus


file_path = 'train.txt'
print(read_txt(file_path))

