import torch
from torch import nn
from torch.nn import init
import torch.optim as optim
import numpy as np
import time
import string
import math
import data_process
import modules

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#######################################################################################################
# 数据
#######################################################################################################

with open("data/computer.csv") as fp:
    data = fp.read()
data = data.replace('\n', ' ')
data = data[0:50000]
print("data length:" + len(data).__str__())

idx_to_char = list(set(data))
char_to_idx = dict([(char, i) for i, char in enumerate(idx_to_char)])
vocab_size = len(char_to_idx)
print("vocab size:" + str(vocab_size))

data_idx = [char_to_idx[char] for char in data]


#####################################################################################################
# 训练
####################################################################################################


num_hide = 256
num_inputs, num_hiddens, num_outputs = vocab_size, 256, vocab_size
print('will use', device)
num_epochs, num_steps, batch_size, lr, clipping_theta = 300, 35, 64, 1e2, 1e-2
pred_period, pred_len, prefixes = 30, 50, ['很大', '很小']

modules.train_and_predict_rnn(modules.gru, modules.get_params,
                        modules.init_gru_state, num_inputs, num_hiddens, num_outputs,
                        vocab_size, device, data_idx, idx_to_char,
                        char_to_idx, True, num_epochs, num_steps, lr,
                        clipping_theta, batch_size, pred_period, pred_len,
                        prefixes)