import torch
import torch.autograd as autograd
from model import *
from utils_gpu import *
import torch.nn as nn
import torch.optim as optim
import logging
import logging.handlers
import time
import torch.backends.cudnn
from myDataSet import *
from torch.utils.data import *
from test import *
LOG_FILE = 'train_text.log'

handler = logging.handlers.RotatingFileHandler(LOG_FILE, maxBytes=1024 * 1024, backupCount=5)  # 实例化handler
fmt = '%(asctime)s - %(filename)s:%(lineno)s - %(name)s - %(message)s'

formatter = logging.Formatter(fmt)  # 实例化formatter
handler.setFormatter(formatter)  # 为handler添加formatter

logger = logging.getLogger('tst')  # 获取名为tst的logger
logger.addHandler(handler)  # 为logger添加handler
logger.setLevel(logging.DEBUG)

torch.cuda.device(0)
batch_size = 10
epochNum = 50
num_layers = 2
embedding_num = 256
hidden_num = 512

word_to_idx, idx_to_word = load_dict('data/vocab.txt')
train_data_set = myTrainDataSet('data/train.txt',word_to_idx)
train_size = len(train_data_set)
train_data_loader = DataLoader(train_data_set,1,shuffle=True)
test_data_set = myTestDataSet('data/test.txt',word_to_idx)
test_data_loader = DataLoader(test_data_set,1,shuffle=False)
model = CNN_RNN(len(word_to_idx),1,1,embedding_dim=embedding_num,hidden_dim=hidden_num)
print(model)
print(len(train_data_set),len(test_data_set))
optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

t1 = time.clock()
for epoch in range(epochNum):
    epoch_loss = 0
    words_count = 0
    acc_count = 0
    case = 0
    loss = 0

    evaluation(model,test_data_loader,word_to_idx,idx_to_word)
    for train_data in train_data_loader:
        model.zero_grad()
        #print(case)
        text = autograd.Variable(train_data[0])
        label = autograd.Variable(train_data[1])
        summary = autograd.Variable(train_data[2])
        #print(text,label.size())
        output, states = model(text, summary)
        words_count += label.size()[1]
        output_seq = []
        for i in range(label.size()[1]):
            topv, topi = output[i].data.topk(1)

            if not label.data[0][i]==0 and topi[0] == label.data[0][i]:
                acc_count += 1
            output_seq.append(topi[0])
        print(output_seq)
        loss += criterion(output, label.view(-1))
        case += 1
        if (case % batch_size ==0):
            loss = loss / batch_size

            if case % (batch_size * 5) == 0:
                print(epoch, case / batch_size, time.clock() - t1,loss.data[0])
                t1 = time.clock()
            loss.backward()
            torch.nn.utils.clip_grad_norm(model.parameters(), 0.5)
            optimizer.step()
            epoch_loss += loss.data[0]
            loss = 0
    logger.info('train')
    logger.info([epoch, epoch_loss, acc_count / words_count])
    print(epoch, epoch_loss, acc_count / words_count)