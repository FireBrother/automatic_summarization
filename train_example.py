import torch
import torch.autograd as autograd
from model import *
from utils_gpu import *
import torch.nn as nn
import torch.optim as optim
import logging
import logging.handlers
import time

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


def sample(s, t, o, text):
    states = model.initHidden_gpu(num_layers)
    length = t.size()[1]
    word_count = length
    acc_count = 0
    loss = 0.0
    output_t = []
    output, states = model(t, text, states)
    for i in range(length):
        topv, topi = output[i].data.topk(1)
        if topi[0] == s[i + 1]:
            acc_count += 1

    loss += criterion(output, o.view(-1))
    return word_count, acc_count, loss


def eval(data_name):
    total_loss = 0
    counts = 0
    total_word_count = 0
    total_acc_count = 0
    for case in range(test_size):
        if data_name == 'test':
            word_count, acc_count, loss = sample(test_data[case], test_X[case], test_Y[case], test_T[case])
        else:
            word_count, acc_count, loss = sample(cv_data[case], cv_X[case], cv_Y[case], cv_T[case])

        total_loss += loss.data[0]
        total_word_count += word_count
        total_acc_count += acc_count
        counts += 1
    total_loss = total_loss / counts
    print(data_name)
    print("loss", total_loss)
    print("acc", total_acc_count / total_word_count)
    logger.info(data_name)
    logger.info([total_loss, total_acc_count / total_word_count])


word_to_id, id_to_word = load_dict('dict.txt')
train_data, train_type = read_data('train_data.txt', word_to_id)
test_data, test_type = read_data('test_data.txt', word_to_id)
cv_data, cv_type = read_data('cv_data.txt', word_to_id)
description_word_to_id, _ = load_dict('description_dict.txt')
type_to_text = read_description('description_words.txt', description_word_to_id)

train_X, train_Y, train_T = make_text_cases(train_data, train_type, type_to_text)
cv_X, cv_Y, cv_T = make_text_cases(cv_data, cv_type, type_to_text)
test_X, test_Y, test_T = make_text_cases(test_data, test_type, type_to_text)
# print(type_to_text)

# print(train_T)
train_size = len(train_type)
test_size = len(test_type)

model = RNNLM_text(len(word_to_id), len(description_word_to_id), embedding_num, hidden_num, num_layers)
print(model)
# model.cuda()
optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

t1 = time.clock()
for epoch in range(epochNum):
    epoch_loss = 0
    words_count = 0
    acc_count = 0

    #    states = model.initHidden_gpu(num_layers)
    #    states = detach(states)
    for batchIndex in range(int(train_size / batch_size)):
        if batchIndex % 50 == 0:
            print(epoch, batchIndex, time.clock() - t1)
            t1 = time.clock()
        model.zero_grad()
        loss = 0
        counts = 0
        #        t1 = time.clock()
        for case in range(batchIndex * batch_size, min((batchIndex + 1) * batch_size, train_size)):
            print(case)
            s = train_data[case]
            t, o, text = train_X[case], train_Y[case], train_T[case]
            states = model.initHidden(num_layers)
            # print(t,o,text)
            output, states = model(t, text, states)
            words_count += t.size()[1]
            for i in range(t.size()[1]):
                topv, topi = output[i].data.topk(1)
                if topi[0] == s[i + 1]:
                    acc_count += 1

            loss += criterion(output, o.view(-1))
            counts += 1
        loss = loss / counts
        loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), 0.5)
        optimizer.step()
        epoch_loss += loss.data[0]
    logger.info('train')
    logger.info([epoch, epoch_loss, acc_count / words_count])
    print(epoch, epoch_loss, acc_count / words_count)
    if epoch % 5 == 0:
        torch.save(model, 'model-' + str(epoch))
    eval('test')
    eval('cv')
