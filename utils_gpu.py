import torch
import torch.autograd as autograd
import codecs

def read_data(filename, word_to_id):
    f = open(filename,'r')
    data = []
    type = []
    for line in f.readlines():
        line  = line.split(' ')
        tmp_data = []
        tmp_type = []
        for word in line[1:]:
            if word in word_to_id:
                tmp_data.append(word_to_id[word])
            else:
                tmp_data.append(0)
            tmp_type.append(int(line[0]))

        #print(torch.LongTensor(tmp_data).size())
        data.append(torch.LongTensor(tmp_data))
        type.append(torch.LongTensor(tmp_type[:-1]))
        if len(data)==10:
            break
    f.close()
    return data,type

def read_description(filename, word_to_id):
    f = codecs.open(filename, 'r',encoding='utf-8')
    type_text = {}
    count = 0
    for line in f.readlines():
        count += 1
        line = line.split(' ')

        tmp_data = torch.LongTensor(1,len(line))
        for i,word in enumerate(line):
            #print(word)
            if word in word_to_id:
                tmp_data[0][i]= word_to_id[word]
            else:
                tmp_data[0][i] = 0

        type_text[count]=autograd.Variable(torch.LongTensor(tmp_data))
    f.close()
    return type_text

def load_dict(filename):
    word_to_idx = dict()
    idx_to_word = dict()
    with open(filename,'r') as f:
        count = 0
        for line in f.readlines():
            line = line.strip()
            word_to_idx[line] = count
            idx_to_word[count] = line
            count+=1
    return word_to_idx,idx_to_word


def make_text_cases(sentences, types, type_to_texts):
    X, Y, Text= [], [], []
    size = len(types)
    for i in range(size):
        x,y = make_case(sentences[i])
        X.append(x)
        Y.append(y)
        Text.append(type_to_texts[types[i][0]])
    return X,Y,Text

def make_case(s):
    tmpIn = torch.LongTensor(1,1, len(s) - 1)
    tmpOut = torch.LongTensor(1,1, len(s) - 1)
    for i in range(1, len(s)):
        w = s[i]
        w_b = s[i - 1]
        tmpIn[0][i - 1] = w_b
        tmpOut[0][i - 1] = w
    return autograd.Variable(tmpIn), autograd.Variable(tmpOut)
