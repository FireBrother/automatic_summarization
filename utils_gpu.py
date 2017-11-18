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
    f = codecs.open(filename,'r',encoding='utf-8')
    word_to_id = {'unk':0}
    id_to_word = {0:'unk'}
    count = 1
    for line in f.readlines():
        word = line.split(' ')[0]
        word_to_id[word] = count
        id_to_word[count] = word
        count+=1
    f.close()
    return word_to_id,id_to_word

def makeCases(sentences, types):
    X, Y, Type= [], [], []
    for i in range(len(types)):
        x,y,z = makeForOneCase(sentences[i],types[i])
        X.append(x)
        Y.append(y)
        Type.append(z)
    return X,Y,Type

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
    tmpIn = torch.LongTensor(1, len(s) - 1)
    tmpOut = torch.LongTensor(1, len(s) - 1)
    for i in range(1, len(s)):
        w = s[i]
        w_b = s[i - 1]
        tmpIn[0][i - 1] = w_b
        tmpOut[0][i - 1] = w
    return autograd.Variable(tmpIn), autograd.Variable(tmpOut)

def makeForOneCase(s, type):

    tmpIn = torch.LongTensor(1,len(s)-1)
    tmpOut = torch.LongTensor(1,len(s)-1)
    tmpType = torch.LongTensor(1,len(s)-1)
    for i in range(1, len(s)):
        w = s[i]
        w_b = s[i - 1]
        tmpIn[0][i-1] = w_b
        tmpOut[0][i-1] = w
        tmpType[0][i-1] = type[i-1]
    return autograd.Variable(tmpIn).cuda(), autograd.Variable(tmpOut).cuda(),autograd.Variable(tmpType).cuda()
