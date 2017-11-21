import json
import codecs
vocab_threhold = 5

data = {'train':[],'test':[],'dev':[]}

symbols = ['<p>','</p>','<s>','</s>']
with open('data/DUC2014/data.json','r') as f:
    jsons = json.load(f)
    for one_json in jsons:
        #print(one_json)
        text = one_json['data']
        for symbol in symbols:
            text = text.replace(symbol, '')
        labels = one_json['label']
        set = one_json['set']
        for label in labels:
            for symbol in symbols:
                label = label.replace(symbol,'')
            data[set].append(text+' ^ '+label+'\n')
samples = []
with open('data/DUC2003/input.txt') as f:
    for line in f.readlines():
        line = '<d> '+line.strip()+' </d>'
        samples.append(line)
for i in range(4):
    with open('data/DUC2003/task1_ref'+str(i)+'.txt') as f:
        for i,line in enumerate(f.readlines()):
            line = '<d> '+line.strip()+' </d>'
            data['train'].append(samples[i]+' ^ '+line+'\n')

'''
samples = []
with open('data/Giga/input.txt') as f:
    for line in f.readlines():
        print(line)
        line = '<d>'+line.strip()+'</d>'
        samples.append(line)
with open('data/Giga/task1_ref0.txt') as f:
    for i,line in enumerate(f.readlines()):
        line = '<d>'+line.strip()+'</d>'
        if len(samples[i].split())>5:
            data['train'].append(samples[i]+' ^ '+line+'\n')
'''

samples = []
count = 0
with codecs.open('../train/train.article.txt','r','utf-8') as f:
    for line in f.readlines():

        if count >100000:
            break
        line = '<d> '+line.strip()+' </d>'
        samples.append(line)
        count += 1

with codecs.open('../train/train.title.txt','r','utf-8') as f:
    for i,line in enumerate(f.readlines()):
        line = '<d> '+line.strip()+' </d>'
        if len(samples[i].split())>5:
            data['train'].append(samples[i]+' ^ '+line+'\n')

        if i == 100000:
            break
print('read data end')
vocab = {}
for text in data['train']:
    words = text.split()
    for word in words:
        if not word in vocab:
            vocab[word] = 0
        vocab[word]+=1

with open('data/vocab.txt','w') as vocab_file:
    vocab_file.write('UNK\n')
    for word in vocab:
        if vocab[word] >= vocab_threhold:
            vocab_file.write(word+'\n')


for set in data:
    with open('data/'+set+'.txt','w') as output:
        for text in data[set]:
            words = text.split()
            is_text = True
            for word in words:
                if word == '^':
                    is_text = False
                if (not word in vocab) or (vocab[word] < vocab_threhold):
                    if set == 'train' or is_text:
                        word = 'UNK'
                output.write(word+' ')
            output.write('\n')
