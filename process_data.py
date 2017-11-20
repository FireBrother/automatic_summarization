import json
vocab_threhold = 5

data = {'train':[],'test':[],'dev':[]}

symbols = ['<d>','</d>','<p>','</p>','<s>','</s>']
with open('data/DUC2014/data.json','r') as f:
    jsons = json.load(f)
    for one_json in jsons:
        print(one_json)
        text = one_json['data']
        labels = one_json['label']
        set = one_json['set']
        for label in labels:
            if not set == 'train':
                for symbol in symbols:
                    label = label.replace(symbol,'')
            data[set].append(text+' # '+label+'\n')

vocab = {}
for text in data['train']:
    words = text.split()
    for word in words:
        if not word in vocab:
            vocab[word] = 0
        vocab[word]+=1

with open('vocab.txt','w') as vocab_file:
    vocab_file.write('unk\n')
    for word in vocab:
        if vocab[word] >= vocab_threhold:
            vocab_file.write(word+'\n')


for set in data:
    with open(set+'.txt','w') as output:
        for text in data[set]:
            words = text.split()
            is_text = True
            for word in words:
                if word == '#':
                    is_label = False
                if (not word in vocab) or (vocab[word] < vocab_threhold):
                    if set == 'train' or is_text:
                        word = 'unk'
                output.write(word+' ')
            output.write('\n')
