from model import  *
from pythonrouge.pythonrouge import Pythonrouge
import torch
import torch.autograd as autograd

symbols = ['<d>','</d>','<p>','</p>','<s>','</s>']

def sample(model,text,word_to_idx, idx_to_word):

    text = autograd.Variable(text)

    pre_summary = [word_to_idx['<d>']]
    output, _= model(text,autograd.Variable(torch.LongTensor(pre_summary).view(1,len(pre_summary))))
    for i in range(25):
        word_weights = output[-1].squeeze().data.exp()
        word_idx = torch.multinomial(word_weights, 1)[0]
        pre_summary.append(word_idx)
        #print(word_idx,word_to_idx['</d>'])
        if word_idx == word_to_idx['</d>']:
            break
        states = model.initHidden()
        output,_ = model(text,autograd.Variable(torch.LongTensor(pre_summary).view(1,len(pre_summary))))
    summary = []
    for token in pre_summary:
        if not (idx_to_word[token] in symbols ):
            summary.append(idx_to_word[token])
    summary = " ".join(summary)
    return summary


def evaluation(model,data_loader,word_to_idx,idx_to_word):
    summary = []
    reference = []
    count = 0
    for data in data_loader:

        reference.append([])
        reference[count].append(data[1])
        output = sample(model,data[0],word_to_idx,idx_to_word)
        summary.append([output])

        count += 1
        if count<5:
            print(output)
    rouge = Pythonrouge(summary_file_exist=False,
                        summary=summary, reference=reference,
                        n_gram=2, ROUGE_SU4=True, ROUGE_L=False,
                        recall_only=True, stemming=True, stopwords=True,
                        word_level=True, length_limit=True, length=50,
                        use_cf=False, cf=95, scoring_formula='average',
                        resampling=True, samples=1000, favor=True, p=0.5)
    score = rouge.calc_score()
    print(score)
