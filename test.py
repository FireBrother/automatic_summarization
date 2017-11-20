from model import  *
from pythonrouge.pythonrouge import Pythonrouge
import torch
import torch.autograd as autograd

symbols = ['<d>','</d>','<p>','</p>','<s>','</s>']

def sample(model,text,word_to_idx, idx_to_word):
    states = model.initHidden()

    text = autograd.Variable(text)

    pre_summary = [word_to_idx['<d>']]
    output, _= model(text,autograd.Variable(torch.LongTensor(pre_summary).view(1,len(pre_summary))), states)
    for i in range(25):
        topv, topi = output[-1].data.topk(1)
        if topi[0] == word_to_idx['</d>']:
            break
        pre_summary.append(topi[0])
        states = model.initHidden()
        output,_ = model(text,autograd.Variable(torch.LongTensor(pre_summary).view(1,len(pre_summary))), states)
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
        count +=1
        print(output)
        print(data[1])
        if count>5:
            break
    rouge = Pythonrouge(summary_file_exist=False,
                        summary=summary, reference=reference,
                        n_gram=2, ROUGE_SU4=True, ROUGE_L=False,
                        recall_only=True, stemming=True, stopwords=True,
                        word_level=True, length_limit=True, length=50,
                        use_cf=False, cf=95, scoring_formula='average',
                        resampling=True, samples=1000, favor=True, p=0.5)
    score = rouge.calc_score()
    print(score)
