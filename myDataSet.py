from torch.utils.data import Dataset
import torch

class myTrainDataSet(Dataset):
    def __init__(self,text_file,word_to_id):
        super(myTrainDataSet, self).__init__()
        self.data = []
        self.summary = []
        self.label = []
        with open(text_file,'r') as f:
            for line in f.readlines():
                text = line.split('^')[0].split()
                label = line.split('^')[1].split()
                tmp_data = []
                tmp_label = []
#                if len(self.data)==10 :
#                    break
                for word in text:
                    if word in word_to_id:
                        tmp_data.append(word_to_id[word])
                    else:
                        tmp_data.append(0)

                for word in label:
                    if word in word_to_id:
                        tmp_label.append(word_to_id[word])
                    else:
                        tmp_label.append(0)
                tmp_summay = tmp_label[:-1]
                # print(torch.LongTensor(tmp_data).size())
                self.data.append(torch.LongTensor(tmp_data))
                self.label.append(torch.LongTensor(tmp_label[1:]))
                self.summary.append(torch.LongTensor(tmp_summay))
    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        data = [self.data[item],self.label[item],self.summary[item]]
        return data


class myTestDataSet(Dataset):
    def __init__(self,text_file,word_to_id):
        super(myTestDataSet, self).__init__()
        self.data = []
        self.label = []
        with open(text_file,'r') as f:

            pre_text = ""
            tmp_labels = []
            tmp_data = []
            for line in f.readlines():

                text = line.split('^')[0]
                words = text.split()
                label = line.split('^')[1]
                if len(tmp_data)==0:
                    for word in words:
                        if word in word_to_id:
                            tmp_data.append(word_to_id[word])
                        else:
                            tmp_data.append(0)
                if not text == pre_text and not pre_text=="":
                    #print(pre_text,line)
                    self.data.append(torch.LongTensor(tmp_data))
                    #print(len(tmp_data))
                    self.label.append(tmp_labels)
                    #print(tmp_labels)
                    tmp_data = []
                    tmp_labels = []
    #                break
                pre_text = text
                tmp_labels.append(label)

            self.data.append(torch.LongTensor(tmp_data))
            print(len(tmp_data))
            self.label.append(tmp_labels)
    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        data = [self.data[item],self.label[item]]
        return data





