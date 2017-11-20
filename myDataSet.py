from torch.utils.data import Dataset
import torch

class myTrainDataSet(Dataset):
    def __init__(self,text_file,word_to_id):
        super(myTrainDataSet, self).__init__()
        self.data = []
        self.label = []
        with open(text_file,'r') as f:
            for line in f.readlines():
                #print(line)
                text = line.split('#')[0].split()
                label = line.split('#')[1].split()
                tmp_data = []
                tmp_label = []
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
                # print(torch.LongTensor(tmp_data).size())
                self.data.append(torch.LongTensor(tmp_data))
                self.label.append(torch.LongTensor(tmp_label))
    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        data = [self.data[item],self.label[item]]
        return data


class myTestDataSet(Dataset):
    def __init__(self,text_file,word_to_id):
        super(myTestDataSet, self).__init__()
        self.data = []
        self.label = []
        with open(text_file,'r') as f:
            pre_text = ""
            tmp_data = []
            tmp_labels = []
            for line in f.readlines():

                if not line == pre_text and not pre_text=="":
                    self.data.append(torch.LongTensor(tmp_data))
                    self.label.append(tmp_labels)

                    tmp_data = []
                    tmp_labels = []
                pre_text = line
                text = line.split('#')[0].split()
                label = line.split('#')[1]
                tmp_labels.append(label)
                for word in text:
                    if word in word_to_id:
                        tmp_data.append(word_to_id[word])
                    else:
                        tmp_data.append(0)
        #print(self.data)
        #print(self.label)
                # print(torch.LongTensor(tmp_data).size())
    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        data = [self.data[item],self.label[item]]
        return data





