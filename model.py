# coding=utf-8
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.init import *


class CNN_RNN(nn.Module):
    def __init__(self, vocab_size, input_channel, hidden_channel, embedding_dim, hidden_dim, num_layers=1):
        super(CNN_RNN, self).__init__()
        self.num_layers = num_layers
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.ReLU = nn.ReLU()
        self.conv1 = nn.ConvTranspose2d(input_channel, hidden_channel, 5, 2, 0)
        self.pool1 = nn.AdaptiveMaxPool2d((32, 256))
        self.conv2 = nn.ConvTranspose2d(hidden_channel, 1, 5, 2, 0)
        self.pool2 = nn.AdaptiveMaxPool2d((12, 256))
        # pooling目标维数瞎比写的
        self.linear_hidden = nn.Linear(self.embedding_dim,self.hidden_dim*self.num_layers)
        self.linear1 = nn.Linear(12 * 256, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim *2, self.hidden_dim, num_layers=self.num_layers, dropout=0.1, batch_first=True)
        self.linear_output = nn.Linear(self.hidden_dim, vocab_size)
        self.init_weights()

    def init_weights(self):
        self.embeddings.weight.data.normal_()
        self.linear_output.bias.data.fill_(0)
        self.linear_output.weight.data.uniform_(-0.1, 0.1)
        self.linear1.bias.data.fill_(0)
        self.linear1.weight.data.uniform_(-0.1, 0.1)
        self.linear_hidden.bias.data.fill_(0)
        self.linear_hidden.weight.data.uniform_(-0.1, 0.1)
        for weights in self.lstm.all_weights:
            for weight in weights:
                if len(weight.size()) > 1:
                    weight = orthogonal(weight)

    def forward(self, text_input, summary_input):
        length = summary_input.size(1)
        #print(summary_input.size())
        summary_embeds = self.embeddings(summary_input)
        # rnn部分输入summary_input Tensor(batch_size = 1, summary_length)
        text_embeds = self.embeddings(text_input)
        # cnn部分输入text_input Tensor(batch_size = 1, text_length)
        # TODO
        # 处理多channel输入

        text_embeds = text_embeds.view(1, 1, text_embeds.size(1), text_embeds.size(2))

        text_features_1 = self.ReLU(self.pool1(self.conv1(text_embeds)))
        text_features_2 = self.ReLU(self.pool2(self.conv2(text_features_1)))
        text_features_2 = text_features_2.view(-1, 12 * 256)
        #print(text_features_2.size())
        text_embed = self.ReLU(self.linear1(text_features_2))

        hidden = (self.ReLU(self.linear_hidden(text_embed)).view(self.num_layers, 1, self.hidden_dim),Variable(torch.zeros(self.num_layers, 1, self.hidden_dim)))

        text_embeds = []
        for i in range(length):
            text_embeds.append(text_embed.view(1,1,self.embedding_dim))
        text_embeds = torch.cat(text_embeds, 1)

        # embeds : (1, summary_length, embed_dim  )
        #print(embeds.size())
        #print(summary_embeds.size(),text_embeds.size())
        summary_embeds = torch.cat((summary_embeds,text_embeds),2)
        output, hidden = self.lstm(summary_embeds, hidden)

#        output = torch.cat((output, text_embeds), 2)
        #print(output.size())
        output = output.contiguous().view(output.size(0) * output.size(1), output.size(2))
        output = self.linear_output(output)

        return output, hidden

    def initHidden(self):
        return (Variable(torch.zeros(self.num_layers, 1, self.hidden_dim)),
                Variable(torch.zeros(self.num_layers, 1, self.hidden_dim)))

    def initHidden_gpu(self):
        return (Variable(torch.zeros(self.num_layers, 1, self.hidden_dim)).cuda(),
                Variable(torch.zeros(self.num_layers, 1, self.hidden_dim)).cuda())
