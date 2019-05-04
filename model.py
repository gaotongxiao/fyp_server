import dill
import copy
import math
from torchtext import data, datasets
import string
import nltk
from nltk.corpus import stopwords
import torch
from torch import nn
import numpy as np
import pdb
import gc

def getSortedOrder(lens):
    sortedLen, fwdOrder = torch.sort(
        lens.contiguous().view(-1), dim=0, descending=True)
    _, bwdOrder = torch.sort(fwdOrder)
    sortedLen = sortedLen.cpu().numpy().tolist()
    return sortedLen, fwdOrder, bwdOrder

def dynamicRNN(rnnModel,
               seqInput,
               seqLens):
    '''
    Inputs:
        rnnModel     : Any torch.nn RNN model
        seqInput     : (batchSize, maxSequenceLength, embedSize)
                        Input sequence tensor (padded) for RNN model
        seqLens      : batchSize length torch.LongTensor or numpy array
        initialState : Initial (hidden, cell) states of RNN

    Output:
        A single tensor of shape (batchSize, rnnHiddenSize) corresponding
        to the outputs of the RNN model at the last time step of each input
        sequence. If returnStates is True, also return a tuple of hidden
        and cell states at every layer of size (num_layers, batchSize,
        rnnHiddenSize)
    '''
    sortedLen, fwdOrder, bwdOrder = getSortedOrder(seqLens)
    sortedSeqInput = seqInput.index_select(dim=0, index=fwdOrder)
    packedSeqInput = nn.utils.rnn.pack_padded_sequence(
        sortedSeqInput, lengths=sortedLen, batch_first=True)

    output, _ = rnnModel(packedSeqInput)
    output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)

    rnn_output = output.index_select(dim=0, index=bwdOrder)
    return rnn_output

class Encoder(nn.Module):
    def __init__(self, vocab, device=torch.device('cpu')):
        super().__init__()
        self.hidden_size = 256
        self.embedding_size = 300
        self.embedding = nn.Embedding.from_pretrained(copy.deepcopy(vocab.vectors), freeze=True)
        self.rnn = nn.GRU(input_size=self.embedding_size, hidden_size=self.hidden_size, batch_first=False, bidirectional=True)
        self.h_u = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.u_w = nn.Parameter(nn.init.xavier_normal_(torch.Tensor(1, 1, self.hidden_size)))
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(-1)
        self.dropout = nn.Dropout(p=0.1)
    def forward(self, data, ret_top_words=False):
        # datalen = 128
        data = self.dropout(self.embedding(data))
        output, _ = self.rnn(data)
        output = torch.transpose(output, 1, 0)
        #output = dynamicRNN(self.rnn, data, data_len)
        u = self.relu(self.h_u(output))
        scores = torch.sum(self.u_w * u, -1)
        alpha = self.softmax(scores)
        if ret_top_words:
           return torch.sort(alpha, 1, True)
           #return torch.topk(scores, ret_top_words)
        s = torch.sum(alpha.unsqueeze(-1) * output, 1)
        return s

class Comparer(nn.Module):
    def __init__(self, vocab, device=torch.device('cpu')):
        super().__init__()
        self.hidden_size = 512
        self.layer_size = 128
        self.encoder = Encoder(vocab, device)
        self.emb = nn.Linear(self.hidden_size*4, self.layer_size)
        self.score = nn.Linear(self.layer_size, 2)
        self.relu = nn.ReLU()
        self.layer1 = nn.Linear(self.layer_size, self.layer_size)
        self.bn = nn.BatchNorm1d(self.hidden_size)
        self.bn2 = nn.BatchNorm1d(self.layer_size)
        self.dropout = nn.Dropout(0.1)
    def forward(self, q1, q2):
        q1_f = self.bn(self.encoder(q1))
        q2_f = self.bn(self.encoder(q2))
        emb = self.dropout(self.bn2(self.relu(self.emb(self.relu(torch.cat([q1_f, q2_f, (q1_f-q2_f).pow(2), q1_f * q2_f], 1))))))
        return self.score(emb)
