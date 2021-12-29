import torch
import torch.nn as nn

import numpy as np


class LSTMpred(nn.Module):
    def __init__(self,input_dim,hidden_dim):
        super(LSTMpred,self).__init__()
        self.input_dim=input_dim
        self.hidden_dim=hidden_dim
        self.lstm=nn.LSTM(input_dim,hidden_dim)
        self.hidden2out=nn.Linear(hidden_dim,1)
        self.hidden=self.init_hidden()

    def init_hidden(self):
        return (torch.zeros(1,1,self.hidden_dim),
                torch.zeros(1,1,self.hidden_dim))

    def forward(self,seq):
        lstm_out,self.hidden=self.lstm(seq.view(len(seq),1,-1),self.hidden)
        outdat=self.hidden2out(lstm_out.view(len(seq,-1)))
        return outdat