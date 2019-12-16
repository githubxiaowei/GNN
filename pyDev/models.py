import torch
import scipy.sparse as sp
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()
        ncode = 128
        self.encoder = nn.Linear(nfeat, ncode)
        self.decoder = nn.Linear(ncode, nfeat)
        self.gc1 = GraphConvolution(ncode, nhid)
        self.gc2 = GraphConvolution(nfeat, nhid)
        self.gc3 = GraphConvolution(nhid + nhid, nclass)
        self.dropout = dropout

    def forward(self, x_in, adj):
        # x = F.relu(self.layer1(x))
        # alpha = 0
        # S = F.normalize(torch.mm(x,x.T),dim=1,p=0)
        # adj = (adj + S * alpha) / (1 + alpha)
        # adj = torch.eye(adj.shape[0]).cuda()

        x_code = self.encoder(x_in)
        x_recover = self.decoder(x_code)

        # beta = 1
        # x = (self.gc2(x_in, adj) + beta*self.gc1(x_code, adj))/(1+beta)
        x = torch.cat([self.gc2(x_in, adj), self.gc1(x_code, adj)],1)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)

        # alpha = 0
        # S = F.normalize(torch.mm(x, x.T), dim=1, p=0)
        # adj = (adj + S * alpha) / (1 + alpha)
        # adj = torch.eye(adj.shape[0]).cuda()

        x = self.gc3(x, adj)
        x_out = F.log_softmax(x, dim=1)
        return x_out, x_recover
