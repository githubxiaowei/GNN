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
        self.gc3 = GraphConvolution(nhid+nhid, nclass)
        self.dropout = dropout

    def forward0(self, x_in, adj):

        x_code = self.encoder(x_in)
        x_recover = self.decoder(x_code)

        x = self.gc2(x_in, adj)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)

        x = self.gc3(x, adj)
        x_out = F.log_softmax(x, dim=1)
        return x_out, x_recover

    def forward(self, x_in, adj):

        x_code = self.encoder(x_in)
        x_recover = self.decoder(x_code)

        x = torch.cat([self.gc2(x_in, adj), self.gc1(x_code, adj)], 1)
        x = F.elu(x, 0.1)
        x = F.dropout(x, self.dropout, training=self.training)

        x = self.gc3(x, adj)
        x_out = F.log_softmax(x, dim=1)
        return x_out, x_recover