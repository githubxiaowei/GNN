import torch.nn as nn
import torch.nn.functional as F
from layers import ChebyConvolution


class ChebNet(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(ChebNet, self).__init__()
        self.cc1 = ChebyConvolution(nfeat, nhid)
        self.cc2 = ChebyConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.cc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.cc2(x, adj)
        return F.log_softmax(x, dim=1)
