import math

import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, num_nodes, in_features, out_features,  bias=None):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_nodes = num_nodes
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.weight_local = Parameter(torch.FloatTensor(num_nodes))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
        self.weight_local.data.uniform_(0, 2)

    def forward(self, x, adj):
        x = torch.mm(x, self.weight)
        left = adj
        # right = 3*torch.eye(self.num_nodes).cuda() - adj
        # x = torch.spmm(right, x)
        # x = torch.spmm(torch.diag(self.weight_local),x)
        x = torch.spmm(left, x)
        if self.bias is not None:
            return x + self.bias
        else:
            return x

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
