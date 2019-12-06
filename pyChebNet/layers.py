import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class ChebyConvolution(Module):
    def __init__(self, in_features, out_features, cheby_order=4, dropout=0.0,
                 use_bias=True):
        super(ChebyConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.cheby_order = cheby_order
        self.use_bias = use_bias

        self.weights = nn.ParameterList([
            Parameter(torch.FloatTensor(in_features, out_features))
            for _ in range(cheby_order)
        ])
        if use_bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU(inplace=True)
        self.initialize_parameters()

    def initialize_parameters(self):
        for i in range(self.cheby_order):
            self.weights[i].data.normal_(mean=0, std=0.05)
        if self.bias is not None:
            self.bias.data.normal_(mean=0, std=0.05)

    def forward(self, x, adj):
        terms = [
            x, torch.spmm(adj, x)
        ]
        for k in range(2, self.cheby_order):
            cheby_k = 2.0 * torch.spmm(adj, terms[k - 1]) - terms[k - 2]
            terms.append(cheby_k)

        output = None
        for k in range(self.cheby_order):
            reshape = terms[k].contiguous()
            reshape = reshape.view(-1, self.in_features)
            if output is None:
                output = reshape.mm(self.weights[k])
            else:
                output = torch.add(output, reshape.mm(self.weights[k]))
        if self.use_bias:
            output = torch.add(output, self.bias)

        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
