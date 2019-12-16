from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
import torch.optim as optim

from utils import load_data1, load_data2, accuracy
from models import GCN

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()



#  显示训练集中每类节点的比例
def plot_bar():
    num_types = 7
    count_i = [np.sum(labels[idx_train].numpy() == i) for i in range(num_types)]  # count num of each type

    matplotlib.rcParams['font.sans-serif'] = ['SimHei']
    matplotlib.rcParams['axes.unicode_minus'] = False
    rects = plt.bar(x=range(num_types), height=count_i, width=0.4, alpha=0.8, color='red', label="训练集")
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2, height + 1, str(height), ha="center", va="bottom")
    plt.ylim(0, 50)
    plt.xlabel('节点类型')
    plt.ylabel('数量')
    plt.legend()
    plt.show()

# plot_bar()

def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output, feat_recover = model(features, adj)
    acc_train = accuracy(output[idx_train], labels[idx_train])
    # prediction = output.argmax(dim=1).view(num_nodes,-1).float()
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    # loss_var = torch.trace(torch.mm(prediction.T,torch.mm(L, prediction)))*0.01
    loss_recover = F.mse_loss(feat_recover,features, reduction='sum')
    loss_total = loss_train + loss_recover*0.1

    loss_total.backward()
    optimizer.step()

    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output, _ = model(features, adj)

    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    # print('Epoch: {:04d}'.format(epoch + 1),
    #       'loss_train: {:.4f}'.format(loss_train.item()),
    #       'loss_recover: {:.4f}'.format(loss_recover.item()),
    #       'acc_train: {:.4f}'.format(acc_train.item()),
    #       'loss_val: {:.4f}'.format(loss_val.item()),
    #       'acc_val: {:.4f}'.format(acc_val.item()),
    #       'time: {:.4f}s'.format(time.time() - t))


def test():
    model.eval()
    output, _ = model(features, adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test),
          "accuracy= {:.4f}".format(acc_test.item()))

    return acc_test.cpu().numpy()


for dataset in ['cora', 'citeseer', 'pubmed']:
    # np.random.seed(args.seed)
    # torch.manual_seed(args.seed)
    # if args.cuda:
    #     torch.cuda.manual_seed(args.seed)

    # Load data
    adj, features, labels, idx_train, idx_val, idx_test = load_data2(dataset)

    # num of nodes
    num_nodes = adj.shape[0]
    acc = []
    for _ in range(10):

        # Model and optimizer
        model = GCN(nfeat=features.shape[1],
                    nhid=args.hidden,
                    nclass=labels.max().item() + 1,
                    dropout=args.dropout)
        optimizer = optim.Adam(model.parameters(),
                               lr=args.lr, weight_decay=args.weight_decay)

        if args.cuda:
            model.cuda()
            features = features.cuda()
            adj = adj.cuda()
            labels = labels.cuda()

        # Train model
        t_total = time.time()
        for epoch in range(args.epochs):
            train(epoch)
        print("Optimization Finished!")
        print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

        # Testing
        acc.append(test())

    print('dataset:{}  mean:{}  std:{}'.format(dataset, np.mean(acc), np.sqrt(np.var(acc))))
