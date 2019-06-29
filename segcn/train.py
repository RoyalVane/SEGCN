from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from utils import load_data, accuracy
from models import GCN
from optim_weight import WeightEMA, WeightEMA_reverse

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=1000,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--unsup_weight', type=float, default=2.0,
                    help='Weight for unsupervised consistency loss.')
parser.add_argument('--alpha', type=float, default=0.999,
                    help='Rate of updating teacher parameters.')
parser.add_argument('--label_num', type=int, default=20,
                    help='Number of labeled samples per class.')
parser.add_argument('--gpu', type=int, default=1,
                    help='GPU id.')
parser.add_argument('--self_thres', type=float, default=0.9,
                    help='Threshold for selecting pseudo labels in self-training.')
parser.add_argument('--tSNE', type=bool, default=False,
                    help='Whether to show tSNE graph.')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
adj_T, adj_S, features, labels, idx_train, idx_val, idx_test = load_data('cora', args.label_num)

idx_self = torch.LongTensor([]) #For self-training
pseudo_labels = torch.LongTensor([]) #For self-training

acc_val = 0 # For validation

# Model and optimizer
model_S = GCN(nfeat=features.shape[1],
            nhid=args.hidden,
            nclass=labels.max().item() + 1,
            dropout=args.dropout)

model_T = GCN(nfeat=features.shape[1],
            nhid=args.hidden,
            nclass=labels.max().item() + 1,
            dropout = 0)

student_params = list(model_S.parameters())
teacher_params = list(model_T.parameters())

for param in teacher_params:
            param.requires_grad = False

optimizer_S = optim.Adam(model_S.parameters(),
                         lr=args.lr, weight_decay = args.weight_decay)
optimizer_T = WeightEMA(teacher_params, 
                        student_params, alpha = args.alpha)

if args.cuda:
    model_S.cuda(args.gpu)
    model_T.cuda(args.gpu)
    features = features.cuda(args.gpu)
    adj_T = adj_T.cuda(args.gpu)
    adj_S = adj_S.cuda(args.gpu)
    labels = labels.cuda(args.gpu)
    pseudo_labels = pseudo_labels.cuda(args.gpu)
    idx_train = idx_train.cuda(args.gpu)
    idx_val = idx_val.cuda(args.gpu)
    idx_test = idx_test.cuda(args.gpu)
    idx_self = idx_self.cuda(args.gpu)

features, labels, pseudo_labels = Variable(features), Variable(labels), Variable(pseudo_labels)


def plot_embedding(data, label, title):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)

    fig = plt.figure()
    #ax = plt.subplot(111)
    for i in range(data.shape[0]):
        plt.text(data[i, 0], data[i, 1], str(label[i]),
                 color=plt.cm.Set1((label[i] + 1) / 10.),
                 fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    return fig


def train(epoch):
    global idx_self, pseudo_labels, acc_val 
    t = time.time()
    model_T.train()
    model_S.train()
    optimizer_S.zero_grad()
    optimizer_T = WeightEMA(teacher_params, 
                        student_params, alpha = args.alpha * (epoch / args.epochs))
    optimizer_T_R = WeightEMA_reverse(teacher_params, 
                        student_params, alpha = args.alpha * (epoch / args.epochs))
    
    output_SS, _ = model_S(features, adj_S)
    output_ST, _ = model_S(features, adj_T)
    output_TT, _ = model_T(features, adj_T)
    
    sup_loss = F.nll_loss(output_ST[idx_train], labels[idx_train])
    if bool(idx_self.numel()):
        self_loss = F.nll_loss(output_ST[idx_self], pseudo_labels)
    else:
        self_loss = 0
    consist_loss = F.binary_cross_entropy(F.softmax(output_SS, dim = 1), F.softmax(output_TT, dim = 1))
    acc_train = accuracy(output_TT[idx_train], labels[idx_train])
    total_loss = sup_loss  + consist_loss * (epoch / args.epochs) * args.unsup_weight + self_loss
    
    #If you try to train the baseline GCN, use the single supervised loss below.
    #total_loss = sup_loss
    
    total_loss.backward()
    optimizer_S.step()
    optimizer_T.step()

    model_T.eval()
    model_S.eval()
    output_T, _ = model_T(features, adj_T)
    output_S, _ = model_S(features, adj_S)

    #loss_val = F.nll_loss(output_T[idx_val], labels[idx_val])
    acc_val_now = accuracy(output_T[idx_val], labels[idx_val])
    if acc_val_now.data.item() > acc_val:
        acc_val = acc_val_now.data.item()
    else:
        optimizer_T_R.step()
    mask_T = F.softmax(output_T[idx_test], dim = 1).gt(args.self_thres - 0.3 * epoch / args.epochs)
    mask_S = F.softmax(output_S[idx_test], dim = 1).gt(args.self_thres - 0.2 * epoch / args.epochs)
    #mask_T = F.softmax(output_T[idx_test], dim = 1).gt(0.99)
    #mask_S = F.softmax(output_S[idx_test], dim = 1).gt(0.99)
    mask = mask_T * mask_S
    A = mask.nonzero().data
    if bool(A.numel()):
        #print(idx_test[A[:, 0]])
    #idx_self = torch.copy_(idx_test[A[:, 0]])
        idx_self.resize_(idx_test[A[:, 0]].size()).copy_(idx_test[A[:, 0]])
        pseudo_labels = pseudo_labels.data
        pseudo_labels.resize_(A[:, 1].size()).copy_(A[:, 1])
        pseudo_labels = Variable(pseudo_labels)
    
    print("epoch={}".format(epoch), "| acc_train={:.3f}".format(acc_train.data.item()), 
          "| acc_val={:.3f}".format(acc_val), "| time: {:.4f}s".format(time.time() - t))

def test():
    model_T.eval()
    output, tSNE_features = model_T(features, adj_T)
    if args.tSNE:
        print("Generating T-SNE...")
        tsne = TSNE(n_components=2, init='pca', random_state=0)
        tSNE_result = tsne.fit_transform(tSNE_features.data.cpu().numpy())
        tSNE_label = labels.data.cpu().numpy()
        fig = plot_embedding(tSNE_result, tSNE_label,
                         't-SNE embedding of Cora'
                         )
        plt.show(fig)
    #loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("***** Test set results *****:\n",
          "alpha = {:.3f}".format(args.alpha),
          "unsup_weight = {:.1f}".format(args.unsup_weight),
          "labeled_nodes_per_class = {}".format(args.label_num),
          "accuracy = {:.4f}".format(acc_test.data.item()))

    return acc_test.data.item()
    

if __name__ == '__main__':
    
    # Train model
    t_total = time.time()
    for epoch in range(args.epochs):
        train(epoch)
    
    # Testing
    result = test()
    fo = open("log/Cora.txt", "w")
    fo.write("Test set results:\n alpha = {:.3f}, \n unsup_weight = {:.1f}, \n labeled_nodes_per_class = {}, \
             \n accuracy= {:.4f}"
             .format(args.alpha, args.unsup_weight, args.label_num, result))
    fo.close()
    
