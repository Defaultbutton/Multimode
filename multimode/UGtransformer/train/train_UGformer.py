#! /usr/bin/env python
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(123)

import numpy as np
np.random.seed(123)
import time

from UGtransformer.model.UGtransformer import FullyConnectedGT_UGformerV2, label_smoothing
from UGtransformer.model.GAT import GAT,SpGAT
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
#from UGtransformer.utils.util import *
from UGtransformer.utils.util import load_data, separate_data,get_kfold_data
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(123)

# Parameters
# ==================================================

parser = ArgumentParser("UGformer", formatter_class=ArgumentDefaultsHelpFormatter, conflict_handler='resolve')

parser.add_argument("--run_folder", default="../", help="")
parser.add_argument("--dataset", default="BRCA", help="Name of the dataset.")
parser.add_argument("--learning_rate", default=0.01, type=float, help="Learning rate")
parser.add_argument("--num_epochs", default=500, type=int, help="Number of training epochs")
parser.add_argument("--model_name", default='UGformer', help="")
parser.add_argument("--dropout", default=0.5, type=float, help="")
parser.add_argument("--num_hidden_layers", default=1, type=int, help="")
parser.add_argument("--nhead", default=1, type=int, help="")
parser.add_argument("--num_timesteps", default=3, type=int, help="Number of self-attention layers within each UGformer layer")
parser.add_argument("--ff_hidden_size", default=256, type=int, help="The hidden size for the feedforward layer")
parser.add_argument('--fold_idx', type=int, default=1, help='The fold index. 0-9.')
parser.add_argument('--batch_size', default=16)
args = parser.parse_args()

print(args)

# Load data
print("Loading data...")

use_degree_as_tag = False
#args.dataset = 'BRCA'


graphs, num_classes, _ = load_data(args.dataset)
train_set, test_set = get_kfold_data(5, 1, graphs)
feature_dim_size = list(graphs)[0].x.shape[1]
#print(feature_dim_size)


def get_Adj_matrix(graph):
    """权重矩阵"""
    Adj_block_idx = torch.LongTensor(graph.edge_index)
    Adj_block_elem = torch.ones(Adj_block_idx.shape[1])

    num_node = graph.num_nodes
    self_loop_edge = torch.LongTensor([range(num_node), range(num_node)])
    elem = torch.ones(num_node)
    Adj_block_idx = torch.cat([Adj_block_idx, self_loop_edge], 1)
    Adj_block_elem = torch.cat([Adj_block_elem, elem], 0)

    Adj_block = torch.sparse.FloatTensor(Adj_block_idx, Adj_block_elem, torch.Size([num_node, num_node]))

    return Adj_block.to(device) # can implement and tune for the re-normalized adjacency matrix D^-1/2AD^-1/2 or D^-1A like in GCN/SGC ???

def get_data(graph):
    Adj_block = get_Adj_matrix(graph)
    y_one_hot = F.one_hot(graph.y, num_classes=5)
    #graph_label = label_smoothing(graph_label.long(),5)
    graph_x = F.normalize(graph.x, dim=0)
    return Adj_block, graph_x.to(device), y_one_hot.to(device)

print("Loading data... finished!")

model = FullyConnectedGT_UGformerV2(feature_dim_size=feature_dim_size, ff_hidden_size=args.ff_hidden_size,
                        num_classes=num_classes, dropout=args.dropout,
                        num_self_att_layers=args.num_timesteps,
                        num_GNN_layers=args.num_hidden_layers,
                        batch_size=args.batch_size,
                        nhead=1).to(device) # nhead is set to 1 as the size of input feature vectors is odd



def cross_entropy(pred, soft_targets): # use nn.CrossEntropyLoss if not using soft labels in Line 159
    logsoftmax = nn.LogSoftmax(dim=1)
    return torch.mean(torch.sum(- soft_targets * logsoftmax(pred), 1))
#criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

def train():
    model.train() # Turn on the train mode
    total_loss = 0.
    labels = []
    prediction_output = []
    for graph in train_set:
        graph = graph.to(device)
        #if graph.y.item() == 1:
        #    continue
        labels.append(graph.y)
        Adj_block, node_features, graph_label = get_data(graph) # one graph per step. should modify to use "padding" (for node_features and Adj_block) within a batch size???
        optimizer.zero_grad()
        prediction_score = model.forward(graph)
        prediction_output.append(torch.unsqueeze(prediction_score, 0))
        loss = cross_entropy(torch.unsqueeze(prediction_score, 0), graph_label)
        #loss = criterion(torch.unsqueeze(prediction_score, 0), graph_label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5) # prevent the exploding gradient problem
        optimizer.step()
        #print("=============更新之后===========")
        #for name, parms in model.named_parameters():
            #print('-->name:', name)
            #print('-->para:', parms)
            #print('-->grad_requirs:', parms.requires_grad)
            #print('-->grad_value:', parms.grad)
           #print("===")
        #print(optimizer)
        total_loss += loss.item()
    labels = torch.cat(labels, dim=0).to(device)
    prediction_output = torch.cat(prediction_output, 0)
    predictions = prediction_output.max(1, keepdim=True)[1]
    correct = predictions.eq(labels.view_as(predictions)).sum().cpu().item()
    acc_train = correct / float(len(train_set))
    return total_loss, acc_train

def evaluate():
    model.eval() # Turn on the evaluation mode
    total_loss = 0.
    with torch.no_grad():
        # evaluating
        prediction_output = []
        labels = []
        for i, graph in enumerate(test_set):
            #if graph.y.item() == 1:
            #    continue
            labels.append(graph.y)
            Adj_block, node_features, graph_label = get_data(graph)
            prediction_score = model.forward(Adj_block, node_features).detach()
            prediction_output.append(torch.unsqueeze(prediction_score, 0))
            #print(prediction_score)
    prediction_output = torch.cat(prediction_output, 0)
    predictions = prediction_output.max(1, keepdim=True)[1]
    #print(predictions)
    #labels = [data.y for data in test_graphs]
    labels = torch.cat(labels, dim=0).to(device)
    #labels = torch.LongTensor([graph.label for graph in test_graphs]).to(device)
    correct = predictions.eq(labels.view_as(predictions)).sum().cpu().item()
    acc_test = correct / float(len(test_set))
    return acc_test,labels.view_as(predictions).to(device), predictions.to(device),prediction_output.to(device)

"""main process"""
import os
from sklearn.metrics import confusion_matrix,roc_curve, roc_auc_score, auc
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import cycle


out_dir = os.path.abspath(os.path.join(args.run_folder, "../FullyConnectedGT_UGformerV2", args.model_name))
print("Writing to {}\n".format(out_dir))
# Checkpoint directory
checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
checkpoint_prefix = os.path.join(checkpoint_dir, "model")
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
write_acc = open(checkpoint_prefix + '_acc.txt', 'w')

cost_loss = []
for epoch in range(1, args.num_epochs + 1):
    epoch_start_time = time.time()
    train_loss, acc_train = train()
    cost_loss.append(train_loss)
    acc_test, y_true, y_pred, y_score = evaluate()
    print('| epoch {:3d} | time: {:5.2f}s | loss {:5.2f} |train acc {:5.2f}| test acc {:5.2f} | '.format(
                epoch, (time.time() - epoch_start_time), train_loss, acc_train*100, acc_test*100))

    write_acc.write('epoch ' + str(epoch) + ' fold ' + str(args.fold_idx) + ' acc ' + str(acc_test*100) + '%\n')


    '''y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().numpy()
    cm = confusion_matrix(y_true, y_pred)
    # 绘制混淆矩阵
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

    #预测概率y_score
    y_score = F.softmax(y_score, dim=1)
    y_score = y_score.cpu().numpy()

    n_classes = y_score.shape[1]

    # 初始化 FPR, TPR 和 AUC
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    # 计算每个类别的 ROC 曲线和 AUC
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true == i, y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # 计算微平均 ROC 曲线和 AUC
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])


    # 绘制 ROC 曲线
    lw = 2
    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'red', 'darkgreen'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.show()'''


write_acc.close()

