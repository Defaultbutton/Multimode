

#! /usr/bin/env python
import torch
import torch.nn as nn
import  torch.nn.functional as F
torch.manual_seed(123)

import numpy as np
np.random.seed(123)
import time

from UGtransformer.model.GCN import GcnNet
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
#from UGtransformer.utils.util import *
from UGtransformer.utils.util import load_data, separate_data,get_kfold_data
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(123)

# Parameters
# ==================================================

parser = ArgumentParser("GCN", formatter_class=ArgumentDefaultsHelpFormatter, conflict_handler='resolve')

parser.add_argument("--run_folder", default="../", help="")
parser.add_argument("--dataset", default="BRCA", help="Name of the dataset.")
parser.add_argument("--learning_rate", default=0.001, type=float, help="Learning rate")
parser.add_argument("--num_epochs", default=50, type=int, help="Number of training epochs")
parser.add_argument("--model_name", default='GCN', help="")
parser.add_argument("--dropout", default=0.5, type=float, help="")
parser.add_argument("--num_hidden_layers", default=1, type=int, help="")
parser.add_argument("--nhead", default=1, type=int, help="")
parser.add_argument("--num_timesteps", default=3, type=int, help="Number of self-attention layers within each UGformer layer")
parser.add_argument("--ff_hidden_size", default=256, type=int, help="The hidden size for the feedforward layer")
parser.add_argument('--fold_idx', type=int, default=1, help='The fold index. 0-9.')
args = parser.parse_args()

print(args)

# Load data
print("Loading data...")

use_degree_as_tag = False
#args.dataset = 'BRCA'

graphs, num_classes, _ = load_data(args.dataset)
# graph_labels = np.array([graph.label for graph in graphs])
# train_idx, test_idx = separate_data_idx(graphs, args.fold_idx)
#train_graphs, test_graphs = separate_data(graphs, args.fold_idx)
train_graphs, test_graphs = get_kfold_data(5, 1, graphs)
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
    graph_label = torch.zeros(1, 5)
    # 将one_hot中对应位置的值设为1
    graph_label[0, graph.y.item()] = 1
    #graph_label = label_smoothing(graph_label.long(),5)
    graph_x = F.normalize(graph.x, dim=0)
    return Adj_block, graph_x.to(device), graph_label.to(device)

print("Loading data... finished!")

model = GcnNet(input_dim=feature_dim_size).to(device)



def cross_entropy(pred, soft_targets): # use nn.CrossEntropyLoss if not using soft labels in Line 159
    logsoftmax = nn.LogSoftmax(dim=1)
    return torch.mean(torch.sum(- soft_targets * logsoftmax(pred), 1))
#criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

def train():
    model.train() # Turn on the train mode
    total_loss = 0.
    for graph in train_graphs:
        Adj_block, node_features, graph_label = get_data(graph) # one graph per step. should modify to use "padding" (for node_features and Adj_block) within a batch size???
        prediction_score = model.forward(Adj_block, node_features)
        #for name, parms in model.named_parameters():
            #print('-->name:', name)
            #print('-->para:', parms)
            #print('-->grad_requirs:', parms.requires_grad)
            #print('-->grad_value:', parms.grad)
            #print("===")
        # loss = criterion(prediction_scores, graph_labels)
        #print("prediction_score:",prediction_score)
        #print("graph_label:",graph_label)
        #graph_label = label_smoothing(graph_label.long(),5)
        loss = cross_entropy(torch.unsqueeze(prediction_score, 0), graph_label)
        #loss = criterion(torch.unsqueeze(prediction_score, 0), graph_label)
        optimizer.zero_grad()
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

    return total_loss

def evaluate():
    model.eval() # Turn on the evaluation mode
    total_loss = 0.
    with torch.no_grad():
        # evaluating
        prediction_output = []
        for i in range(0, len(test_graphs)):
            Adj_block, node_features, graph_label = get_data(test_graphs[i])
            prediction_score = model.forward(Adj_block, node_features).detach()
            #print("Adj_block:",Adj_block)
            #print("node_features:",node_features)
            prediction_output.append(torch.unsqueeze(prediction_score, 0))
            #print(prediction_score)
    prediction_output = torch.cat(prediction_output, 0)
    #print(prediction_output)
    predictions = prediction_output.max(1, keepdim=True)[1]
    labels = [data.y for data in test_graphs]
    labels = torch.cat(labels, dim=0).to(device)
    #labels = torch.LongTensor([graph.label for graph in test_graphs]).to(device)
    correct = predictions.eq(labels.view_as(predictions)).sum().cpu().item()
    acc_test = correct / float(len(test_graphs))
    return acc_test

"""main process"""
import os
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
    train_loss = train()
    cost_loss.append(train_loss)
    acc_test = evaluate()
    print('| epoch {:3d} | time: {:5.2f}s | loss {:5.2f} | test acc {:5.2f} | '.format(
                epoch, (time.time() - epoch_start_time), train_loss, acc_test*100))

    write_acc.write('epoch ' + str(epoch) + ' fold ' + str(args.fold_idx) + ' acc ' + str(acc_test*100) + '%\n')

write_acc.close()

