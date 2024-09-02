import torch
import math
from torch import nn
import torch.nn.functional as F
class GraphConvolution(torch.nn.Module):
    def __init__(self, in_features, out_features, act=torch.tanh, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.act = act
        self.bn = torch.nn.BatchNorm1d(out_features)

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, adj, input):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            output = output + self.bias
        output = self.bn(output)
        output = self.act(output)
        return output


class GcnNet(nn.Module):
    def __init__(self, input_dim):
        super(GcnNet, self).__init__()
        self.lin = nn.Linear(input_dim,16)
        self.gcn1 = GraphConvolution(16, 16)
        self.gcn2 = GraphConvolution(16, 16)
        self.predictions = nn.Linear(16,5)
        self.bn = nn.BatchNorm1d(16)
        self.dropouts = nn.Dropout(0.5)

    def forward(self, adjacency, feature):
        input = self.lin(feature)
        input = self.gcn1(adjacency, input)
        logits = self.gcn2(adjacency, input)
        graph_embedding = torch.sum(logits, dim=0)
        graph_embedding = self.dropouts(graph_embedding)
        # Produce the final scores
        prediction_scores = self.predictions(graph_embedding)
        return prediction_scores

