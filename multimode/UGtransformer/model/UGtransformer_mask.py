import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch_geometric.utils import to_dense_batch, to_dense_adj, dense_to_sparse

class FullyConnectedGT_UGformerV2(nn.Module):

    def __init__(self, feature_dim_size, ff_hidden_size, num_classes,
                 num_self_att_layers, dropout, num_GNN_layers, batch_size, nhead):
        super(FullyConnectedGT_UGformerV2, self).__init__()
        self.feature_dim_size = feature_dim_size
        self.ff_hidden_size = ff_hidden_size
        self.num_classes = num_classes
        self.num_self_att_layers = num_self_att_layers #Each layer consists of a number of self-attention layers
        self.num_GNN_layers = num_GNN_layers
        self.batch_size = batch_size
        self.nhead = nhead
        self.lst_gnn = torch.nn.ModuleList()
        #
        self.ugformer_layers = torch.nn.ModuleList()
        for _layer in range(self.num_GNN_layers):
            encoder_layers = TransformerEncoderLayer(d_model=16, nhead=self.nhead, dim_feedforward=self.ff_hidden_size, dropout=0.5) # Default batch_first=False (seq, batch, feature)
            self.ugformer_layers.append(TransformerEncoder(encoder_layers, self.num_self_att_layers))
            self.lst_gnn.append(GraphConvolution(16, 16, act=torch.tanh))

        # Linear function
        self.predictions = torch.nn.ModuleList()
        self.dropouts = torch.nn.ModuleList()
        self.lins = torch.nn.ModuleList()
        # BatchNormalize
        self.bn = nn.BatchNorm1d(self.feature_dim_size)

        for i in range(self.num_GNN_layers):
            self.predictions.append(nn.Linear(16, self.num_classes))
            self.dropouts.append(nn.Dropout(dropout))
            if i == 0:
                self.lins.append(nn.Linear(self.feature_dim_size,16))
            else:
                self.lins.append(nn.Linear(16,16))
        #self.prediction = nn.Linear(16, self.num_classes)
        #self.dropout = nn.Dropout(dropout)

    def forward(self, graph_batch):
        x, edge_index, batch = graph_batch.x.float(), graph_batch.edge_index.long(), graph_batch.batch
        x, mask = to_dense_batch(x, batch=batch)
        #x = F.normalize(x, dim=1)
        adj = to_dense_adj(edge_index, batch=batch)
        prediction_scores = 0
        input_Tr = x
        for layer_idx in range(self.num_GNN_layers):

            input_Tr = self.lins[layer_idx](input_Tr)

            mask = ~mask.view(-1, 16) #boolean value 'true' in padding_mask matirx will become nan
            # self-attention over all nodes
            input_Tr = self.ugformer_layers[layer_idx](input_Tr, src_key_padding_mask=mask) #[batch_size,seq_length, d_model] for pytorch transformer
            mask = ~mask.view(16, -1)
            input_Tr = torch.where(torch.isnan(input_Tr), torch.zeros_like(input_Tr), input_Tr)
            # take a sum over neighbors followed by a linear transformation and an activation function --> similar to GCN
            input_Tr = self.lst_gnn[layer_idx](input_Tr, adj, padding_mask=mask)
            # take a sum over all node representations to get graph representations
            graph_embedding = torch.sum(input_Tr, dim=1)
            graph_embedding = self.dropouts[layer_idx](graph_embedding)
            # Produce the final scores
            prediction_scores += self.predictions[layer_idx](graph_embedding)
        # # Can modify the code by commenting Lines 48-51 and uncommenting Lines 33-34, 53-56 to only use the last layer to make a prediction.
        #graph_embedding = torch.sum(input_Tr, dim=1)
        #graph_embedding = self.dropout(graph_embedding)
        ## Produce the final scores
        #prediction_scores = self.prediction(graph_embedding)

        return prediction_scores


""" GCN layer, similar to https://arxiv.org/abs/1609.02907 """
class GraphConvolution(torch.nn.Module):
    def __init__(self, in_features, out_features, batch_size=16, act=torch.tanh, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.batch_size = batch_size
        self.weight = nn.Parameter(torch.FloatTensor(batch_size, in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(batch_size,out_features))
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

    def forward(self, input, adj, padding_mask):
        support = torch.bmm(input, self.weight)
        if padding_mask is not None:
            support = support * padding_mask.unsqueeze(-1)
        output = torch.bmm(adj, support)
        if padding_mask is not None:
            output = output * padding_mask.unsqueeze(-1)
        if self.bias is not None:
            output = output + self.bias
        x = output.view(self.batch_size, self.out_features, -1)
        x = self.bn(x)
        x = x.view(self.batch_size, -1, self.out_features)
        x = self.act(x)
        return x

def label_smoothing(true_labels: torch.Tensor, classes: int, smoothing=0.1):
    """
    if smoothing == 0, it's one-hot method
    if 0 < smoothing < 1, it's smooth method
    """
    assert 0 <= smoothing < 1
    confidence = 1.0 - smoothing
    label_shape = torch.Size((true_labels.size(0), classes))
    with torch.no_grad():
        true_dist = torch.empty(size=label_shape, device=true_labels.device)
        true_dist.fill_(smoothing / (classes - 1))
        true_dist.scatter_(1, true_labels.data.view(1, -1), confidence)
    return true_dist

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

    return Adj_block