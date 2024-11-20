
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
import torch_geometric
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops
from torch_geometric.utils import get_laplacian
from torch.nn import Linear
from torch.nn.parameter import Parameter
import pandas as pd


def xavier_init(m):
    if type(m) == nn.Linear:
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.0)


class GraphConvolution(nn.Module):
    def __init__(self, in_feature, out_feature, bias=True):
        super().__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.weight = nn.Parameter(torch.FloatTensor(in_feature, out_feature))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_feature))
        nn.init.xavier_normal_(self.weight.data)
        if self.bias is not None:
            self.bias.data.fill_(0.0)

    def forward(self, x, adj):
        support = torch.mm(x, self.weight)
        output = torch.sparse.mm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class EvenProp(MessagePassing):
    def __init__(self, K, alpha, Init, **kwargs):
        super(EvenProp, self).__init__(aggr='add', **kwargs)
        self.K = K
        self.alpha = alpha
        self.Init = Init
        TEMP = alpha * (1 - alpha) ** (2 * torch.arange(K // 2 + 1))
        self.temp = nn.Parameter(TEMP.clone().detach(), requires_grad=False)

    def forward(self, x, edge_index, edge_weight=None):
        edge_index = edge_index.long()
        edge_index1, norm1 = get_laplacian(edge_index, edge_weight, normalization='sym', dtype=x.dtype, num_nodes=x.size(0))
        edge_index2, norm2 = add_self_loops(edge_index1, -norm1, fill_value=1., num_nodes=x.size(0))

        hidden = x * self.temp[0]
        for k in range(self.K - 1):
            x = self.propagate(edge_index2, x=x, norm=norm2)
            x = self.propagate(edge_index2, x=x, norm=norm2)
            gamma = self.temp[k + 1]
            hidden += gamma * x
            break
        return hidden

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j


class GCN_Encoder(nn.Module):
    def __init__(self, in_dim, hgcn_dim, dropout):
        super(GCN_Encoder, self).__init__()
        self.gc1 = GraphConvolution(in_dim, hgcn_dim)
        self.gc2 = GraphConvolution(hgcn_dim, hgcn_dim)
        self.dropout = dropout

    def forward(self, H, G):
        H = self.gc1(H, G)
        H = F.leaky_relu(H, 0.25)
        H = F.dropout(H, self.dropout, training=True)

        H = self.gc2(H, G)
        H = F.leaky_relu(H, 0.25)
        return H


class Event_Encoder(nn.Module):
    def __init__(self, in_dim, hgcn_dim, hidden_dim, K, alpha, Init, dropout):
        super(Event_Encoder, self).__init__()
        self.lin1 = Linear(hgcn_dim, hidden_dim)
        self.lin2 = Linear(hidden_dim, hgcn_dim)
        self.prop1 = EvenProp(K, alpha, Init)
        self.dropout = dropout
        self.in_dim = in_dim

    def reset_parameters(self):
        self.lin1.reset_parameters()

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        x = self.prop1(x, edge_index)
        return x


class Decoder(nn.Module):
    def __init__(self, train_W):
        super().__init__()
        self.train_W = nn.Parameter(train_W)

    def forward(self, H, drug_num, target_num):
        HR = H[0:drug_num]
        HD = H[drug_num:(drug_num + target_num)]
        supp1 = torch.mm(HR, self.train_W)
        decoder = torch.mm(supp1, HD.transpose(0, 1))
        return decoder


def to_edge_index(matrix):
    num_drugs = matrix.shape[0]
    rows, cols = matrix.nonzero(as_tuple=True)
    cols += num_drugs
    edge_index = torch.stack([rows, cols], dim=0)
    return edge_index


class DualEncoderModel(nn.Module):
    def __init__(self, in_dim, hgcn_dim, hidden_dim, K, alpha, Init, train_W, dropout):
        super(DualEncoderModel, self).__init__()
        self.gcn_encoder = GCN_Encoder(in_dim, hgcn_dim, dropout)
        self.event_encoder = Event_Encoder(in_dim, hgcn_dim, hidden_dim, K, alpha, Init, dropout)
        self.decoder = Decoder(train_W)
        self.dropout = dropout

    def forward(self, H, G, drug_num, target_num, w):
        # GCN Encoding
        H1 = self.gcn_encoder(H, G)

        # Event Encoding
        drug_protein_matrix = G[:drug_num, drug_num:(drug_num + target_num)]
        drug_protein_matrix = drug_protein_matrix.float()
        edge_index = to_edge_index(drug_protein_matrix)
        H1 = H1.to('cuda:0')
        edge_index = edge_index.to('cuda:0')
        H2 = self.event_encoder(H1, edge_index)

        # Fusion of encodings
        H = w * H1 + (1 - w) * H2

        # Decoding
        decoder_output = self.decoder(H, drug_num, target_num)
        return decoder_output
