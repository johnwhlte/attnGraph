from utils.ops import MHABlock, Graph_Conv, Initializer, MHDABlock
import torch
from torch_geometric.nn import GCNConv
import torch.nn as nn
import numpy as np



class end2end(nn.Module):

    def __init__(self,feat_dim, in_dim, out_dim, pred_dim, n_attn, n_convs, conv_type, geo_edges, learned_edges=True, model_name='test'):
        super(end2end, self).__init__()
        self.attn_edges = nn.ModuleList()
        self.lin_embed = nn.Linear(feat_dim, in_dim)
        self.convs = nn.ModuleList()
        self.lin1 = nn.Linear(out_dim, 100)
        self.lin2 = nn.Linear(100, 100)
        self.lin3 = nn.Linear(100, pred_dim)
        self.leaky = nn.LeakyReLU()
        self.learned_edges = learned_edges
        self.model_name=model_name
        self.geo_edges = geo_edges

        if learned_edges:

            for i in range(n_attn):
                if i == 0:
                    self.attn_edges.append(MHABlock(in_dim, out_dim))
                else:
                    self.attn_edges.append(MHABlock(out_dim, out_dim))
        for j in range(n_convs):
            self.convs.append(conv_type(out_dim, out_dim))

        Initializer.weights_init(self)
        

    def forward(self, batch):

        x = batch.x

        x = self.lin_embed(x)

        if self.learned_edges:

            for i, attn_block in enumerate(self.attn_edges):
                if i == 0:
                    attn = attn_block(x, end=False)
                elif i < len(self.attn_edges) - 1:
                    attn = attn_block(attn, end=False)
                else:
                    adj = attn_block(attn, end=True)
        else:
            adj = batch.edge_index
        
        for i, conv in enumerate(self.convs):
            x = conv(x, adj)

        x = self.lin1(x)
        x = self.leaky(x)
        x = self.lin2(x)
        x = self.leaky(x)
        pred = self.lin3(x)

        return pred

class DoubleAttention(nn.Module):

    def __init__(self,feat_dim, in_dim, out_dim, pred_dim, n_attn, model_name='test'):
        super(DoubleAttention, self).__init__()
        self.attn_edges = nn.ModuleList()
        self.model_name = model_name
        self.lin_embed = nn.Linear(feat_dim, in_dim)
        self.convs = nn.ModuleList()
        self.ff1 = nn.ModuleList()
        self.ff2 = nn.ModuleList()
        self.lnorms = nn.ModuleList()
        self.lin1 = nn.Linear(out_dim, 100)
        self.lin2 = nn.Linear(100, 100)
        self.lin3 = nn.Linear(100, pred_dim)
        self.leaky = nn.LeakyReLU()

        for i in range(n_attn):
            if i == 0:
                self.attn_edges.append(MHDABlock(in_dim, out_dim))
                self.ff1.append(nn.Linear(out_dim, out_dim))
                self.ff2.append(nn.Linear(out_dim, out_dim))
                self.lnorms.append(nn.LayerNorm(out_dim))
            else:
                self.attn_edges.append(MHDABlock(in_dim, out_dim))
                self.ff1.append(nn.Linear(out_dim, out_dim))
                self.ff2.append(nn.Linear(out_dim, out_dim))
                self.lnorms.append(nn.LayerNorm(out_dim))


        Initializer.weights_init(self)
        

    def forward(self, batch):

        x = batch.x

        x = self.lin_embed(x)

        for i, attn_layer in enumerate(self.attn_edges):
            x = attn_layer(x) 
            x2 = self.ff1[i](x)
            x2 = self.leaky(x2)
            x2 = self.ff2[i](x2)
            #x+=x2
            x3 = x + x2
            x = self.lnorms[i](x3)

        x = self.lin1(x)
        x = self.leaky(x)
        x = self.lin2(x)
        x = self.leaky(x)
        pred = self.lin3(x)

        return pred