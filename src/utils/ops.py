import torch
import numpy as np
from torch_geometric.nn import GCNConv, SAGEConv, GATv2Conv
from torch_geometric.utils import to_torch_coo_tensor, to_dense_adj, add_self_loops
import torch.nn as nn
import torch.nn.functional as F



class Initializer(object):
    """ This is the exact same init as in Graph U-nets with a possible bias=False switch"""
    @classmethod
    def _glorot_uniform(cls, w):
        if len(w.size()) == 2:
            fan_in, fan_out = w.size()
        elif len(w.size()) == 3:
            fan_in = w.size()[1] * w.size()[2]
            fan_out = w.size()[0] * w.size()[2]
        else:
            fan_in = np.prod(w.size())
            fan_out = np.prod(w.size())
        limit = np.sqrt(6.0 / (fan_in + fan_out))
        w.uniform_(-limit, limit)

    @classmethod
    def _param_init(cls, m):
        if isinstance(m, nn.parameter.Parameter):
            cls._glorot_uniform(m.data)
        elif isinstance(m, nn.Linear):
            if m.bias is not None:
                m.bias.data.zero_()
                cls._glorot_uniform(m.weight.data)

    @classmethod
    def weights_init(cls, m):
        for p in m.modules():
            if isinstance(p, nn.ParameterList):
                for pp in p:
                    cls._param_init(pp)
            else:
                cls._param_init(p)

        for name, p in m.named_parameters():
            if '.' not in name:
                cls._param_init(p)

class contResLoss(nn.Module):

    def __init__(self):
        super(contResLoss, self).__init__()

    def forward(self, V: torch.tensor, edge_feats: torch.tensor, end2end = True) -> torch.tensor:

        feat_tensor = V.T

        u, v= feat_tensor[0], feat_tensor[1]#, feat_tensor[2]
        u = u.repeat(u.shape[0], 1)#.to_sparse().requires_grad_()
        v = v.repeat(v.shape[0], 1)#.to_sparse().requires_grad_()
        #z_c = z_c.repeat(z_c.shape[0], 1)
        uT = u.T
        vT = v.T
        #z_c2 = z_c.T
        u_diff = u - uT
        v_diff = v - vT
        #z_diff = z_c - z_c2
        #delta_x, delta_y = torch.cat((edge_feats.to_dense()[0], edge_feats.to_dense()[2], edge_feats.to_dense()[4])), torch.cat((edge_feats.to_dense()[1], edge_feats.to_dense()[3], edge_feats.to_dense()[5]))
        delta_x, delta_y = edge_feats.to_dense()[800:1200],edge_feats.to_dense()[1200:]
        #delta_x = torch.reshape(delta_x, (delta_x.shape // 4, delta_x.shape // 4))
        if end2end:
            cont_error_mat = u_diff.mul(delta_x) + v_diff.mul(delta_y)
        else:
            cont_error_mat = u_diff[400:].T[400:].T.mul(delta_x) + v_diff[400:].T[400:].T.mul(delta_y)

        cont_residual = torch.abs(cont_error_mat.to_dense()).mean()

        return torch.abs(cont_residual)

### New idea ####

class MHABlock(nn.Module):

    def __init__(self, in_dim, out_dim):
        super(MHABlock, self).__init__()
        self.Q = nn.Linear(in_dim, out_dim,bias=False)
        self.K = nn.Linear(in_dim,out_dim,bias=False)
        self.V = nn.Linear(in_dim, out_dim, bias=False)
        self.lnorm = nn.LayerNorm(out_dim)

    def head_splitter(self,x,qqkkv):

        # batch_size = 1 
        num_heads =10
        d_k = qqkkv[0].shape[1] // num_heads
        new_shape = (num_heads,qqkkv[0].shape[0], d_k)
        new_mats = []
        for mat in qqkkv:
            new_mats.append(torch.reshape(mat, new_shape))
    
        return new_mats 
    

    def forward(self,x, end=False):

        Q = self.Q(x)
        K = self.K(x)
        V = self.V(x)
        

        if not(end):
            [Q, K, V] = self.head_splitter(x, (Q,K, V))
            K = torch.reshape(K, (K.shape[0], K.shape[2], K.shape[1]))
            wghts = F.softmax((Q@K) / np.sqrt(K.shape[2]), dim=-1)
            values = wghts @ V
            values = torch.reshape(values, (values.shape[1],values.shape[0]*values.shape[2]))
            values+=x
            return self.lnorm(values)
        else:
            wghts = F.softmax((Q@K.T) / np.sqrt(K.shape[1]), dim=-1)
            #wghts = torch.reshape(wghts, (wghts.shape[1],wghts.shape[0]*wghts.shape[2]))
            ident = torch.div(wghts, wghts)
            adj_mat = torch.where(wghts>(1/K.shape[0]),ident,0)
            edges = adj_mat.nonzero().t().contiguous()

            return edges

class Graph_Conv(nn.Module):

    def __init__(self, in_dim, out_dim):
        super(Graph_Conv, self).__init__()
        self.lin1 = nn.Linear(in_dim, out_dim, bias=False)
        self.leaky = nn.LeakyReLU()
        self.lnorm = nn.LayerNorm(out_dim)


    def forward(self, x, A):

        deg = torch.sum(A, dim=1)
        normA = torch.div(A.T, deg).T

        node_rep = self.lin1((normA @ x))
        node_rep = self.leaky(node_rep)
        node_rep+=x
        node_rep = self.lnorm(node_rep)

        return node_rep

def mape_error(x, target):

    mape_err = torch.mean(torch.abs(torch.div((x - target),(target+0.00001))))

    return mape_err