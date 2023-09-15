import numpy as np
import os
import argparse
from trainer import Trainer
from net import end2end, DoubleAttention
import torch
from torch_geometric.nn import GCNConv, SAGEConv, GATv2Conv
import torch.nn.functional as F


def get_args():
    """
    Grabs all hyperparameter values from the config file specified when running ./train.sh
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('-batch_size', type=int, default=1, help='batch_size')
    parser.add_argument('-seed', type=int, default=24, help='seed')
    parser.add_argument('-num_epochs', type=int, default=100, help='num_epochs')
    parser.add_argument('-early_stop', type=int, default=5, help='early_stop')
    parser.add_argument('-lr', type=float, default=0.01, help='lr')
    parser.add_argument('-data_path', type=str, default='../data/', help='data_path')
    parser.add_argument('-device', type=str, default='cpu', help='device')
    parser.add_argument('-num_snaps', type=int, default=100, help='num_snaps')
    parser.add_argument('-funcs_list', type=str, nargs="+", default=['func1'], help='funcs')
    parser.add_argument('-alpha_vals', type=float, nargs="+", default=[0.1, 0.5], help='alpha_vals')
    parser.add_argument('-feat_dim', type=int, default=4, help='feat_dim')
    parser.add_argument('-in_dim', type=int, default=50, help='in_dim')
    parser.add_argument('-out_dim', type=int, default=100, help='out_dim')
    parser.add_argument('-pred_dim', type=int, default=2, help='pred_dim')
    parser.add_argument('-n_attn', type=int, default=2, help='n_attn')
    parser.add_argument('-n_convs', type=int, default=2, help='n_convs')

    args, _ = parser.parse_known_args()

    return args


if __name__=="__main__":

    args = get_args()
    options = vars(args)
    geo_edges = torch.load('/home/sysiphus/bigData/snapshots/true_edges.pt').to(args.device)
    models = ['lrnd_gcn', 'lrnd_sage', 'lrnd_gat', 'geo_gcn', 'geo_sage', 'geo_gat']
    lrn_bool = [True, True, True, False, False, False]
    conv_list = [GCNConv, SAGEConv, GATv2Conv, GCNConv, SAGEConv, GATv2Conv]
    #models = ['dbl_attn']
    torch.autograd.set_detect_anomaly(True)
    #lrn_bool = [True]
    #conv_list = [SAGEConv, SAGEConv]
    for i, mode in enumerate(models):
        model = end2end(feat_dim=args.feat_dim, in_dim=args.in_dim, out_dim=args.out_dim, pred_dim=args.pred_dim, n_attn=args.n_attn, n_convs=args.n_convs, conv_type=conv_list[i], learned_edges=lrn_bool[i], model_name=mode, geo_edges=geo_edges)
        #model = DoubleAttention(feat_dim=args.feat_dim, in_dim=args.in_dim, out_dim=args.out_dim, pred_dim=args.pred_dim, n_attn=args.n_attn, model_name=mode) 
        trainer = Trainer(args, model, device=args.device)
        trainer.train()