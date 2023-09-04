import torch
import numpy as np
import matplotlib.pyplot as plt
from utils.dataset import flowDataGenerator, flowDataSet
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
from utils.ops import contResLoss
from net import end2end
import time
import argparse
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, SAGEConv, GATv2Conv
from tqdm import tqdm

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


args = get_args()


def mape_error(x, target):

    mape_err = torch.mean(torch.abs(torch.div((x - target),(target+0.00001))))

    return mape_err.item()

def calculate_bcs(t):

    t_true = t + 3*0.01

    boundary_condns = [(torch.tensor(range(760,780)),(0.02*np.sin(3*t_true)+0.01,0,0)),(torch.tensor(range(780,840)),(0.0001,0.0001,0))] 
    #boundary_condns = [(torch.tensor(range(760,780)),(0.05,0,0)),(torch.tensor(range(780,840)),(0,0,0))] 

    return boundary_condns
@torch.no_grad()
def run_time_step(batch):

    out = MODEL(batch)

    mse_loss = F.mse_loss(out, batch.y.T[2:].T)
    mape_err = F.mse_loss(out, batch.y.T[2:].T).item()
    #residual = CONT_LOSS(V=out, edge_feats=batch.deltas)
    #mape_err = mape_error(x=out, target=batch.y.T[2:].T)

    #print(f'MSE Loss: {mse_loss}')
    #print(f'Residual Loss: {residual}')

    return mse_loss, mape_err, out


def add_boundary_conditions(boundary_table, x):

    for boundary in boundary_table:

        x.T[0][boundary[0]] = boundary[1][0]

    return x


def write_to_of_file(x, filename, filepath):

    return 0
model_list = ['lrnd_gcn', 'lrnd_sage', 'lrnd_gat', 'geo_gcn', 'geo_sage', 'geo_gat']
lrnd_bool = [True, True, True, False, False, False]
conv_list = [GCNConv, SAGEConv, GATv2Conv, GCNConv, SAGEConv, GATv2Conv]
geo_edges = torch.load('/home/sysiphus/bigData/snapshots/edges.pt').to(args.device)
if __name__ == "__main__":
    fig = plt.figure()
    for i, mod in enumerate(model_list):
        MODEL = end2end(feat_dim=args.feat_dim, in_dim=args.in_dim, out_dim=args.out_dim, pred_dim=args.pred_dim, n_attn=args.n_attn, n_convs=args.n_convs, conv_type=conv_list[i], learned_edges=lrnd_bool[i], model_name=mod, geo_edges=geo_edges)
        MODEL.load_state_dict(torch.load(f'models/{mod}'))
        MODEL.to(args.device)
        CONT_LOSS = contResLoss()
        PATH = '/home/sysiphus/bigData/snapshots'
        PATH_VAR=800
        START_DATA = Data(x=torch.load(f'{PATH}/func4_{PATH_VAR}_0.05_snap.pt'), y=torch.load(f'{PATH}/func4_{PATH_VAR+1}_0.05_snap.pt')).to(args.device)
        BOUNDARIES = [(torch.tensor(range(760,780)),),(torch.tensor(range(780,840)),0)]
        mse_loss_plot = []
        residual_loss_plot = []
        mape_plot = []
        num_time_steps = 998

        for time_step in tqdm(range(800,num_time_steps)):

            msel, mape, xnew = run_time_step(START_DATA)
            mape_plot.append(mape)
            mse_loss_plot.append(msel.item())
            x = np.reshape(xnew.T[0].T, (20,20))
            y = np.reshape(xnew.T[1].T, (20,20))
            vx = np.reshape(xnew.T[2].T, (20,20))
            vy = np.reshape(xnew.T[3].T, (20,20))
            mag_v = np.reshape(torch.sqrt((torch.square(xnew.T[2].T) + torch.square(xnew.T[3].T))), (20,20))


            fig, ax = plt.subplots()
            c = ax.pcolormesh(x,y,vx, cmap='RdBu')#, vmin=-z.max(), vmax=z.max())
            ax.set_title('pcolormesh')
            ax.axis([x.min(), x.max(), y.min(), y.max()])
            fig.colorbar(c, ax=ax)
            plt.savefig(f'../{mod}/vx_{time_step}.png')
            fig, ax = plt.subplots()
            c = ax.pcolormesh(x,y,vy, cmap='RdBu')#, vmin=-z.max(), vmax=z.max())
            ax.set_title('pcolormesh')
            ax.axis([x.min(), x.max(), y.min(), y.max()])
            fig.colorbar(c, ax=ax)
            plt.savefig(f'../{mod}/vy_{time_step}.png')
            fig, ax = plt.subplots()
            c = ax.pcolormesh(x,y,mag_v, cmap='RdBu')#, vmin=-z.max(), vmax=z.max())
            ax.set_title('pcolormesh')
            ax.axis([x.min(), x.max(), y.min(), y.max()])
            fig.colorbar(c, ax=ax)
            plt.savefig(f'../{mod}/mag_v_{time_step}.png')


            xnew2 = torch.cat((START_DATA.x.T[0:2].T,xnew), dim=-1)
            PATH_VAR += 1
            START_DATA = Data(x=xnew2, y=torch.load(f'{PATH}/func4_{PATH_VAR+1}_0.05_snap.pt')).to(args.device)


        plt.plot(np.linspace(330,494,len(mape_plot)), mape_plot, label=f"{mod[0:13]}")

#plt.yscale('log')
#plt.ylim([1,10**1])
#plt.xlim([0,1])
plt.legend()
#plt.xlim([0,5])
plt.savefig('figs/testvalall.png')
