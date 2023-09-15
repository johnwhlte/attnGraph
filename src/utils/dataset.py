import os
from torch_geometric.loader import DataLoader
from torch.utils.data import Dataset
from torch_geometric.data import Data
import torch
import random
import numpy as np
import math



class flowDataGenerator:

    def __init__(self, path, seed, num_snaps, funcs_list, alpha_vals, split_data=0.8):
        self.seed = seed
        self.path=path
        self.split_data = split_data
        self.num_snaps = num_snaps
        self.funcs_list = funcs_list
        self.alpha_vals = alpha_vals

    def segment_data_paths(self):

        snaps = range(0, self.num_snaps-1)
        train_cases_x = []
        val_cases_x = []
        train_cases_y = []
        val_cases_y = []

        for alpha in self.alpha_vals:
            for func in self.funcs_list:
                for snap in snaps:
                    if snap <= self.split_data*self.num_snaps:
                        train_cases_x.append(f'{self.path}/{func}_{snap}_{alpha}_snap.pt')
                        train_cases_y.append(f'{self.path}/{func}_{snap+1}_{alpha}_snap.pt')

                    else:
                        val_cases_x.append(f'{self.path}/{func}_{snap}_{alpha}_snap.pt')
                        val_cases_y.append(f'{self.path}/{func}_{snap+1}_{alpha}_snap.pt')

        random.seed(self.seed)

        zipped_train = list(zip(train_cases_x, train_cases_y))
        zipped_val = list(zip(val_cases_x, val_cases_y))
        random.shuffle(zipped_train)
        random.shuffle(zipped_val)
        train_x, train_y = zip(*zipped_train)
        val_x, val_y = zip(*zipped_val)
        return train_x, train_y, val_x, val_y

class flowDataSet(Dataset):
    """ Dataset object with specifications for this network """
    def __init__(self, feats_paths, label_paths, transform=None, test=False):

        self.feats_paths = feats_paths
        self.transform = transform
        self.test=test
        self.label_paths = label_paths

    def __len__(self):
        return len(self.feats_paths)

    def __getitem__(self,idx):
        """ Loads in the data """
        X = torch.load(self.feats_paths[idx])
        Y = torch.load(self.label_paths[idx]) 
        geo_edges = torch.load('/home/sysiphus/bigData/snapshots/edges.pt')

        if self.transform != None:
            X = self.transform[0](X)
            Y = self.transform[1](X,Y)

        data = Data(x=X, edge_index=geo_edges, y=Y) # create a torch_geometric Data object from the data loaded in this dataset

        return data  

def x_transform(x):

    return (x - torch.mean(x)) / torch.std(x)
def y_transform(x,y):

    return (y - torch.mean(x)) / torch.std(x)