import torch
import numpy as np
import matplotlib.pyplot as plt
from utils.dataset import flowDataGenerator, flowDataSet
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
from utils.ops import contResLoss, mape_error
import time


class Trainer:


    def __init__(self, args, model, device):

        self.path = args.data_path
        self.batch_size = args.batch_size
        self.num_epochs = args.num_epochs
        self.seed = args.seed
        self.num_snaps = args.num_snaps
        self.funcs_list = args.funcs_list
        self.alpha_vals = args.alpha_vals
        self.device = device
        self.model = model
        self.name = model.model_name
        self.model.to(self.device)
        self.lr = args.lr
        self.early_stop = args.early_stop
        self.TRAIN_RUN=f'{self.name}'
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.custom_loss = contResLoss()

    def load_data(self):

        train_x, train_y, val_x, val_y = flowDataGenerator(path=self.path, seed=self.seed, num_snaps=self.num_snaps, funcs_list=self.funcs_list, alpha_vals=self.alpha_vals, split_data=0.8).segment_data_paths()
        
        train_ds = flowDataSet(train_x, train_y)
        val_ds = flowDataSet(val_x, val_y)

        self.train_loader = DataLoader(train_ds, batch_size=self.batch_size)
        self.val_loader = DataLoader(val_ds, batch_size=self.batch_size)

        return 0

    def train_step(self, batch, batch_num, loader, epoch):

        self.optim.zero_grad()
        out = self.model(batch)
        # mse_loss = F.mse_loss(out, batch.y.T[2:].T)
        mape_loss = mape_error(out, batch.y.T[2:].T)
        mape_loss.backward()
        # (mse_loss).backward()
        self.optim.step()
        # train_loss = mse_loss.item()
        train_loss = mape_loss.item()

        print ("\033[A                             \033[A")
        print(f"Epoch:{epoch} T | {int(100*batch_num/len(loader))} % | MAPE Loss = {train_loss:.4f}")#  | Cont Loss = {residual:.10f}")

        return train_loss

    @torch.no_grad()
    def val_step(self, batch, batch_num, loader, epoch):

        out = self.model(batch)
        # loss = F.mse_loss(out, batch.y.T[2:].T)
        # val_loss = loss.item()
        loss = mape_error(out, batch.y.T[2:].T)
        val_loss = loss.item()

        print ("\033[A                             \033[A")
        print(f"Epoch:{epoch} V | {int(100*batch_num/len(loader))} % | MSE Loss = {val_loss:.4f}  ")

        return val_loss

    def train(self):

        self.train_loss = []
        self.val_loss = []
        epoch_plot = []
        avg_val_loss = float('inf')
        count = 0
        self.load_data()
        #self.model.load_state_dict(torch.load('models/test1_func4_ckpt'))

        for epoch in range(self.num_epochs):

            val_loss_list = []
            train_loss_list = []
            print("\n")

            self.model.train()
            for i, batch in enumerate(self.train_loader):
                batch = batch.to(self.device)
                train_loss = self.train_step(batch=batch, batch_num=i, loader=self.train_loader, epoch=epoch)

                train_loss_list.append(train_loss)

            print("\n")
            self.model.eval()
            for i, batch in enumerate(self.val_loader):
                batch = batch.to(self.device)
                val_loss = self.val_step(batch=batch, batch_num=i, loader=self.val_loader, epoch=epoch)
            
                val_loss_list.append(val_loss)

            
            avg_val_loss_new = sum(val_loss_list) / len(val_loss_list)
            avg_train_loss = sum(train_loss_list) / len(train_loss_list)

            if avg_val_loss_new >= avg_val_loss:
                count+=1
                avg_val_loss = avg_val_loss_new
            else:
                count = 0
                avg_val_loss = avg_val_loss_new

            if count >= self.early_stop:
                break

            self.train_loss.append(avg_train_loss)
            self.val_loss.append(avg_val_loss)
            epoch_plot.append(epoch)

            torch.save(self.model.state_dict(), f'models/{self.TRAIN_RUN}_ckpt')

        torch.save(self.model.state_dict(), f'models/{self.TRAIN_RUN}')


        fig = plt.figure()
        plt.plot(epoch_plot,self.train_loss, label="training mse loss")
        plt.plot(epoch_plot, self.val_loss, label="validation mse loss")
        #plt.plot([0], [0], label=f"Loss: {self.validation_loss[-1]:.2f} + N: {len(epoch_plot)}")
        plt.legend()
        plt.savefig(f'figs/{self.TRAIN_RUN}_loss.png')

        return 0



