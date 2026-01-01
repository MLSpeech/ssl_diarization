import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from vicReg import vicreg_loss_func

def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd
    def forward(self, x):
        return self.lambd(x)


class Barlow_diarization(nn.Module):
    def __init__(self, dim, dim2, pred_steps=1, pred_offset=0):
        super(Barlow_diarization, self).__init__()

        # our backbone (CPC CNN)
        self.cpc_enc = nn.Sequential(
            # nn.Conv1d(1, dim, kernel_size=200, stride=100, padding=0, bias=False),
            # nn.BatchNorm1d(dim),
            # nn.LeakyReLU(),
            nn.Conv1d(1, dim, kernel_size=10, stride=5, padding=0, bias=False),
            nn.BatchNorm1d(dim),
            nn.LeakyReLU(),
            nn.Conv1d(dim, dim, kernel_size=10, stride=5, padding=0, bias=False),
            nn.BatchNorm1d(dim),
            nn.LeakyReLU(),
            nn.Conv1d(dim, dim, kernel_size=10, stride=5, padding=0, bias=False),
            nn.BatchNorm1d(dim),
            nn.LeakyReLU(),
            nn.Conv1d(dim, dim, kernel_size=8, stride=4, padding=0, bias=False),
            nn.BatchNorm1d(dim),
            nn.LeakyReLU(),
            nn.Conv1d(dim, dim, kernel_size=4, stride=2, padding=0, bias=False),
            nn.BatchNorm1d(dim),
            nn.LeakyReLU(),
            nn.Conv1d(dim, dim, kernel_size=4, stride=2, padding=0, bias=False),
            nn.BatchNorm1d(dim),
            nn.LeakyReLU(),
            nn.Conv1d(dim, dim, kernel_size=4, stride=2, padding=0, bias=False),
            LambdaLayer(lambda x: x.transpose(1,2)),
        )        
        
        # self.dive_enc = nn.Sequential(
        # )
        # barlow projector
        # dim2 = 2048
        print("internal dim = ", dim2)
        sizes = [dim, dim2, dim2, dim2]
        # sizes = [dim, 2048, 2048, dim2]
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        self.projector = nn.Sequential(*layers)

        # normalization layer for the representations z1 and z2
        self.bn = nn.BatchNorm1d(sizes[-1], affine=False)
                
        # # similarity estimation projections
        self.pred_steps = list(range(1 + pred_offset, 1 + pred_offset + pred_steps))
        print(f"prediction steps: {self.pred_steps}")

    def project(self, x):
        b, t, d = x.shape
        x = rearrange(x, 'b t d -> (b t) d')
        x = self.projector(x)
        #  x = rearrange(x, '(b t) d -> b t d', b=b, t=t)
        return x
    
    def forward(self, x):
        device = x.device
        b, t = x.shape
        #print(x.shape, x.device)
        # # wav => latent z
        x = rearrange(x, 'b t -> b 1 t')
        z = self.cpc_enc(x)
        # print(z.shape)    
        preds = defaultdict(list)
        for i, t in enumerate(self.pred_steps):  # predict for steps 1...t
            z1 = self.project(z[:, :-t])
            z2 = self.project(z[:, t:])
            
            # print(z1.shape)
        # empirical cross-correlation matrix
        c = self.bn(z1).T @ self.bn(z2)

        # sum the cross-correlation matrix between all gpus
        c.div_(b)
        #  torch.distributed.all_reduce(c)

        # use --scale-loss to multiply the loss by a constant factor
        # see the Issues section of the readme
        scale_loss = float(1 / 32)
        lambd = float(3.9e-3)
        # Positives
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum().mul(scale_loss)
        off_diag = off_diagonal(c).pow_(2).sum().mul(scale_loss)
        loss = on_diag + lambd * off_diag
        # Negatives
        # off_diag = off_diagonal(c).add_(-1).pow_(2).sum().mul(scale_loss)
        # on_diag = torch.diagonal(c).pow_(2).sum().mul(scale_loss)
        # loss = off_diag + lambd * on_diag
        return loss

    def forward_vicReg(self, x):
        device = x.device
        b, t = x.shape
        #print(x.shape, x.device)
        # # wav => latent z
        x = rearrange(x, 'b t -> b 1 t')
        z = self.cpc_enc(x)
        # print(z.shape)    
        preds = defaultdict(list)
        for i, t in enumerate(self.pred_steps):  # predict for steps 1...t
            z1 = self.project(z[:, :-t])
            z2 = self.project(z[:, t:])
        
        loss = vicreg_loss_func(z1, z2)

        return loss

    def embed(self, x):
        device = x.device
        b, t = x.shape
        x = rearrange(x, 'b t -> b 1 t')
        z = self.cpc_enc(x)
        z1 = self.project(z)
    
        # print(x.shape)
        # print(z.shape)
        # print(z1.shape)
        return z, z1

def window(a, w=12800, o=6400):
    leftover = a.size % w
    sh = (a.size - w +1, w)
    st = a.strides * 2
    # print(a.size)
    # print(w)
    # print(leftover)
    # print(sh)
    # print(st)
    view = np.lib.stride_tricks.as_strided(a, strides = st, shape = sh)[0::o]
    padding = np.array([a[-1-w:-1]])
    view = np.append(view, padding, axis=0)
    return torch.from_numpy(view.flatten())
        
if __name__ == "__main__":
    model = Barlow_diarization(128, 1024, 1, 0)
    # x = torch.rand(1, 160000)
    x = np.random.rand(1, 8000)
    print(x.shape)
    b, t = x.shape
    x = torch.from_numpy(rearrange(x, 'b t -> b 1 t')).float()
    print(x.shape)
    z = model.cpc_enc(x)
    # x1 = rearrange(window(x.numpy().squeeze(), 8000, 4000), 't -> 1 t')
    # 
    # x2 = torch.from_numpy(np.pad(x1.numpy(), ((0,0), (16000,0))))
    # out = model.forward(x)
    print(z.shape)
    # model.eval()
    # out = model.embed(x)
    # out1 = model.embed(x1)
    # out2 = model.embed(x2)