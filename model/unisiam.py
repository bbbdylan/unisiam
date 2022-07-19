import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class UniSiam(nn.Module):
    def __init__(self, encoder, lamb=0.1, temp=2.0, dim_hidden=None, dist=False, dim_out=2048):
        super(UniSiam, self).__init__()
        self.encoder = encoder
        self.encoder.fc = None

        dim_in = encoder.out_dim
        dim_hidden = dim_in if dim_hidden is None else dim_hidden

        self.proj = nn.Sequential(
            nn.Linear(dim_in, dim_hidden),
            nn.BatchNorm1d(dim_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(dim_hidden, dim_hidden),
            nn.BatchNorm1d(dim_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(dim_hidden, dim_hidden),
            nn.BatchNorm1d(dim_hidden),
        )
        self.pred = nn.Sequential(
                nn.Linear(dim_hidden, dim_hidden//4),
                nn.BatchNorm1d(dim_hidden//4),
                nn.ReLU(inplace=True),
                nn.Linear(dim_hidden//4, dim_hidden)
            )

        if dist:
            self.pred_dist = nn.Sequential(
                nn.Linear(dim_in, dim_out),
                nn.BatchNorm1d(dim_out),
                nn.ReLU(inplace=True),
                nn.Linear(dim_out, dim_out),
                nn.BatchNorm1d(dim_out),
                nn.ReLU(inplace=True),
                nn.Linear(dim_out, dim_out),
                nn.BatchNorm1d(dim_out),
                nn.ReLU(inplace=True),
                nn.Linear(dim_out, dim_out//4),
                nn.BatchNorm1d(dim_out//4),
                nn.ReLU(inplace=True),
                nn.Linear(dim_out//4, dim_out)
                )

        self.lamb = lamb
        self.temp = temp

    def forward(self, x, z_dist=None):
        bsz = x.shape[0]//2

        f = self.encoder(x)
        z = self.proj(f)
        p = self.pred(z)
        z1, z2 = torch.split(z, [bsz, bsz], dim=0)
        p1, p2 = torch.split(p, [bsz, bsz], dim=0)

        loss_pos = (self.pos(p1, z2)+self.pos(p2,z1))/2
        loss_neg = self.neg(z)
        loss = loss_pos + self.lamb * loss_neg

        if z_dist is not None:
            p_dist = self.pred_dist(f)
            loss_dist = self.pos(p_dist, z_dist)
            loss = 0.5 * loss + 0.5 * loss_dist

        std = self.std(z)

        return loss, loss_pos, loss_neg, std

    @torch.no_grad()
    def std(self, z):
        return torch.std(F.normalize(z, dim=1), dim=0).mean()

    def pos(self, p, z):
        z = z.detach()
        z = F.normalize(z, dim=1)
        p = F.normalize(p, dim=1)
        return -(p*z).sum(dim=1).mean()

    def neg(self, z):
        batch_size = z.shape[0] //2
        n_neg = z.shape[0] - 2
        z = F.normalize(z, dim=-1)
        mask = 1-torch.eye(batch_size, dtype=z.dtype, device=z.device).repeat(2,2)
        out = torch.matmul(z, z.T) * mask
        return (out.div(self.temp).exp().sum(1)-2).div(n_neg).mean().log()