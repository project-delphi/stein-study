"""Encoder, predictor, decoder (VAE tier). One backbone for every arm."""
import copy
import torch
import torch.nn as nn


class Encoder(nn.Module):
    """32x32x3 -> d. Conv(ch) x3 stride 2 -> MLP. No normalisation layers, so collapse is visible."""

    def __init__(self, d, ch=64, hidden=256, out_mult=1):
        super().__init__()
        c1 = ch // 2
        self.conv = nn.Sequential(
            nn.Conv2d(3, c1, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(c1, ch, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(ch, ch, 4, 2, 1), nn.ReLU(),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(ch * 16, hidden), nn.ReLU(), nn.Linear(hidden, d * out_mult))

    def forward(self, x):
        return self.head(self.conv(x))


class Predictor(nn.Module):
    def __init__(self, d_in, d_out, hidden=256, norm="none"):
        super().__init__()
        mid = [nn.BatchNorm1d(hidden)] if norm == "bn" else []
        self.net = nn.Sequential(nn.Linear(d_in, hidden), *mid, nn.ReLU(), nn.Linear(hidden, d_out))

    def forward(self, z):
        return self.net(z)


class Decoder(nn.Module):
    def __init__(self, d, ch=64, hidden=256):
        super().__init__()
        c1 = ch // 2
        self.ch = ch
        self.fc = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Linear(hidden, ch * 16), nn.ReLU())
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(ch, ch, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(ch, c1, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(c1, 3, 4, 2, 1),
        )

    def forward(self, z):
        return self.deconv(self.fc(z).view(-1, self.ch, 4, 4))


def make_ema(model):
    m = copy.deepcopy(model)
    for p in m.parameters():
        p.requires_grad_(False)
    return m


@torch.no_grad()
def ema_update(target, online, tau):
    for pt, po in zip(target.parameters(), online.parameters()):
        pt.mul_(tau).add_(po.detach(), alpha=1 - tau)
