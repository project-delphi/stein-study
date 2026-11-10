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


class LinearEncoder(nn.Module):
    """Study 2: one linear map from the flattened input (pixels, or the synthetic (s, n)) to R^d."""

    def __init__(self, d_in, d):
        super().__init__()
        self.lin = nn.Linear(d_in, d)

    def forward(self, x):
        return self.lin(x.flatten(1))


class LinearPredictor(nn.Linear):
    """Study 2: the theory's linear predictor W, initialised at I + 0.1 N(0, 1/d) (as in
    theory/check_t1.py), bias 0."""

    def __init__(self, d_in, d_out):
        super().__init__(d_in, d_out)
        with torch.no_grad():
            self.weight.copy_(torch.eye(d_out, d_in) + 0.1 * torch.randn(d_out, d_in) / d_in ** 0.5)
            self.bias.zero_()


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
