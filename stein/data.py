"""View samplers with ground-truth content and nuisance.

3DShapes factor order: floor_hue(10), wall_hue(10), object_hue(10), scale(8), shape(4), orientation(15).
Content s  = (object_hue, scale, shape); nuisance n = (floor_hue, wall_hue, orientation).
Probe coding: content -> [cos h, sin h, scale, simplex3(shape)]  (k = 6)
              nuisance -> [cos fh, sin fh, cos wh, sin wh, orientation]  (5 dims)
"""
import math
import numpy as np
import torch

PROC = "data/processed"
SIZES = np.array([10, 10, 10, 8, 4, 15])
FH, WH, OH, SC, SH, OR = range(6)
CONTENT, NUISANCE = (OH, SC, SH), (FH, WH, OR)
K_CONTENT = 6
SIMPLEX4 = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], dtype=np.float32) / math.sqrt(3)


def law_weights(n, law, spread=2.5):
    """Discretised law over n ordered grid values; grid spans +-spread standard deviations."""
    u = (np.arange(n) - (n - 1) / 2) / ((n - 1) / 2) * spread
    if law == "uniform":
        w = np.ones(n)
    elif law == "gaussian":
        w = np.exp(-0.5 * u ** 2)
    elif law == "laplace":
        w = np.exp(-np.abs(u) * math.sqrt(2))
    else:
        raise ValueError(law)
    return w / w.sum()


def content_code(f):
    f = np.asarray(f)
    h = 2 * np.pi * f[:, OH] / 10
    sc = (f[:, SC] - 3.5) / 2.2913
    return np.concatenate([np.cos(h)[:, None], np.sin(h)[:, None], sc[:, None], SIMPLEX4[f[:, SH]]], 1).astype(np.float32)


def nuisance_code(f):
    f = np.asarray(f)
    a = 2 * np.pi * f[:, FH] / 10
    b = 2 * np.pi * f[:, WH] / 10
    o = (f[:, OR] - 7) / 4.32
    return np.stack([np.cos(a), np.sin(a), np.cos(b), np.sin(b), o], 1).astype(np.float32)


class PreadImages:
    """Random-access reader for a uint8 .npy image array via os.pread. All workers share the OS
    page cache (one copy of the file in RAM). Random gathers through np.memmap cost ~75 ms per
    512 images on macOS (page faults); pread costs ~1 ms."""

    def __init__(self, fn):
        mm = np.load(fn, mmap_mode="r")
        self.shape, self.offset = mm.shape, mm.offset
        self.item = int(np.prod(mm.shape[1:]))
        del mm
        self.fn, self.fd = fn, None

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, idx):
        import os
        if self.fd is None:
            self.fd = os.open(self.fn, os.O_RDONLY)
        idx = np.atleast_1d(np.asarray(idx, dtype=np.int64))
        sz = self.item
        buf = bytearray(len(idx) * sz)
        mv = memoryview(buf)
        for j, i in enumerate(idx):
            mv[j * sz:(j + 1) * sz] = os.pread(self.fd, sz, self.offset + int(i) * sz)
        return np.frombuffer(buf, np.uint8).reshape(len(idx), *self.shape[1:])


_MEMMAP = {}


def shapes_memmap():
    if "s" not in _MEMMAP:
        _MEMMAP["s"] = PreadImages(f"{PROC}/shapes32_images.npy")
    return _MEMMAP["s"]


class Shapes:
    """3DShapes pair / clip sampler from the memory-mapped 32x32 array."""

    def __init__(self, law="uniform", seed=0, test_frac=0.1, noise=0.02):
        self.images = shapes_memmap()
        self.law, self.noise = law, noise
        self.rng = np.random.default_rng(seed)
        split_rng = np.random.default_rng(12345)          # fixed split for all runs
        self.is_test = split_rng.random(int(SIZES.prod())) < test_frac
        self.w = {OH: law_weights(10, law), SC: law_weights(8, law), SH: np.ones(4) / 4}

    def _content(self, n):
        return {j: self.rng.choice(SIZES[j], size=n, p=self.w[j]) for j in CONTENT}

    def _idx(self, f):
        return np.ravel_multi_index(tuple(f[:, j] for j in range(6)), SIZES)

    def sample_factors(self, n, split="train"):
        """Single images with content from the law and uniform nuisance."""
        out = np.zeros((0, 6), dtype=np.int64)
        while len(out) < n:
            m = 2 * (n - len(out)) + 16
            f = np.zeros((m, 6), dtype=np.int64)
            for j, v in self._content(m).items():
                f[:, j] = v
            for j in NUISANCE:
                f[:, j] = self.rng.integers(0, SIZES[j], m)
            keep = self.is_test[self._idx(f)] == (split == "test")
            out = np.concatenate([out, f[keep]])
        return out[:n]

    def sample_pair_factors(self, n, alpha):
        """View 1 ~ law x uniform nuisance; view 2 shares content and resamples each nuisance
        factor independently with prob alpha. Both views avoid the test split."""
        f1s, f2s = [], []
        got = 0
        while got < n:
            m = 2 * (n - got) + 16
            f1 = np.zeros((m, 6), dtype=np.int64)
            for j, v in self._content(m).items():
                f1[:, j] = v
            for j in NUISANCE:
                f1[:, j] = self.rng.integers(0, SIZES[j], m)
            f2 = f1.copy()
            for j in NUISANCE:
                r = self.rng.random(m) < alpha
                f2[r, j] = self.rng.integers(0, SIZES[j], r.sum())
            keep = ~self.is_test[self._idx(f1)] & ~self.is_test[self._idx(f2)]
            f1s.append(f1[keep]); f2s.append(f2[keep]); got += keep.sum()
        return np.concatenate(f1s)[:n], np.concatenate(f2s)[:n]

    def sample_clip_factors(self, n, T, alpha):
        """Clips of T frames: content fixed; each nuisance factor takes a +-1 step per frame
        with prob alpha (hues wrap around the circle, orientation reflects at the ends)."""
        out = []
        got = 0
        while got < n:
            m = 2 * (n - got) + 16
            f = np.zeros((m, T, 6), dtype=np.int64)
            for j, v in self._content(m).items():
                f[:, :, j] = v[:, None]
            for j in NUISANCE:
                f[:, 0, j] = self.rng.integers(0, SIZES[j], m)
            for t in range(1, T):
                for j in NUISANCE:
                    step = self.rng.choice([-1, 1], m) * (self.rng.random(m) < alpha)
                    v = f[:, t - 1, j] + step
                    if j == OR:
                        v = np.where(v < 0, 1, np.where(v > 14, 13, v))
                    else:
                        v = v % SIZES[j]
                    f[:, t, j] = v
            keep = ~self.is_test[self._idx(f.reshape(-1, 6))].reshape(m, T).any(1)
            out.append(f[keep]); got += keep.sum()
        return np.concatenate(out)[:n]

    def images_for(self, f, device="cpu", noise=True):
        idx = self._idx(f.reshape(-1, 6))
        x = torch.from_numpy(self.images[idx].copy()).to(device).permute(0, 3, 1, 2).float().div_(255.0)
        if noise and self.noise > 0:
            x = x + self.noise * torch.randn_like(x)
        x = x - 0.5
        return x.reshape(*f.shape[:-1], 3, 32, 32)


# ----------------------------------------------------------------------------- CIFAR-10

AUG_NAMES = ["cx", "cy", "log_scale", "log_aspect", "flip", "brightness", "contrast", "saturation"]


class Cifar:
    """Two independent augmentations per image; augmentation parameters are the nuisance.
    alpha scales augmentation strength (alpha=0: identity up to pixel noise)."""

    def __init__(self, seed=0, noise=0.02):
        self.xtr = np.load(f"{PROC}/cifar_train_x.npy", mmap_mode="r")
        self.ytr = np.load(f"{PROC}/cifar_train_y.npy")
        self.xte = np.load(f"{PROC}/cifar_test_x.npy", mmap_mode="r")
        self.yte = np.load(f"{PROC}/cifar_test_y.npy")
        self.rng = np.random.default_rng(seed)
        self.noise = noise

    def raw(self, idx, split="train", device="cpu"):
        X = self.xtr if split == "train" else self.xte
        o = np.argsort(idx)
        x = np.empty((len(idx), 32, 32, 3), dtype=np.uint8)
        x[o] = X[np.asarray(idx)[o]]
        return torch.from_numpy(x).to(device).permute(0, 3, 1, 2).float().div_(255.0)

    def aug_params(self, n, alpha):
        r = self.rng
        smin = 1 - 0.8 * alpha
        log_scale = np.log(r.uniform(smin, 1.0, n))
        log_aspect = r.uniform(-np.log(4 / 3), np.log(4 / 3), n) * alpha
        s, a = np.exp(log_scale), np.exp(log_aspect)
        w = np.minimum(np.sqrt(s * a), 1.0); h = np.minimum(np.sqrt(s / a), 1.0)
        cx = r.uniform(-1, 1, n) * (1 - w); cy = r.uniform(-1, 1, n) * (1 - h)
        flip = (r.random(n) < 0.5 * alpha).astype(np.float64)
        j = 0.4 * alpha
        br, ct, sat = (r.uniform(1 - j, 1 + j, n) for _ in range(3))
        return np.stack([cx, cy, log_scale, log_aspect, flip, br, ct, sat], 1).astype(np.float32), w, h

    def augment(self, x, alpha, device="cpu"):
        n = x.shape[0]
        p, w, h = self.aug_params(n, alpha)
        P = torch.from_numpy(p).to(device)
        theta = torch.zeros(n, 2, 3, device=device)
        sx = torch.from_numpy(w.astype(np.float32)).to(device) * (1 - 2 * P[:, 4])
        theta[:, 0, 0] = sx; theta[:, 0, 2] = P[:, 0]
        theta[:, 1, 1] = torch.from_numpy(h.astype(np.float32)).to(device); theta[:, 1, 2] = P[:, 1]
        grid = torch.nn.functional.affine_grid(theta, x.shape, align_corners=False)
        y = torch.nn.functional.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)
        b, c, s = P[:, 5, None, None, None], P[:, 6, None, None, None], P[:, 7, None, None, None]
        y = y * b
        gray = (0.299 * y[:, 0] + 0.587 * y[:, 1] + 0.114 * y[:, 2]).unsqueeze(1)
        y = (y - gray.mean((2, 3), keepdim=True)) * c + gray.mean((2, 3), keepdim=True)
        gray = (0.299 * y[:, 0] + 0.587 * y[:, 1] + 0.114 * y[:, 2]).unsqueeze(1)
        y = gray + (y - gray) * s
        y = y.clamp(0, 1)
        if self.noise > 0:
            y = y + self.noise * torch.randn_like(y)
        return y - 0.5, p

    def sample_pair(self, n, alpha, device="cpu"):
        idx = self.rng.integers(0, len(self.ytr), n)
        x = self.raw(idx, "train", device)
        x1, p1 = self.augment(x, alpha, device)
        x2, p2 = self.augment(x, alpha, device)
        return x1, x2, p1, p2, self.ytr[idx]
