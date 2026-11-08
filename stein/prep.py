"""Preprocess raw datasets once into memory-mapped uint8 .npy files under data/processed/.

3DShapes: 480000x64x64x3 -> 2x2 mean-pooled 32x32 uint8 (1.47 GB) + factor indices (int8).
CIFAR-10: train/test uint8 NHWC + labels.
MNIST: train/test uint8 + labels (for reference; temporal tier uses 3DShapes).
Run: .venv/bin/python -m stein.prep
"""
import gzip, os, pickle, tarfile, io
import numpy as np
import h5py

RAW = "data/raw"
OUT = "data/processed"
SHAPES_SIZES = [10, 10, 10, 8, 4, 15]  # floor_hue, wall_hue, object_hue, scale, shape, orientation


def prep_3dshapes():
    img_out = f"{OUT}/shapes32_images.npy"
    lab_out = f"{OUT}/shapes_factors.npy"
    if os.path.exists(img_out) and os.path.exists(lab_out):
        print("3dshapes: exists"); return
    f = h5py.File(f"{RAW}/3dshapes.h5", "r")
    n = f["images"].shape[0]
    labels = f["labels"][:]
    # factor values -> integer indices
    idx = np.zeros((n, 6), dtype=np.int8)
    for j in range(6):
        vals = np.unique(labels[:, j])
        assert len(vals) == SHAPES_SIZES[j], (j, len(vals))
        idx[:, j] = np.searchsorted(vals, labels[:, j])
    # verify row-major grid order: row index == ravel_multi_index(factors)
    flat = np.ravel_multi_index(idx.T.astype(np.int64), SHAPES_SIZES)
    assert np.array_equal(flat, np.arange(n)), "3dshapes rows are not in grid order"
    mm = np.lib.format.open_memmap(img_out + ".tmp", mode="w+", dtype=np.uint8, shape=(n, 32, 32, 3))
    step = 15000
    for s in range(0, n, step):
        x = f["images"][s:s + step].astype(np.float32)
        x = x.reshape(-1, 32, 2, 32, 2, 3).mean(axis=(2, 4))
        mm[s:s + step] = np.clip(np.rint(x), 0, 255).astype(np.uint8)
        print(f"3dshapes {s + step}/{n}", flush=True)
    mm.flush(); del mm
    os.replace(img_out + ".tmp", img_out)
    np.save(lab_out, idx)


def prep_cifar():
    if os.path.exists(f"{OUT}/cifar_train_x.npy"):
        print("cifar: exists"); return
    tf = tarfile.open(f"{RAW}/cifar-10-python.tar.gz", "r:gz")
    def load(name):
        d = pickle.load(tf.extractfile(f"cifar-10-batches-py/{name}"), encoding="bytes")
        x = d[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        return x, np.array(d[b"labels"], dtype=np.int64)
    xs, ys = zip(*[load(f"data_batch_{i}") for i in range(1, 6)])
    np.save(f"{OUT}/cifar_train_x.npy", np.concatenate(xs).astype(np.uint8))
    np.save(f"{OUT}/cifar_train_y.npy", np.concatenate(ys))
    x, y = load("test_batch")
    np.save(f"{OUT}/cifar_test_x.npy", x.astype(np.uint8)); np.save(f"{OUT}/cifar_test_y.npy", y)


def prep_mnist():
    if os.path.exists(f"{OUT}/mnist_train_x.npy"):
        print("mnist: exists"); return
    for split, pre in [("train", "train"), ("test", "t10k")]:
        with gzip.open(f"{RAW}/mnist/{pre}-images-idx3-ubyte.gz") as g:
            x = np.frombuffer(g.read(), np.uint8, offset=16).reshape(-1, 28, 28)
        with gzip.open(f"{RAW}/mnist/{pre}-labels-idx1-ubyte.gz") as g:
            y = np.frombuffer(g.read(), np.uint8, offset=8).astype(np.int64)
        np.save(f"{OUT}/mnist_{split}_x.npy", x); np.save(f"{OUT}/mnist_{split}_y.npy", y)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    prep_cifar(); prep_mnist(); prep_3dshapes()
    print("done")
