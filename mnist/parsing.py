from array import array as pyarray
import numpy as np
import os
import struct


def load_mnist(dataset="training", digits=np.arange(10), path=".", shape="1d"):
    """
    Adapted from: http://g.sweyla.com/blog/2012/mnist-numpy/
    """
    if dataset == "training":
        fname_img = os.path.join(path, 'train-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels.idx1-ubyte')
    elif dataset == "testing":
        fname_img = os.path.join(path, 't10k-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels.idx1-ubyte')
    else:
        raise ValueError("dataset must be 'testing' or 'training'")

    flbl = open(fname_lbl, 'rb')
    magic_nr, size = struct.unpack(">II", flbl.read(8))
    lbl = pyarray("b", flbl.read())
    flbl.close()

    fimg = open(fname_img, 'rb')
    magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
    img = pyarray("B", fimg.read())
    fimg.close()

    ind = [k for k in range(size) if lbl[k] in digits]
    N = len(ind)

    if shape == "1d":
        images = np.zeros((N, rows * cols), dtype=np.int32)
        labels = np.zeros((N, ), dtype=np.int8)
        for i in range(len(ind)):
            images[i] = np.array(img[ind[i] * rows * cols: (ind[i] + 1) * rows * cols]).reshape((rows * cols))
            labels[i] = lbl[ind[i]]
    elif shape == "2d":
        images = np.zeros((N, rows, cols), dtype=np.int32)
        labels = np.zeros((N, ), dtype=np.int8)
        for i in range(len(ind)):
            images[i] = np.array(img[ind[i] * rows * cols: (ind[i] + 1) * rows * cols]).reshape((rows, cols))
            labels[i] = lbl[ind[i]]
    else:
        raise ValueError("shape must be '1d' or '2d'")

    return images, labels
