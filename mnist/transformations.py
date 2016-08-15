import math
import numpy as np
from skimage.feature import hog


def rotate90(data):
    a2 = data.shape[0]
    a = int(math.sqrt(a2))
    image = data.reshape(a, a)
    image = np.rot90(image)
    data_rotated = image.reshape(a2)
    return data_rotated


def shift3(data):
    a2 = data.shape[0]
    a = int(math.sqrt(a2))
    image = data.reshape(a, a)
    image = np.roll(image, 3, axis=1)
    data_shifted = image.reshape(a2)
    return data_shifted


def reconstruct_dataset(dataset, transformation):
    transformed = []
    for data in dataset[0]:
        transformed.append(transformation(data))
    return np.array(transformed), dataset[1]


def extract_hog(data):
    a2 = data.shape[0]
    a = int(math.sqrt(a2))
    image = data.reshape(a, a)
    fd = hog(image, pixels_per_cell=(4, 4), cells_per_block=(1, 1))
    return fd
