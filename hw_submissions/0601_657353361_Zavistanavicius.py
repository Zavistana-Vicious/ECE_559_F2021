from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import numpy as np
import matplotlib.pyplot as plt

import os


def b_and_w_norm(image, files):
    x = np.asarray(image)
    if np.ptp(x) == 0:
        x = image[:, :, 0]
    if np.ptp(x) == 0:
        x = image[:, :, 1]
    if np.ptp(x) == 0:
        x = image[:, :, 2]
    if np.ptp(x) == 0:
        raise Exception(files)
    x = (x - x.min()) / np.ptp(x)

    if x[0][0] == 1:
        x = 1 - x
    return x


def image_to_array(image, files, pixels=200):
    wperc = pixels / float(image.size[0])
    hsize = int((float(image.size[1]) * float(wperc)))
    image = image.resize((pixels, hsize), Image.ANTIALIAS)
    array = b_and_w_norm(image, files)
    return array


def png_to_npy(class_dict, path_in="data/geometry_dataset_raw/", pixels=200):
    for c in class_dict:
        array_list = []
        for files in os.listdir(path_in):
            if files.startswith(c):
                image = Image.open(path_in + files)
                array = image_to_array(image, files, pixels)
                array_list.append(array)
        path_out = "data/geometry_dataset_nparray/"
        np.save(path_out + c + ".npy", np.asarray(array_list))


class_dict = {
    "Circle": 0,
    "Square": 1,
    "Octagon": 2,
    "Heptagon": 3,
    "Nonagon": 4,
    "Star": 5,
    "Hexagon": 6,
    "Pentagon": 7,
    "Triangle": 8,
}
