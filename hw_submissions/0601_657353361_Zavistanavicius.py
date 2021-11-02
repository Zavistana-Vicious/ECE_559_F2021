#%%
from PIL import Image
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import numpy as np
import matplotlib.pyplot as plt

import os
import math


def b_and_w_norm(image, files):
    a = np.asarray(image)
    x = a[:, :, 0]
    if np.ptp(x) == 0:
        x = a[:, :, 1]
    if np.ptp(x) == 0:
        x = a[:, :, 2]
    if np.ptp(x) == 0:
        raise Exception(files)

    x = (x - x.min()) / np.ptp(x)
    if x[0][0] == 1:
        x = 1 - x
    return x


def image_to_array(image, files, pixels):
    wperc = pixels / float(image.size[0])
    hsize = int((float(image.size[1]) * float(wperc)))
    image = image.resize((pixels, hsize), Image.ANTIALIAS)
    array = b_and_w_norm(image, files)
    return array


def png_to_npy(class_dict, pixels=200, test_split=2 / 10):
    path_in = "data/geometry_dataset_raw/"
    path_out = "data/geometry_dataset_nparray/"
    for c in class_dict:
        array_list = []
        for files in os.listdir(path_in):
            if files.startswith(c):
                image = Image.open(path_in + files)
                array = image_to_array(image, files, pixels)
                array_list.append(array)

        array_len = len(array_list)
        indices = list(range(array_len))
        split = int(np.floor(test_split * array_len))

        train_indices, val_indices = indices[split:], indices[:split]

        a_array = np.asarray(array_list)
        train_a = a_array[train_indices]
        test_a = a_array[val_indices]

        np.save(path_out + c + "_train.npy", train_a)
        np.save(path_out + c + "_test.npy", test_a)
        print(c + " Done")


class GeometricDataset(Dataset):
    def __init__(
        self, class_dict, t_or_t, preprocess=False, pixels=200, test_split=2 / 10
    ):
        path = "data/geometry_dataset_nparray/"
        a_list = []

        for c in class_dict:
            temp = np.load(path + c + "_" + t_or_t + ".npy")
            a_list.append(temp)

        array = np.concatenate(a_list)

        self.class_length = np.size(array, 0)
        self.x_tensor = torch.Tensor(array)

    def __getitem__(self, index):
        x = self.x_tensor[index]
        onehot = math.floor(index * 9 / self.class_length)
        y = [0] * 9
        y[onehot] = 1
        y = torch.Tensor(np.array(y))
        return (x, y)

    def __len__(self):
        return self.class_length


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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

#%%
train_set = GeometricDataset(class_dict, "train")
test_set = GeometricDataset(class_dict, "test")

train_loader = DataLoader(train_set, batch_size=10, shuffle=True)
test_loader = DataLoader(test_set, batch_size=10)


# %%
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv = torch.nn.Sequential(nn.Conv2d(1, 5, 5, 3, 0), nn.ReLU())
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.25)
        self.fc1 = torch.nn.Sequential(nn.Linear(5445, 128), nn.ReLU())
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = torch.nn.Sequential(nn.Linear(128, 64), nn.ReLU())
        self.output = torch.nn.Sequential(nn.Linear(64, 9), nn.Softmax())

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout1(x)
        x = self.fc1(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        x = self.output(x)

        return x


# %%
model = CNN()
for batch_idx, (data, target) in enumerate(train_loader):
    output = model(data)
    print(output, target)
    break

print(torch.nn.BCELoss()(output, target))
# %%
