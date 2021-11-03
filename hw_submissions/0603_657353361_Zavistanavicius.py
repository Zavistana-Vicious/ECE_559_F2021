#%%
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

import os


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


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Sequential(
            nn.Conv2d(1, 8, 3, 1),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 8, 5, 1),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.MaxPool2d(2),
        )

        self.conv2 = torch.nn.Sequential(
            nn.Conv2d(8, 8, 3, 1),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 8, 5, 1),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.MaxPool2d(2),
        )

        self.full_c = torch.nn.Sequential(
            nn.Linear(3200, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 9),
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = torch.flatten(x, 1)
        x = self.full_c(x)
        return x


display_images = True
pixels = 100

class_list = [
    "Circle",
    "Square",
    "Octagon",
    "Heptagon",
    "Nonagon",
    "Star",
    "Hexagon",
    "Pentagon",
    "Triangle",
]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = torch.load("0602_657353361_Zavistanavicius.pt").to(device)
model.eval()

path_in = ""
file_suffix = ".png"
for files in os.listdir():
    if files.endswith(file_suffix):
        image = Image.open(path_in + files)
        array = image_to_array(image, files, pixels)
        tensor = torch.Tensor(array).to(device)
        output = model(tensor.unsqueeze(0))
        pred = output.argmax(dim=1, keepdim=True).item()
        print(files + ": " + class_list[pred])

        if display_images:
            plt.imshow(image)
            plt.show()

# %%
