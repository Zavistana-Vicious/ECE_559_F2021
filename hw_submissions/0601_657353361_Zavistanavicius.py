#%%
from PIL import Image
import torchvision
import torch
import numpy as np
import matplotlib.pyplot as plt

import os


classes = [
    "Circle",  # 0
    "Square",  # 1
    "Octagon",  # 2
    "Heptagon",  # 3
    "Nonagon",  # 4
    "Star",  # 5
    "Hexagon",  # 6
    "Pentagon",  # 7
    "Triangle",  # 8
]


def normalize(x, files):
    path_in = "data/geometry_dataset_raw/"
    x = np.asarray(x)
    # for special case when
    #   convert to black and white makes the shape disapear
    if np.ptp(x) == 0:
        image = np.asarray(Image.open(path_in + files))
        x = image[:, :, 0]
    if np.ptp(x) == 0:
        x = image[:, :, 1]
    if np.ptp(x) == 0:
        x = image[:, :, 2]
    if np.ptp(x) == 0:
        raise Exception(files)
    x = (x - x.min()) / np.ptp(x)
    # make sure that shape is 1 and background is 0
    if x[0][0] == 1:
        x = 1 - x
    return x


def display_image(nparray):
    plt.imshow(nparray, cmap="Greys")


# does one class at a time so there are no memory problems
def png_to_array(png_prefix):
    image_list = []
    path_in = "data/geometry_dataset_raw/"
    ext = ".png"
    for files in os.listdir(path_in):
        if files.endswith(ext) and files.startswith(png_prefix):
            image = Image.open(path_in + files)
            image = image.convert("L")
            image = normalize(image, files)
            image_list.append(image)
    path_out = "data/geometry_dataset_nparray/"
    np.save(path_out + png_prefix + "s.npy", np.asarray(image_list))
    # resize image code if we need it
    """
    basewidth = 100 # target resolution
    wpercent = (basewidth / float(image.size[0]))
    hsize = int((float(image.size[1]) * float(wpercent)))
    img = image.resize((basewidth, hsize), Image.ANTIALIAS)
    plt.imshow(img)
    """


def test_arrays(file_prefix):
    path = "data/geometry_dataset_nparray/"
    arrays = np.load(path + file_prefix + "s.npy")
    for i in range(100):
        plt.imshow(arrays[i])
        plt.show()


class DataLoader(torch.utils.data.Dataset):
    def __init__(self, image_class, onehot, validation_split=2 / 10):
        path = "data/geometry_dataset_nparray/"

        temp_x = np.load(path + image_class + "s.npy")
        y = [0] * 9
        y[onehot] = 1
        self.y = torch.Tensor(np.array(y))

        indices = list(range(temp_x.shape[0]))
        split = int(np.floor(validation_split * temp_x.shape[0]))

        np.random.seed(0)
        np.random.shuffle(indices)

        train_indices, val_indices = indices[split:], indices[:split]

        train_x = temp_x[train_indices]
        test_x = temp_x[val_indices]

        self.x_train_ten = torch.Tensor(train_x)
        self.x_test_ten = torch.Tensor(test_x)


#%%
# raw data images .png to numpy arrays stored as .npy
for c in classes:
    png_to_array(c)

# %%
dataloader_list = []
for c in range(len(classes)):
    dataloader_list.append(DataLoader(classes[c], c))

# %%
