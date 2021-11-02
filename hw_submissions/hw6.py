#%%
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
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

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 1, 5, 3)
        self.conv2 = nn.Conv2d(1, 1, 3, 3)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(484, 128)
        self.fc2 = nn.Linear(128, 9)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        #x = F.max_pool2d(x, 1)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        return x

    def accuracy(self, x_val, y_val):
        self.eval()
        correct = 0
        actual = y_val.numpy().argmax()
        predictions = self.forward(x_val)
        for i in range(len(x_val)):
            predict = predictions[i].detach().numpy().argmax()
            if predict == actual:
                correct = correct + 1

        accuracy = correct / len(x_val)
        return accuracy

    def train_model(self, data_list, epoch_limit=1000, device = 'cuda:0'):
        self.train()
        criterion = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=0.3)

        for epoch in range(epoch_limit):
            for data in data_list:
                optimizer.zero_grad()
                outputs = self.forward(data.x_train_ten.unsqueeze(1).to(device))
                y = data.y.tolist()
                y = torch.Tensor(np.asarray([y]*8000)).to(device)
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()


#%%
# raw data images .png to numpy arrays stored as .npy
for c in classes:
    png_to_array(c)

# %%
# get dataloader list containing necessary tensors within the classes
dataloader_list = []
for c in range(len(classes)):
    dataloader_list.append(DataLoader(classes[c], c))

# %%
device = "cuda:0"
model = CNN().to(device)
data = dataloader_list[0].x_test_ten.unsqueeze(1)

#%%
import time
model.to(device)
t0 = time.time()
model.train_model(dataloader_list, 100)
print(time.time() - t0)
# %%
for i in range(9):
    data = dataloader_list[i].x_test_ten.unsqueeze(1)
    model.to('cpu')
    print(model.accuracy(data, dataloader_list[i].y))
# %%
