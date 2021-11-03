#%%
from PIL import Image
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
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


def png_to_npy(
    class_dict,
    pixels,
    test_split=2 / 10,
    path_in="data/geometry_dataset_raw/",
    path_out="data/geometry_dataset_nparray/",
):
    file_suffix = ".png"
    print("Processing all " + file_suffix + " files in " + path_in + " to " + path_out)
    for c in class_dict:
        array_list = []
        for files in os.listdir(path_in):
            if files.startswith(c) and files.endswith(file_suffix):
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
        print("\t" + c + " Done")


class GeometricDataset(Dataset):
    def __init__(
        self, class_dict, t_or_t, preprocess=False, pixels=200, test_split=2 / 10
    ):
        path = "data/geometry_dataset_nparray/"

        if preprocess:
            png_to_npy(class_dict, pixels, test_split)

        a_list = []
        for c in class_dict:
            temp = np.load(path + c + "_" + t_or_t + ".npy")
            a_list.append(temp)

        array = np.concatenate(a_list)

        self.class_length = np.size(array, 0)
        self.x_tensor = torch.Tensor(array)

    def __getitem__(self, index):
        x = self.x_tensor[index]
        onehot = int(math.floor(index * 9 / self.class_length))
        y = [0.0] * 9
        y[onehot] = 1.0
        y = torch.Tensor(np.array(y))
        return (x, onehot)

    def __len__(self):
        return self.class_length


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


def train(model, device, train_loader, optimizer):
    model.train()
    tot_loss = 0
    correct = 0
    for i, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        pred = output.argmax(dim=1, keepdim=True)
        loss = torch.nn.CrossEntropyLoss()(output, target)
        loss.backward()
        optimizer.step()
        correct += pred.eq(target.view_as(pred)).sum().item()

    return correct, tot_loss


def test(model, device, test_loader):
    model.eval()
    tot_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            tot_loss += torch.nn.CrossEntropyLoss()(output, target).item()
            correct += pred.eq(target.view_as(pred)).sum().item()

    return correct, tot_loss


#%%
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

preprocess = False
pixels = 100
train_set = GeometricDataset(class_dict, "train", preprocess, pixels)
test_set = GeometricDataset(class_dict, "test")
train_loader = DataLoader(train_set, batch_size=100, shuffle=True)
test_loader = DataLoader(test_set, batch_size=100)

max_epoch = 20
model_path = "0602_657353361_Zavistanavicius.pt"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# %%
model = Net().to(device)

ta, tl = test(model, device, test_loader)
tra, trl = test(model, device, train_loader)

test_accuracy = [ta / 18000]
test_loss = [tl]
train_accuracy = [tra / 72000]
train_loss = [trl]

optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

best_test_loss = 99999999999
best_epoch = 0
print("\n\nTraining the Model...")
for epoch in range(1, max_epoch + 1):
    train(model, device, train_loader, optimizer)
    print("\t" + str(epoch / max_epoch * 100) + "% complete")
    ta, tl = test(model, device, test_loader)
    tra, trl = test(model, device, train_loader)
    test_accuracy.append(ta / 18000)
    test_loss.append(tl)
    train_accuracy.append(tra / 72000)
    train_loss.append(trl)
    scheduler.step()

    if best_test_loss > tl:
        best_test_loss = tl
        best_epoch = epoch
        best_accuracy = ta / 18000
        torch.save(model, model_path)

#%%
e = list(range(max_epoch + 1))
plt.plot(e, train_loss, label="Training Set Loss")
plt.plot(e, test_loss, label="Testing Set Loss")
plt.scatter([best_epoch], [best_test_loss], label="The Model", c="red")
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss vs Epochs")
plt.show()

plt.plot(e, train_accuracy, label="Training Set Accuracy")
plt.plot(e, test_accuracy, label="Testing Set Accuracy")
plt.scatter([best_epoch], [best_accuracy], label="The Model", c="red")
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Accuracy vs Epochs")
plt.show()

# %%
