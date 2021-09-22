#%%
import torch
import torchvision
from torch.utils.data import Dataset
from torchvision import datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np


class MultiPrcptrn(torch.nn.Module):
    def __init__(self):
        super(MultiPrcptrn, self).__init__()
        self.linear = torch.nn.Linear(28 * 28, 10)

    def forward(self, x, misclass=False):
        x = x.view(-1, 28 * 28)
        x = self.linear(x)
        y = torch.heaviside(x, torch.tensor([1.0]))

        if misclass:
            if torch.count_nonzero(y) != torch.tensor([1.0]):
                y = torch.argmax(x)
            else:
                y = torch.argmax(y)

        return y

    def misclass(self, dataload):
        self.eval()
        num_incor = 0
        for data, target in dataload:
            predict = self.forward(data, True).item()
            target = target.item()
            if target != predict:
                num_incor += 1

        return num_incor

    def update_w(self, data, target, lr):
        error = target - self.forward(data)
        error = torch.transpose(error, 0, 1)
        data = data.view(-1, 28 * 28)
        self.linear.weight.data = self.linear.weight.data + lr * torch.matmul(
            error, data
        )

    def an_epoch(self, dataload, lr):
        self.train()
        for data, target in dataload:
            self.update_w(data, target, lr)


training_data = datasets.MNIST(
    root="data", train=True, download=False, transform=transforms.ToTensor()
)
validation_data = datasets.MNIST(
    root="data", train=False, download=False, transform=transforms.ToTensor()
)
v_data = torch.utils.data.DataLoader(validation_data, shuffle=False)

# %%
# f, g, h
n_list = [50]
lr = 1
epsilon = 0

for n in n_list:
    model = MultiPrcptrn()

    # change size of training set here
    training_subset = torch.utils.data.Subset(training_data, list(range(n)))
    t_data = torch.utils.data.DataLoader(training_subset, shuffle=True)

    epochs = 0
    misclass_list = [model.misclass(t_data)]
    while misclass_list[-1] > epsilon:
        epochs += 1
        model.an_epoch(t_data, lr)
        wrong = model.misclass(t_data)
        print(wrong)
        misclass_list.append(wrong)

    print(epochs)
    print(misclass_list)
    print(model.misclass(v_data))

# %%
