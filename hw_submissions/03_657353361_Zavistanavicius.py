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
        # x = x.flatten()
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
            predict = model.forward(data, True).item()
            target = target.item()
            if target != predict:
                num_incor += 1

        return num_incor

    def update_w(self, data, target, lr):
        error = target - self.forward(data)
        self.linear = self.linear + lr * error * data

        # to do NEXT: figure out how to get a sigle value for error


# training set size
n = 50
# learning rate
lr = 1


batch_size = None

training_data = datasets.MNIST(
    root="data", train=True, download=False, transform=transforms.ToTensor()
)
validation_data = datasets.MNIST(
    root="data", train=False, download=False, transform=transforms.ToTensor()
)

training_subset = torch.utils.data.Subset(training_data, list(range(n)))

t_data = torch.utils.data.DataLoader(training_subset, shuffle=True)
v_data = torch.utils.data.DataLoader(validation_data, shuffle=False)


# pixels = training_data[0][0][0]
# plt.imshow(pixels)
# plt.show()

model = MultiPrcptrn()
data, target = 0, 0
for data, target in t_data:
    data = data
    target = target
    break
print(model(data))
print(model.forward(data, True).item())
model.update_w(data, target, 1)
print()
print(target.item())
print()
print(model.misclass(v_data))
print(model.misclass(t_data))

# %%
