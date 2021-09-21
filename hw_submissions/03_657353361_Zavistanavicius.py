import torch
import torchvision
from torch.utils.data import Dataset
from torchvision import datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np


batch_size = None

training_data = datasets.MNIST(
    root="data", train=True, download=False, transform=transforms.ToTensor()
)
validation_data = datasets.MNIST(
    root="data", train=False, download=False, transform=transforms.ToTensor()
)

t_data = torch.utils.data.DataLoader(training_data, shuffle=True)
v_data = torch.utils.data.DataLoader(validation_data, shuffle=False)
