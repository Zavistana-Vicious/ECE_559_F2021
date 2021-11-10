#%%
from torch.utils import data
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import numpy as np
import matplotlib.pyplot as plt

import os
import math
import re


# %%
def onehot_to_letter(onehot):
    if onehot > 26 or onehot < 0:
        raise Exception("invalid onehot input")
    if onehot == 0:
        letter = "EON"
    else:
        letter = chr(onehot + 96)
    return letter


def letter_to_onehot(letter):
    if letter == "EON":
        onehot = 0
    else:
        onehot = ord(letter) - 96
    if onehot > 26 or onehot < 0:
        raise Exception("invalid letter input")
    return onehot


def convert_to(a_list, to):
    for i in range(len(a_list)):
        if to == "onehot":
            a_list[i] = letter_to_onehot(a_list[i])
        if to == "letter":
            a_list[i] = onehot_to_letter(a_list[i])

    return a_list


class NamesDataset(Dataset):
    def __init__(self, target_len=11):
        _file = open("data/names/names.txt", "r")
        lines = []
        for line in _file:
            lines.append(line)

        x = []
        y = []

        for name in lines:
            temp = name.lower()
            temp = re.sub(r"\W+", "", temp)
            temp = list(temp) + ["EON"] * (target_len - len(temp))
            x.append(temp)
            y.append(temp[1:] + ["EON"])

        for i in range(len(x)):
            x[i] = convert_to(x[i], "onehot")
            y[i] = convert_to(y[i], "onehot")

        x_array = np.array(x)
        y_array = np.array(y)

        self.class_length = np.size(x_array, 0)
        self.x_tensor = torch.Tensor(x_array)
        self.y_tensor = torch.Tensor(y_array)
        self.target_len = target_len

    def __getitem__(self, index):
        x = self.x_tensor[index]
        y = self.y_tensor[index]
        return (x, y)

    def __len__(self):
        return self.class_length


#%%
class LSTM(nn.Module):
    def __init__(self, input_size=27, hidden_size=128, num_layers=2, output_size=27):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden, cell):
        x = torch.nn.functional.one_hot(x.to(torch.int64), self.input_size)
        x, (hidden, cell) = self.lstm(
            x.unsqueeze(0).unsqueeze(0).float(), (hidden, cell)
        )
        x = self.fc(x.reshape(x.shape[0], -1))
        return x, (hidden, cell)

    def init_hidden(self, batch_size=1):
        hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        cell = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        return hidden, cell


def train(model, device, data_loader):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    model.train()
    tot_loss = 0
    for i, (data, target) in enumerate(data_loader):
        data, target = data[0].to(device), target[0].to(device)
        hidden, cell = model.init_hidden()
        hidden = hidden.to(device)
        cell = cell.to(device)
        optimizer.zero_grad()

        loss = 0
        for c in range(11):
            output, (hidden, cell) = model(data[c].to(device), hidden, cell)
            loss += criterion(
                output.to(device),
                target[c].unsqueeze(0).type(torch.LongTensor).to(device),
            )

        loss.backward()
        optimizer.step()
        # loss = loss.item() / model.target_len
        tot_loss += loss

    return tot_loss


#%%
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
data_set = NamesDataset()
data_loader = DataLoader(data_set, batch_size=1, shuffle=True)
model = LSTM().to(device)

# %%
epochs = 50
loss_list = []
best_test_loss = 9999999999999999999
model_path = "0702_657353361_Zavistanavicius.pt"
for i in range(epochs):
    print(i / epochs)
    loss_list.append(train(model, device, data_loader))

    if best_test_loss > loss_list[-1]:
        best_test_loss = loss_list[-1]
        best_epoch = i
        torch.save(model, model_path)

e = list(range(epochs))
plt.plot(e, loss_list)
plt.scatter([best_epoch], [best_test_loss], label="The Model", c="red")
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss vs Epochs")
plt.show()

# %%
def generate(model, initial_str="a"):
    name = initial_str
    hidden, cell = model.init_hidden(1)
    initial_input = letter_to_onehot(initial_str)

    for i in range(11):
        inp = torch.Tensor([initial_input])

        output, (hidden, cell) = model(
            inp[0].to(device), hidden.to(device), cell.to(device)
        )

        output = output.cpu().detach().numpy()[0]
        top_3_ind = list(np.argpartition(output, -3)[-3:])
        choice = np.random.choice(top_3_ind)

        if choice == 0:
            break
        else:
            name = name + onehot_to_letter(choice)
            initial_input = choice

    return name
