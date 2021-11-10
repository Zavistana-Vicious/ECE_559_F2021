#%%
from torch.utils import data
import torch.nn as nn
import torch
import numpy as np

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
        hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        cell = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        return hidden, cell


def generate(model, initial_str="a"):
    name = initial_str
    hidden, cell = model.init_hidden(1)
    initial_input = letter_to_onehot(initial_str)

    for i in range(11):
        inp = torch.Tensor([initial_input])

        output, (hidden, cell) = model(inp[0], hidden, cell)

        output = output.cpu().detach().numpy()[0]
        top_3_ind = list(np.argpartition(output, -3)[-3:])
        choice = np.random.choice(top_3_ind)

        if choice == 0:
            break
        else:
            name = name + onehot_to_letter(choice)
            initial_input = choice

    return name


#%%
model = torch.load("0602_657353361_Zavistanavicius.pt")
for onehot in range(1, 27):
    letter = letter_to_onehot(onehot)
    for i in range(2):
        print(generate(model, letter))
