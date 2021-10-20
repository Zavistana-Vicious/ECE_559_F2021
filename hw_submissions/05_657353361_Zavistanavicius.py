#%%
import numpy as np
import matplotlib.pyplot as plt


def sech_squared(x):
    sech = 1 / np.cosh(x)
    return sech * sech


def MSE(n, d, z):
    return np.sum((d - z) * (d - z)) / n


class NeuralNet:
    def __init__(self, N=24):
        # input weights
        self.w0 = np.random.randn(N)
        # input bias
        self.wb0 = np.random.randn(N)
        # output weight
        self.w1 = np.random.randn(N)
        # output bias
        self.wb1 = np.random.randn(1)

    def forward(self, x):
        v0 = np.dot(self.w0, x) + self.wb0
        self.y = np.tanh(v0)
        v1 = np.dot(self.w1, self.y) + self.wb1
        self.z = v1
        return self.z

    def backward(self, d, x):
        diff = d - self.z
        self.dEdw1 = diff * self.y
        self.dEdwb1 = diff
        self.dEdw0 = diff * sech_squared(self.y) * self.w1 * x
        self.dEdwb0 = diff * sech_squared(self.y) * self.w1

    def update_w(self, lr):
        self.w1 = self.w1 + lr * self.dEdw1
        self.wb1 = self.wb1 + lr * self.dEdwb1
        self.w0 = self.w0 + lr * self.dEdw0
        self.wb0 = self.wb0 + lr * self.dEdwb0


n = 300
x = np.array([np.random.uniform(0, 1) for i in range(n)])
v = np.array([np.random.uniform(-0.1, 0.1) for i in range(n)])
d = np.sin(20 * x) + 3 * x + v

nn = NeuralNet()
z = np.array([nn.forward(x[i]) for i in range(n)])

#%%
loss = [MSE(n, d, z)]
lr = 0.003
for i in range(11000):
    for j in range(n):
        nn.forward(x[j])
        nn.backward(d[j], x[j])
        nn.update_w(lr)

    z_ = np.array([nn.forward(x[i]) for i in range(n)])
    loss.append(MSE(n, d, z_))

plt.scatter(x, d)
plt.scatter(x, z)
plt.scatter(x, z_)
plt.show()

plt.plot(list(range(11001)), loss)
plt.show()
