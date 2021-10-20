#%%
import numpy as np
import matplotlib.pyplot as plt


def sech_squared(x):
    # sech = 1 / np.cosh(x)
    # return sech * sech
    return 1 - x ** 2


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
        # self.dEdw0 = self.dEdw1 * (1 / np.cosh(self.v1) / np.cosh(self.v1)) * self.x
        self.dEdw0 = diff * sech_squared(self.y) * self.w1 * x
        # self.dEdwb0 = self.dEdw1 * (1 / np.cosh(self.v1)) * (1 / np.cosh(self.v1))
        self.dEdwb0 = diff * sech_squared(self.y) * self.w1

    def update_w(self, lr):
        self.w0 = self.w0 + lr * self.dEdw0
        self.wb0 = self.w0 + lr * self.dEdwb0
        self.w1 = self.w1 + lr * self.dEdw1
        self.wb1 = self.w1 + lr * self.dEdwb1

    def total_MSE(self, n, x, d):
        temp = 0
        for i in range(n):
            temp += (d[i] - self.forward(x[i])) * (d[i] - self.forward(x[i]))
        return temp / n


n = 300
x = np.array([np.random.uniform(0, 1) for i in range(n)])
v = np.array([np.random.uniform(-0.1, 0.1) for i in range(n)])
d = np.sin(20 * x) + 3 * x + v

nn = NeuralNet()
y = np.array([nn.forward(x[i]) for i in range(n)])

#%%
loss = [nn.total_MSE(n, x, d)]
lr = 0.000003
for i in range(1000):
    temp = nn.total_MSE(n, x, d)
    loss.append(temp)
    for j in range(n):
        nn.forward(x[j])
        nn.backward(d[j], x[j])
        nn.update_w(lr)

y_ = np.array([nn.forward(x[i]) for i in range(n)])

plt.scatter(x, d)
plt.scatter(x, y)
plt.scatter(x, y_)
plt.show()

plt.plot(list(range(1001)), loss)
plt.show()

#%%
