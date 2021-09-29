#%%
import numpy as np
import matplotlib.pyplot as plt


def check_domain(w):
    w = list(w)
    print(w)
    in_domain = True
    if (w[0] + w[1]) >= 1:
        in_domain = False
    if w[0] < 0 or w[1] < 0:
        in_domain = False

    return in_domain


def f(x, y):
    z = -1 * np.log(1 - x - y) - np.log(x) - np.log(y)
    return z


def fx(w):
    z = 1 / ((1 - w[0] - w[1])) - 1 / (w[0])
    return z


def fy(w):
    z = 1 / ((1 - w[0] - w[1])) - 1 / (w[1])
    return z


def gradient(w):
    out_0 = fx(w)
    out_1 = fy(w)
    return [out_0, out_1]


def fxx(w):
    z = 1 / ((1 - w[0] - w[1]) * (1 - w[0] - w[1])) - 1 / (w[0] * w[0])
    return z


def fxy(w):
    z = 1 / ((1 - w[0] - w[1]) * (1 - w[0] - w[1]))
    return z


def fyy(w):
    z = 1 / ((1 - w[0] - w[1]) * (1 - w[0] - w[1])) - 1 / (w[1] * w[1])
    return z


def hessian(w):
    out_00 = fxx(w)
    out_01 = fxy(w)
    out_11 = fyy(w)
    return np.array([[out_00, out_01], [out_01, out_11]])


def plot_descent(func, path, title=""):
    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 1, 100)

    X, Y = np.meshgrid(x, y)
    Z = f(X, Y)

    fig = plt.figure()
    ax = plt.axes(projection="3d")
    ax.plot_wireframe(X, Y, Z)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    path = np.array(path)
    x = path[:, 0]
    y = path[:, 1]
    z = f(x, y)
    c = x + y
    ax.scatter(x, y, z, c=c, cmap="inferno")
    plt.show()


xy = np.random.uniform(0, 1)
frac = np.random.uniform(0, 1)
x = xy * frac
y = xy * (1 - frac)

lr = 0.003
w0 = [x, y]
diff = 0.01

print("w0 = " + str(w0) + "\tlearning rate = " + str(lr))

#%%
# Gradient Descent
list_w = [w0]
while True:
    w = list_w[-1]
    temp_0 = w[0] - lr * gradient(w)[0]
    temp_1 = w[1] - lr * gradient(w)[1]
    list_w.append([temp_0, temp_1])

    if not check_domain(list_w[-1]):
        print("w outside of domain")
        break

    if gradient(list_w[-1]) == gradient(list_w[-2]):
        print("converged")
        break

print("epochs = " + str(len(list_w)))
plot_descent(f, list_w, "gradient descent trajectory")

# %%
# Newton's Method
w0 = [0.4, 0.4]
list_w = [w0]
lr = 0.0001
print("w0 = " + str(w0) + "\tlearning rate = " + str(lr))
while True:
    w = np.array(list_w[-1])
    hess = hessian(w)
    w = w - lr * np.matmul(hess, gradient(w))
    list_w.append(list(w))

    if not check_domain(list_w[-1]):
        print("w outside of domain")
        break

    if gradient(list_w[-1]) == gradient(list_w[-2]):
        print("converged")
        break

print("epochs = " + str(len(list_w)))
plot_descent(f, list_w, "gradient descent trajectory")

# %%
# Least Squares Fit


def Ex(w, x, y):
    total = 0
    for i in x:
        total += 2 * (w[0] + w[1] * i - y[i - 1])
    return total


def Ey(w, x, y):
    total = 0
    for i in x:
        total += -2 * (-w[0] - w[1] * i + y[i - 1])
    return total


def gradient_(w, x, y):
    out_0 = Ex(w, x, y)
    out_1 = Ey(w, x, y)
    return [out_0, out_1]


x = list(range(1, 51))
y = []
for i in x:
    y.append(i + np.random.uniform(-1, 1))

ave_x = np.average(x)
ave_y = np.average(y)

numer = sum((x + ave_x) * (y + ave_y))
denom = sum((x + ave_x) * (x + ave_x))
b = numer / denom

a = ave_y - b * ave_x

y_ = []
for i in x:
    y_.append(a + b * i)

list_w = [[0, 0]]
epoch = 0
while True:
    epoch += 1
    w = list_w[-1]
    temp_0 = w[0] - lr * gradient_(w, x, y)[0]
    temp_1 = w[1] - lr * gradient_(w, x, y)[1]
    list_w.append([temp_0, temp_1])

    if gradient_(list_w[-1], x, y) == gradient_(list_w[-2], x, y):
        print("converged")
        break

    if epoch == 1000:
        break


w = list_w[-1]
_y_ = []
for i in x:
    _y_.append(w[0] + w[1] * i)

plt.scatter(x, y, label="linear points with noise", c="black")
label = "least squares fit: y = " + str(a) + "+ " + str(b) + "x"
plt.plot(x, y_, label=label, c="red")
label = "least squares fit: y = " + str(w[0]) + "+ " + str(w[1]) + "x"
plt.plot(x, _y_, label=label, c="blue")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.title("least squares fit and gradient descent")

# %%
