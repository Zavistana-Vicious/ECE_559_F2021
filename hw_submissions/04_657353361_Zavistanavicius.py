#%%
import numpy as np
import matplotlib.pyplot as plt

def check_domain(w):
    in_domain = True
    if (w[0] + w[1]) >= 1:
        in_domain = False
    if w[0] < 0 or w[1] < 0:
        in_domain = False

    return in_domain

def f(x,y):
    z = -1 * np.log(1 - x - y) - np.log(x) - np.log(y)
    return z

def fx(w):
    z = 1 / ((1-w[0] - w[1])) - 1 / (w[0])
    return z

def fy(w):
    z = 1 / ((1-w[0] - w[1])) - 1 / (w[1])
    return z

def gradient(w):
    out_0 = fx(w)
    out_1 = fy(w)
    return [out_0, out_1]

def fxx(w):
    z = 1 / ((1-w[0] - w[1]) * (1-w[0] - w[1])) - 1 / (w[0] * w[0])
    return z

def fxy(w):
    z = 1 / ((1-w[0] - w[1]) * (1-w[0] - w[1]))
    return z

def fyy(w):
    z = 1 / ((1-w[0] - w[1]) * (1-w[0] - w[1])) - 1 / (w[1] * w[1])
    return z

def hessian(w):
    out_00 = fxx(w)
    out_01 = fxy(w)
    out_11 = fyy(w)
    return np.array([[out_00, out_01], [out_01, out_11]])

def plot_descent(func, path, title =''):
    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 1, 100)
    
    X, Y = np.meshgrid(x, y)
    Z = f(X, Y)
    
    fig = plt.figure()
    ax = plt.axes(projection ='3d')
    ax.plot_wireframe(X, Y, Z)
    ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    path = np.array(path)
    x = path[:, 0]
    y = path[:, 1]
    z = f(x, y)
    c=x+y
    ax.scatter(x, y, z, c=c, cmap="inferno")
    plt.show()


xy = np.random.uniform(0,1)
frac = np.random.uniform(0,1)
x = xy * frac
y = xy * (1-frac)

lr = .003
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
plot_descent(f, list_w, 'gradient descent trajectory')

# %%
# Newton's Method
# https://www.stat.cmu.edu/~ryantibs/convexopt-S15/scribes/14-newton-scribed.pdf
list_w = [w0]
while True:
    w = np.array(list_w[-1])
    hess = hessian(w)
    w = w - lr * np.linalg.inv(hess) * gradient(w)
    list_w.append(w)

    if not check_domain(list_w[-1]):
        print("w outside of domain")
        break

    if gradient(list_w[-1]) == gradient(list_w[-2]):
        print("converged")
        break

print("epochs = " + str(len(list_w)))
plot_descent(f, list_w, 'gradient descent trajectory')