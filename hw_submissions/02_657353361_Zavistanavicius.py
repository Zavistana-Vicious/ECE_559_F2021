#%%
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.function_base import linspace


def step_func(x):
    if x >= 0:
        y = 1
    else:
        y = 0
    return y


def output(input_set, parameters, n):
    output_set = []
    for i in range(n):
        y = step_func(sum(input_set[i] * parameters))
        output_set.append(y)
    output_set = np.array(output_set)
    return output_set


def misclass(actual, prediction, n):
    incorrect = 0
    for i in range(n):
        if actual[i] != prediction[i]:
            incorrect += 1
    return incorrect


seed = 0

# a, b, c
np.random.seed(seed)
w0 = np.random.uniform(-0.25, 0.25)
w1 = np.random.uniform(-1, 1)
w2 = np.random.uniform(-1, 1)
w = np.array([w0, w1, w2])

print("a: w0 = " + str(w0) + "\nb: w1 = " + str(w1) + "\nc: w2 = " + str(w2))

# d
np.random.seed(seed)
n = 100
Sx = []
for i in range(n):
    x1 = np.random.uniform(-1, 1)
    x2 = np.random.uniform(-1, 1)
    Sx.append([1, x1, x2])
Sx = np.array(Sx)

# e, f
Sy = output(Sx, w, n)

# g
fig = plt.figure(figsize=(5, 5))
x = np.linspace(-1, 1, 1000)
y = (w[1] * x + w[0]) / (-1 * w[2])
plt.plot(x, y, label="separation line")
plt.title("g")
plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.xlabel("x1")
plt.ylabel("x2")

index_1 = np.where(Sy == 1)
index_0 = np.where(Sy == 0)

S1_x = Sx[index_1, 1]
S1_y = Sx[index_1, 2]
S0_x = Sx[index_0, 1]
S0_y = Sx[index_0, 2]

plt.scatter(S1_x, S1_y, label="S1")
plt.scatter(S0_x, S0_y, label="S0")
plt.legend()
plt.show()

# h
#   i
lr = 1

#   ii
np.random.seed(seed)
_w0 = np.random.uniform(-1, 1)
_w1 = np.random.uniform(-1, 1)
_w2 = np.random.uniform(-1, 1)
_w = np.array([_w0, _w1, _w2])

print("h ii: w' = [w0', w1', w2'] = " + str(_w))

#   iii
_Sy = output(Sx, _w, n)
incorrect = misclass(Sy, _Sy, n)
print("h iii: number of misclassifications = " + str(incorrect))

#   iv, v, vi
incorr_array = [incorrect]
epochs = 0
while incorrect > 0:
    epochs += 1
    for i in range(n):
        if _Sy[i] == 1 and Sy[i] == 0:
            _w = _w - (lr * Sx[i])

        if _Sy[i] == 0 and Sy[i] == 1:
            _w = _w + (lr * Sx[i])

    _Sy = output(Sx, _w, n)
    incorrect = misclass(Sy, _Sy, n)
    incorr_array.append(incorrect)

#   vii
print("h vii: actual weights = " + str(w) + " vs optimal weights = " + str(_w))
print(
    "h vii (normalized): actual weights = "
    + str(w / max(w))
    + " vs optimal weights = "
    + str(_w / max(_w))
)

# i
x = list(range(0, len(incorr_array)))
y = incorr_array
plt.plot(x, y)
plt.title("i")
plt.xlabel("epochs")
plt.ylabel("misclassifications")
plt.show

# j through m done by rerunning code with different parameters for "lr" and "n"
#   w are kept the same by reseting the RNG seed when generating w

# %%
