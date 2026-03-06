import numpy as np
import matplotlib.pyplot as plt

# for repeatable results
np.random.seed(42)

# fabricate two clusters
n_per_class = 50

class_pos = np.random.randn(n_per_class, 2) + np.array([1, 1])   # label +1
class_neg = np.random.randn(n_per_class, 2) + np.array([-1, -1]) # label -1

# combine into one dataset
X = np.vstack((class_pos, class_neg))
y = np.hstack((np.ones(n_per_class), -np.ones(n_per_class)))

w = [0, 0]
b = 0


for index, x in enumerate(X):
    print(f"node: {x[0]},{x[1]} ")
    z = np.dot(w, x) + b
    y_pred = 0
    y_true = y[index]

    if z < 0:
        y_pred = -1
    else:
        y_pred = 1;

    if(y_pred == y_true):
        print("pred: correct")
        continue;
    else:
        w= w + y_true*x
        b = b + y_true
        print(f"pred: wrong. updated w={w} b={b}")
