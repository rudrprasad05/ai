import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

n_per_class = 50

class_pos = np.random.randn(n_per_class, 2) + np.array([2, 2])
class_neg = np.random.randn(n_per_class, 2) + np.array([-2, -2])

X = np.vstack((class_pos, class_neg)) # format -> [n1(x1, x2), n2(x1, x2), ...nx(x1, x2)]
y = np.hstack((np.ones(n_per_class), -np.ones(n_per_class))) # format -> [-1, +1, ..., +-1]

def sign(x):
    if x >= 0:
        return 1
    else:
        return -1

def train_perceptron(X, y, max_epochs=100):

    w = np.zeros(X.shape[1])
    b = 0

    for epoch in range(max_epochs):

        errors = 0

        for i, x in enumerate(X):

            z = np.dot(w, x) + b # w1x1 + w2x2 + b
            y_pred = sign(z)
            y_true = y[i]

            if y_pred != y_true:
                w = w + y_true * x
                b = b + y_true
                errors += 1

        print(f"epoch {epoch+1} errors: {errors}")

        if errors == 0:
            print("converged")
            break

    return w, b


w, b = train_perceptron(X, y)

print("final w:", w)
print("final b:", b)

plt.figure(figsize=(6,6))

plt.scatter(class_pos[:,0], class_pos[:,1], label="Class +1")
plt.scatter(class_neg[:,0], class_neg[:,1], label="Class -1")

x_vals = np.linspace(X[:,0].min()-1, X[:,0].max()+1, 100)

y_vals = -(w[0]*x_vals + b) / w[1]

plt.plot(x_vals, y_vals, label="Decision Boundary")

plt.xlabel("x1")
plt.ylabel("x2")
plt.legend()
plt.grid(True)

plt.show()