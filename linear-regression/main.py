import numpy as np
import matplotlib.pyplot as plt
import re
from pathlib import Path


def sign(values):
    """
    Convert values to class labels {-1, +1}.
    If value is 0, treat it as +1.
    """
    return np.where(values >= 0, 1, -1)


def generate_regression_data(n=80, noise_std=0.8, seed=42):
    """
    Generate 2D data for regression.

    create x values in [0, 5], and y values from a noisy line:
        y = 1.2 + 0.9x + noise

    Returns:
        x  -> shape (N,)
        y  -> shape (N,)
    """
    rng = np.random.default_rng(seed)
    x = np.sort(rng.uniform(0, 5, n))
    noise = rng.normal(0, noise_std, n)
    y = 1.2 + 0.9 * x + noise
    return x, y


def run_linear_regression_demo():
    """
    Generate 2D data for regression.

    create x values in [0, 5], and y values from a noisy line:
        y = 1.2 + 0.9x + noise

    Returns:
        x  -> shape (N,)
        y  -> shape (N,)
    """
    seed = 42
    noise = 0.8
    n = 80

    rng = np.random.default_rng(seed)
    noise = rng.normal(0, noise, n)

    x = np.sort(rng.uniform(0, 5, n))
    y = 1.2 + 0.9 * x + noise

    """
    For linear regression, X should include a bias term. Bais is just x0
    Since data is 1D (x only), X becomes [1, x]
    """
    X = np.c_[np.ones(len(x)), x]

    """
    Compute w using pseudoinverse
    @ used for matrix multiplication
    w = x_dagger * y
    """
    w = np.linalg.pinv(X) @ y

    # predicted y values
    y_pred = X @ w

    # fidn error
    E_in = np.mean((y - y_pred) ** 2)

    print("=" * 60)
    print("PART A: LINEAR REGRESSION")
    print("=" * 60)
    print(f"Learned weights: w0 = {w[0]:.4f}, w1 = {w[1]:.4f}")
    print(f"In-sample error (E_in / MSE) = {E_in:.6f}")

    # plot data
    plt.figure(figsize=(8, 5))
    plt.scatter(x, y, label="Data points")
    plt.plot(x, y_pred, label="Best-fit line")
    plt.title("Linear Regression: Data and Best-Fit Line")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    plt.show()

    return w, E_in


def generate_classification_clusters(n1=60, n2=60, seed=7, loc1=[2,2], loc2=[6,6]):
    """
    two separate clusters for binary classification.

    cluster 1 gets label +1
    cluster 2 gets label -1
    """
    rng = np.random.default_rng(seed)

    # cluster +1 around (2, 2)
    class_pos = rng.normal(loc=loc1, scale=[0.9, 0.9], size=(n1, 2))

    # cluster -1 around (6, 6)
    class_neg = rng.normal(loc=loc2, scale=[1.0, 1.0], size=(n2, 2))

    X = np.vstack([class_pos, class_neg])
    y = np.hstack([np.ones(n1), -np.ones(n2)])

    return X, y


def classification_error(y_true, y_pred):
    """
    Fraction of misclassified points.
    eg  y_true = [ 1, -1, 1, -1, 1 ]
        y_pred = [ 1,  1, 1, -1, -1 ]
        y_true != y_pred => [False, True, False, False, True]
        np.mean averages it by considering false as 0 and true as 1
        [0,1,0,0,1] sum / 5 = 0.4
    """
    return np.mean(y_true != y_pred)


def plot_decision_boundary(X, y, w, title="Linear Separator"):
    """
    Plot 2D data and decision boundary:
        w0 + w1*x1 + w2*x2 = 0

    Solve for x2:
        x2 = -(w0 + w1*x1) / w2
    """
    pos = X[y == 1]
    neg = X[y == -1]

    plt.figure(figsize=(7, 6))
    plt.scatter(pos[:, 0], pos[:, 1], marker="+", label="Class +1")
    plt.scatter(neg[:, 0], neg[:, 1], marker="o", label="Class -1")

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x_vals = np.linspace(x_min, x_max, 200)

    # check if w2 is too close to zero to avoid division problems
    if abs(w[2]) > 1e-10:
        y_vals = -(w[0] + w[1] * x_vals) / w[2]
        plt.plot(x_vals, y_vals, label="Decision boundary")
    else:
        # vertical line: w0 + w1*x1 = 0 => x1 = -w0/w1
        if abs(w[1]) > 1e-10:
            x_vertical = -w[0] / w[1]
            plt.axvline(x=x_vertical, label="Decision boundary")
        else:
            print("Cannot plot boundary: both w1 and w2 are near zero.")

    plt.title(title)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()
    plt.grid(True)
    plt.show()


def run_classification_demo():
    """
    Run classification using linear regression.
    """
    X, y = generate_classification_clusters(n1=80, n2=80, seed=12, loc1=[2,2], loc2=[4,4])

    print("=" * 60)
    print("PART B: CLASSIFICATION WITH LINEAR REGRESSION")
    print("=" * 60)

    # draw clusters
    pos = X[y == 1]
    neg = X[y == -1]

    plt.figure(figsize=(7, 6))
    plt.scatter(pos[:, 0], pos[:, 1], marker="+", label="Class +1")
    plt.scatter(neg[:, 0], neg[:, 1], marker="o", label="Class -1")
    plt.title("Generated Two-Class Cluster Data")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Add bias
    Xb = np.c_[np.ones(X.shape[0]), X]

    # train using linear regression formula
    w = np.linalg.pinv(Xb) @ y

    # predictions
    scores = Xb @ w
    y_pred = sign(scores)

    err = classification_error(y, y_pred)

    print(f"Learned classification weights: {w}")
    print(f"Classification error = {err:.6f}")
    print(f"Accuracy = {(1 - err) * 100:.2f}%")

    # Plot separating boundary
    plot_decision_boundary(X, y, w, title="Using PLA")

    return X, y, w, err

def pla(Xb, y, w_init=None, max_iters=1000):
    """
    Standard Perceptron Learning Algorithm (PLA).

    Parameters:
        Xb -> data with bias already added
        y -> labels in {-1, +1}
        w_init -> initial weight vector
        max_iters -> maximum updates

    Returns:
        w -> final weight vector
        updates -> number of updates made
    """
    N, d = Xb.shape
    w = np.zeros(d) if w_init is None else w_init.copy()

    updates = 0
    for _ in range(max_iters):
        predictions = sign(Xb @ w)
        mis_idx = np.where(predictions != y)[0]

        # Stop if perfectly classified
        if len(mis_idx) == 0:
            break

        # Pick one misclassified point
        i = mis_idx[0]

        # PLA update rule
        w = w + y[i] * Xb[i]
        updates += 1

    return w, updates


def pocket_pla(Xb, y, w_init=None, max_iters=1000):
    """
    Pocket PLA:
    Keeps the best weight vector found so far.

    Useful when data is not perfectly linearly separable.
    """
    N, d = Xb.shape
    w = np.zeros(d) if w_init is None else w_init.copy()
    best_w = w.copy()

    # Current best error
    best_pred = sign(Xb @ best_w)
    best_err = classification_error(y, best_pred)

    updates = 0

    for _ in range(max_iters):
        pred = sign(Xb @ w)
        mis_idx = np.where(pred != y)[0]

        if len(mis_idx) == 0:
            # Perfect classification found
            best_w = w.copy()
            best_err = 0.0
            break

        # Pick a misclassified point
        i = mis_idx[0]

        # Update current weights
        w = w + y[i] * Xb[i]
        updates += 1

        # Check if this new weight is better
        current_pred = sign(Xb @ w)
        current_err = classification_error(y, current_pred)

        if current_err < best_err:
            best_err = current_err
            best_w = w.copy()

    return best_w, best_err, updates


def compare_random_vs_linear_regression_initialization():
    """
    Demonstrate that using linear regression weights as the initial
    weights for PLA / Pocket often improves performance.
    """
    X, y = generate_classification_clusters(n1=80, n2=80, seed=12, loc1=[2,2], loc2=[4,4])
    Xb = np.c_[np.ones(X.shape[0]), X]

    print("=" * 60)
    print("PART C: HOW LINEAR REGRESSION IMPROVES PLA / POCKET")
    print("=" * 60)

    # 1) Random / zero initialization
    w_pla_zero, pla_updates_zero = pla(Xb, y, w_init=None, max_iters=1000)
    pred_pla_zero = sign(Xb @ w_pla_zero)
    err_pla_zero = classification_error(y, pred_pla_zero)

    w_pocket_zero, pocket_err_zero, pocket_updates_zero = pocket_pla(
        Xb, y, w_init=None, max_iters=1000
    )

    # 2) Linear regression initialization
    w_lr = np.linalg.pinv(Xb) @ y

    w_pla_lr, pla_updates_lr = pla(Xb, y, w_init=w_lr, max_iters=1000)
    pred_pla_lr = sign(Xb @ w_pla_lr)
    err_pla_lr = classification_error(y, pred_pla_lr)

    w_pocket_lr, pocket_err_lr, pocket_updates_lr = pocket_pla(
        Xb, y, w_init=w_lr, max_iters=1000
    )

    print("PLA starting from zero weights:")
    print(f"  updates = {pla_updates_zero}")
    print(f"  final classification error = {err_pla_zero:.6f}")

    print("\nPLA starting from linear regression weights:")
    print(f"  updates = {pla_updates_lr}")
    print(f"  final classification error = {err_pla_lr:.6f}")

    print("\nPocket starting from zero weights:")
    print(f"  updates = {pocket_updates_zero}")
    print(f"  best classification error = {pocket_err_zero:.6f}")

    print("\nPocket starting from linear regression weights:")
    print(f"  updates = {pocket_updates_lr}")
    print(f"  best classification error = {pocket_err_lr:.6f}")

    # Plot the best Pocket result using LR initialization
    plot_decision_boundary(
        X, y, w_pocket_lr,
        title="Pocket PLA Boundary (Initialized with Linear Regression)"
    )

    return {
        "pla_zero_err": err_pla_zero,
        "pla_lr_err": err_pla_lr,
        "pocket_zero_err": pocket_err_zero,
        "pocket_lr_err": pocket_err_lr
    }

def main():
    """
    Run all parts of the lab.
    """
    # Part A: Linear Regression
    run_linear_regression_demo()

    # Part B: Classification using Linear Regression
    run_classification_demo()

    # Part C: Show LR improving PLA / Pocket
    compare_random_vs_linear_regression_initialization()

if __name__ == "__main__":
    main()