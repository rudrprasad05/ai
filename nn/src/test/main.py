
# %% [markdown]
# LAB 4 (Week 6?)

# %%
import csv
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt


# %%
# load data

df = pd.read_csv('../data/pima-indians-diabetes.data', skiprows=2, header=None)

X = df.iloc[:, :-1] # everything except last column
y = df.iloc[:, -1] # last col
y = np.where(y == 0, -1, 1) # tanh uses -1/+1

# %%
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.4,
    random_state=42
)

# scale the values
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_train = np.hstack((np.ones((X_train.shape[0], 1)), X_train))

X_test = scaler.transform(X_test)
X_test = np.hstack((np.ones((X_test.shape[0], 1)), X_test))

# verify shape
print(X_train.shape)
print(X_test.shape)

# %%
def tanh(x):
    x = np.clip(x, -500, 500)
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

def tanh_derivative(x):
    return 1 - x**2

LR = 0.1
output_size = 1

target_success = 0.85
max_epochs = 5000
wait_epoch = 300
min_improvement = 0.0


def train_and_evaluate(hidden_size: int, seed: int = 42):
    np.random.seed(seed)
    input_size = X_train.shape[1]

    W1 = np.random.randn(input_size, hidden_size) * 0.01
    W2 = np.random.randn(hidden_size + 1, output_size) * 0.01

    allLoss = []
    allAccuracy = []

    best_success = 0
    epochs_without_improvement = 0

    for epoch in range(max_epochs):
        epoch_loss = 0
        correct = 0

        for i in range(len(X_train)):
            # rn vector is (1,). we need to reshape it to (1,1)
            x_i = X_train[i].reshape(1, -1)
            y_i = np.array([[y_train[i]]])

            # forward pass
            hidden_linear = x_i @ W1
            hidden_output = tanh(hidden_linear)
            hidden_output = np.hstack((np.ones((hidden_output.shape[0], 1)), hidden_output))

            output_linear = hidden_output @ W2
            predicted_output = tanh(output_linear)

            # loss
            sample_loss = np.mean((predicted_output - y_i) ** 2)
            epoch_loss += sample_loss

            # classification
            predicted_label = 1 if predicted_output.item() >= 0 else -1
            if predicted_label == y_i.item():
                correct += 1

            # backward pass
            delta_output = (predicted_output - y_i) * tanh_derivative(predicted_output)
            delta_hidden_full = (delta_output @ W2.T) * tanh_derivative(hidden_output)
            delta_hidden = delta_hidden_full[:, 1:]

            # update
            W2 = W2 - LR * (hidden_output.T @ delta_output)
            W1 = W1 - LR * (x_i.T @ delta_hidden)

        current_loss = epoch_loss / len(X_train)
        success_rate = correct / len(X_train)

        allLoss.append(current_loss)
        allAccuracy.append(success_rate)

        print(f"Hidden={hidden_size:2d} | Epoch {epoch+1}: Loss = {current_loss:.6f}, Success Rate = {success_rate:.4f}")

        # check improvement
        if success_rate > best_success + min_improvement:
            best_success = success_rate
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        # stop only if:
        # - success rate has reached target
        # - it has plateaued
        if best_success >= target_success and epochs_without_improvement >= wait_epoch:
            print(f"Hidden={hidden_size:2d} stopped: success rate plateaued at {best_success:.4f} for {wait_epoch} epochs")
            break
    else:
        print(f"Hidden={hidden_size:2d} stopped: reached max epochs")

    # %%
    correct = 0

    for i in range(len(X_test)):
        x_i = X_test[i].reshape(1, -1)
        y_i = y_test[i]

        hidden_linear = x_i @ W1
        hidden_output = tanh(hidden_linear)
        hidden_output = np.hstack((np.ones((hidden_output.shape[0], 1)), hidden_output))

        output_linear = hidden_output @ W2
        predicted_output = tanh(output_linear)

        predicted_label = 1 if predicted_output.item() >= 0 else -1

        if predicted_label == y_i:
            correct += 1

    test_accuracy = correct / len(X_test)
    print(f"Hidden={hidden_size:2d} | Test Accuracy: {test_accuracy:.4f}")

    return {
        "hidden_size": hidden_size,
        "epochs_ran": epoch + 1,
        "best_train_success": best_success,
        "final_train_success": allAccuracy[-1],
        "final_train_loss": allLoss[-1],
        "test_accuracy": test_accuracy,
    }


results = []
for hidden_size in range(1, 31):
    results.append(train_and_evaluate(hidden_size))

results_df = pd.DataFrame(results)
results_df.to_csv("nn_hidden_size_results.csv", index=False)
print("Saved results to nn_hidden_size_results.csv")

best_row = results_df.sort_values(by="test_accuracy", ascending=False).iloc[0]
print("Best hidden size based on test accuracy:")
print(best_row)

plt.figure(figsize=(10, 6))
plt.plot(results_df["hidden_size"], results_df["test_accuracy"], marker='o', label="Test Accuracy")
plt.plot(results_df["hidden_size"], results_df["best_train_success"], marker='s', label="Best Train Success")
plt.xlabel("Hidden Layer Neurons")
plt.ylabel("Accuracy / Success Rate")
plt.title("Empirical Test: Hidden Size vs Performance")
plt.xticks(range(1, 31))
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("nn_hidden_size_results.png", dpi=300)
plt.show()
print("Saved graph to nn_hidden_size_results.png")
