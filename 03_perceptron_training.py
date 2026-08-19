import numpy as np


# Input combinations
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])


# Perceptron Training Function
def train_perceptron(X, y, learning_rate=1, epochs=100):

    weights = np.zeros(X.shape[1])
    bias = 0

    for epoch in range(epochs):

        errors = 0

        for xi, target in zip(X, y):

            z = np.dot(weights, xi) + bias

            prediction = 1 if z >= 0 else 0

            update = learning_rate * (target - prediction)

            weights += update * xi
            bias += update

            if update != 0:
                errors += 1

        if errors == 0:
            return weights, bias, True

    return weights, bias, False


# AND Training
y_and = np.array([0, 0, 0, 1])

weights, bias, converged = train_perceptron(X, y_and)

print("\n== AND TRAINING ==")
print("Weights:", weights)
print("Bias:", bias)
print("Converged:", converged)


# OR Training
y_or = np.array([0, 1, 1, 1])

weights, bias, converged = train_perceptron(X, y_or)

print("\n== OR TRAINING ==")
print("Weights:", weights)
print("Bias:", bias)
print("Converged:", converged)


# NAND Training
y_nand = np.array([1, 1, 1, 0])

weights, bias, converged = train_perceptron(X, y_nand)

print("\n== NAND TRAINING ==")
print("Weights:", weights)
print("Bias:", bias)
print("Converged:", converged)


# NOR Training
y_nor = np.array([1, 0, 0, 0])

weights, bias, converged = train_perceptron(X, y_nor)

print("\n== NOR TRAINING ==")
print("Weights:", weights)
print("Bias:", bias)
print("Converged:", converged)


# XOR Training
y_xor = np.array([0, 1, 1, 0])

weights, bias, converged = train_perceptron(
    X,
    y_xor,
    learning_rate=1,
    epochs=100
)

print("\n== XOR TRAINING ==")
print("Weights:", weights)
print("Bias:", bias)
print("Converged:", converged)


# XNOR Training
y_xnor = np.array([1, 0, 0, 1])

weights, bias, converged = train_perceptron(
    X,
    y_xnor,
    learning_rate=1,
    epochs=100
)

print("\n== XNOR TRAINING ==")
print("Weights:", weights)
print("Bias:", bias)
print("Converged:", converged)