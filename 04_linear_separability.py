import numpy as np
import itertools
import pandas as pd


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


print("\n==================================")
print("PART 1: ALL 16 BOOLEAN FUNCTIONS")
print("====================================")

for outputs in itertools.product([0, 1], repeat=4):
    print(outputs)

results = []

for outputs in itertools.product([0, 1], repeat=4):

    y = np.array(outputs)

    weights, bias, converged = train_perceptron(
        X,
        y,
        learning_rate=1,
        epochs=100
    )

    results.append({
        "Function": outputs,
        "Linearly Separable": converged
    })


# Create DataFrame
df = pd.DataFrame(results)

linearly_separable = df[
    df["Linearly Separable"] == True
]

print("\n========================================")
print("PART 2: LINEARLY SEPARABLE FUNCTIONS")
print("========================================")

print(linearly_separable.to_string(index=False))

non_linearly_separable = df[
    df["Linearly Separable"] == False
]

print("\n========================================")
print("PART 3: NON-LINEARLY SEPARABLE FUNCTIONS")
print("========================================")

print(non_linearly_separable.to_string(index=False))