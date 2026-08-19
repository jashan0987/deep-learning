import numpy as np


# Input combinations
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])


# Basic Perceptron
def perceptron(x1, x2, w1, w2, bias):

    z = w1 * x1 + w2 * x2 + bias

    if z >= 0:
        return 1
    else:
        return 0


# Perceptron AND
def perceptron_and(x1, x2):
    return perceptron(
        x1, x2,
        w1=1,
        w2=1,
        bias=-1.5
    )


# Perceptron OR
def perceptron_or(x1, x2):
    return perceptron(
        x1, x2,
        w1=1,
        w2=1,
        bias=-0.5
    )


# Perceptron NOT X1
def perceptron_not_x1(x1, x2):
    return perceptron(
        x1, x2,
        w1=-1,
        w2=0,
        bias=0.5
    )


# Perceptron NOT X2
def perceptron_not_x2(x1, x2):
    return perceptron(
        x1, x2,
        w1=0,
        w2=-1,
        bias=0.5
    )


# Perceptron NAND
def perceptron_nand(x1, x2):
    return perceptron(
        x1, x2,
        w1=-1,
        w2=-1,
        bias=1.5
    )


# MLP XOR
def mlp_xor(x1, x2):

    # Hidden layer
    h1 = perceptron_or(x1, x2)
    h2 = perceptron_nand(x1, x2)

    # Output layer
    output = perceptron_and(h1, h2)

    return output


print("\n== MLP XOR ==")

for x1, x2 in X:
    print(x1, x2, "->", mlp_xor(x1, x2))


# MLP XNOR
def mlp_xnor(x1, x2):

    xor_result = mlp_xor(x1, x2)

    return 1 - xor_result


print("\n== MLP XNOR ==")

for x1, x2 in X:
    print(x1, x2, "->", mlp_xnor(x1, x2))


# MLP AND
def mlp_and(x1, x2):

    # Hidden layer
    h1 = perceptron_and(x1, x2)

    # Output layer
    output = perceptron_and(h1, 0)

    return output


print("\n== MLP AND ==")

for x1, x2 in X:
    print(x1, x2, "->", mlp_and(x1, x2))


# MLP OR
def mlp_or(x1, x2):

    # Hidden layer
    h1 = perceptron_or(x1, x2)

    # Output layer
    output = perceptron_or(h1, 0)

    return output


print("\n== MLP OR ==")

for x1, x2 in X:
    print(x1, x2, "->", mlp_or(x1, x2))


# MLP NAND
def mlp_nand(x1, x2):

    # Hidden layer
    h1 = perceptron_nand(x1, x2)

    # Output layer
    output = perceptron_or(h1, 0)

    return output


print("\n== MLP NAND ==")

for x1, x2 in X:
    print(x1, x2, "->", mlp_nand(x1, x2))


# MLP NOT X1
def mlp_not_x1(x1, x2):

    # Hidden layer
    h1 = perceptron_not_x1(x1, x2)

    # Output layer
    output = perceptron_or(h1, 0)

    return output


print("\n== MLP NOT X1 ==")

for x1, x2 in X:
    print(x1, x2, "->", mlp_not_x1(x1, x2))


# MLP NOT X2
def mlp_not_x2(x1, x2):

    # Hidden layer
    h1 = perceptron_not_x2(x1, x2)

    # Output layer
    output = perceptron_or(h1, 0)

    return output


print("\n== MLP NOT X2 ==")

for x1, x2 in X:
    print(x1, x2, "->", mlp_not_x2(x1, x2))