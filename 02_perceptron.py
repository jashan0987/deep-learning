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


# AND Gate
def perceptron_and(x1, x2):
    return perceptron(
        x1, x2,
        w1=1,
        w2=1,
        bias=-1.5
    )


print("\n== PERCEPTRON AND GATE ==")

for x1, x2 in X:
    print(x1, x2, "->", perceptron_and(x1, x2))


# OR Gate
def perceptron_or(x1, x2):
    return perceptron(
        x1, x2,
        w1=1,
        w2=1,
        bias=-0.5
    )


print("\n== PERCEPTRON OR GATE ==")

for x1, x2 in X:
    print(x1, x2, "->", perceptron_or(x1, x2))


# NOT X1
def perceptron_not_x1(x1, x2):
    return perceptron(
        x1, x2,
        w1=-1,
        w2=0,
        bias=0.5
    )


print("\n== PERCEPTRON NOT X1 ==")

for x1, x2 in X:
    print(x1, x2, "->", perceptron_not_x1(x1, x2))


# NOT X2
def perceptron_not_x2(x1, x2):
    return perceptron(
        x1, x2,
        w1=0,
        w2=-1,
        bias=0.5
    )


print("\n== PERCEPTRON NOT X2 ==")

for x1, x2 in X:
    print(x1, x2, "->", perceptron_not_x2(x1, x2))


# NAND Gate
def perceptron_nand(x1, x2):
    return perceptron(
        x1, x2,
        w1=-1,
        w2=-1,
        bias=1.5
    )


print("\n== PERCEPTRON NAND GATE ==")

for x1, x2 in X:
    print(x1, x2, "->", perceptron_nand(x1, x2))


# NOR Gate
def perceptron_nor(x1, x2):
    return perceptron(
        x1, x2,
        w1=-1,
        w2=-1,
        bias=0.5
    )


print("\n== PERCEPTRON NOR GATE ==")

for x1, x2 in X:
    print(x1, x2, "->", perceptron_nor(x1, x2))