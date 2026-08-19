import numpy as np


# Input combinations
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])


# Pattern Detector
def pattern_detector(x1, x2, target_x1, target_x2):
    """
    Returns 1 only when (x1, x2)
    matches the target pattern.
    """

    return int(x1 == target_x1 and x2 == target_x2)


# General Boolean Function
def mlp_boolean_function(x1, x2, truth_table):
    """
    truth_table:

    [output_for_00,
     output_for_01,
     output_for_10,
     output_for_11]
    """

    h00 = pattern_detector(x1, x2, 0, 0)
    h01 = pattern_detector(x1, x2, 0, 1)
    h10 = pattern_detector(x1, x2, 1, 0)
    h11 = pattern_detector(x1, x2, 1, 1)

    hidden_outputs = [h00, h01, h10, h11]

    # Exactly one hidden neuron is active
    for h, desired_output in zip(hidden_outputs, truth_table):

        if h == 1:
            return desired_output

    return 0


# AND
AND_TABLE = [0, 0, 0, 1]

print("\n== AND ==")

for x1, x2 in X:
    print(
        x1, x2,
        "->",
        mlp_boolean_function(x1, x2, AND_TABLE)
    )


# OR
OR_TABLE = [0, 1, 1, 1]

print("\n== OR ==")

for x1, x2 in X:
    print(
        x1, x2,
        "->",
        mlp_boolean_function(x1, x2, OR_TABLE)
    )


# NAND
NAND_TABLE = [1, 1, 1, 0]

print("\n== NAND ==")

for x1, x2 in X:
    print(
        x1, x2,
        "->",
        mlp_boolean_function(x1, x2, NAND_TABLE)
    )


# NOR
NOR_TABLE = [1, 0, 0, 0]

print("\n== NOR ==")

for x1, x2 in X:
    print(
        x1, x2,
        "->",
        mlp_boolean_function(x1, x2, NOR_TABLE)
    )


# XOR
XOR_TABLE = [0, 1, 1, 0]

print("\n== XOR ==")

for x1, x2 in X:
    print(
        x1, x2,
        "->",
        mlp_boolean_function(x1, x2, XOR_TABLE)
    )


# XNOR
XNOR_TABLE = [1, 0, 0, 1]

print("\n== XNOR ==")

for x1, x2 in X:
    print(
        x1, x2,
        "->",
        mlp_boolean_function(x1, x2, XNOR_TABLE)
    )