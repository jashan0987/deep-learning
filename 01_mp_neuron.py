import numpy as np

X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])


# McCulloch-Pitts Neuron
def mp_neuron(excitatory_inputs, threshold, inhibitory_inputs=None):

    if inhibitory_inputs is not None:
        if any(inhibitory_inputs):
            return 0

    total = sum(excitatory_inputs)

    if total >= threshold:
        return 1
    else:
        return 0


# AND Gate
def mp_and(x1, x2):
    return mp_neuron(
        excitatory_inputs=[x1, x2],
        threshold=2
    )


print("\n== AND GATE ==")

for x1, x2 in X:
    print(x1, x2, "->", mp_and(x1, x2))


# OR Gate
def mp_or(x1, x2):
    return mp_neuron(
        excitatory_inputs=[x1, x2],
        threshold=1
    )


print("\n== OR GATE ==")

for x1, x2 in X:
    print(x1, x2, "->", mp_or(x1, x2))


# NOT X1
def mp_not_x1(x1, x2):
    return mp_neuron(
        excitatory_inputs=[],
        threshold=0,
        inhibitory_inputs=[x1]
    )


print("\n== NOT X1 GATE ==")

for x1, x2 in X:
    print(x1, x2, "->", mp_not_x1(x1, x2))


# NOT X2
def mp_not_x2(x1, x2):
    return mp_neuron(
        excitatory_inputs=[],
        threshold=0,
        inhibitory_inputs=[x2]
    )


print("\n== NOT X2 GATE ==")

for x1, x2 in X:
    print(x1, x2, "->", mp_not_x2(x1, x2))


# NAND Gate
def mp_nand(x1, x2):
    and_result = mp_and(x1, x2)
    return 1 - and_result


print("\n== NAND GATE ==")

for x1, x2 in X:
    print(x1, x2, "->", mp_nand(x1, x2))


# NOR Gate
def mp_nor(x1, x2):
    return mp_neuron(
        excitatory_inputs=[],
        threshold=0,
        inhibitory_inputs=[x1, x2]
    )


print("\n== NOR GATE ==")

for x1, x2 in X:
    print(x1, x2, "->", mp_nor(x1, x2))


# XOR Gate
def mp_xor(x1, x2):
    or_result = mp_or(x1, x2)
    and_result = mp_and(x1, x2)

    not_and = 1 - and_result

    return mp_and(or_result, not_and)


print("\n== XOR GATE ==")

for x1, x2 in X:
    print(x1, x2, "->", mp_xor(x1, x2))


# XNOR Gate
def mp_xnor(x1, x2):
    return 1 - mp_xor(x1, x2)


print("\n== XNOR GATE ==")

for x1, x2 in X:
    print(x1, x2, "->", mp_xnor(x1, x2))