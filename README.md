# Deep Learning

This repository contains Python implementations of fundamental Deep Learning

## Topics Covered

### 1. McCulloch-Pitts Neuron
File:

`01_mp_neuron.py`
This file implements the McCulloch-Pitts neuron and basic logic gates:
- AND
- OR
- NOT X1
- NOT X2
- NAND
- NOR
- XOR
- XNOR
---

### 2. Perceptron
File:

`02_perceptron.py`
This file implements a basic perceptron and logic gates using weights and bias:
- AND
- OR
- NOT X1
- NOT X2
- NAND
- NOR
---

### 3. Perceptron Training
File:

`03_perceptron_training.py`
This file demonstrates how a perceptron can learn Boolean functions using the perceptron learning algorithm.

Functions tested:
- AND
- OR
- NAND
- NOR
- XOR
- XNOR
The results demonstrate that a single perceptron can learn linearly separable functions but cannot learn XOR and XNOR.
---

### 4. Linear Separability
File:

`04_linear_separability.py`
This file generates and tests all 16 possible Boolean functions for two binary inputs.

The results show:
- 16 total Boolean functions
- 14 linearly separable functions
- 2 non-linearly separable functions

The two non-linearly separable functions are:
- XOR
- XNOR
---

### 5. Multi-Layer Perceptron
File:

`05_mlp.py`
This file demonstrates Multi-Layer Perceptron concepts using multiple perceptron layers.

Functions implemented:
- XOR
- XNOR
- AND
- OR
- NAND
- NOT X1
- NOT X2

The MLP demonstrates how multiple layers can implement functions that cannot be implemented by a single perceptron.
---

### 6. Boolean Functions
File:

`06_boolean_functions.py`
This file implements a general Boolean function using pattern detectors and truth tables.

Functions demonstrated:
- AND
- OR
- NAND
- NOR
- XOR
- XNOR

---
## Project Structure
```text
deep-learning/
│
├── 01_mp_neuron.py
├── 02_perceptron.py
├── 03_perceptron_training.py
├── 04_linear_separability.py
├── 05_mlp.py
├── 06_boolean_functions.py
├── main.py
└── README.md