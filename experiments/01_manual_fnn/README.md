# Manual Feedforward Neural Network

A from-scratch implementation of a feedforward neural network using **only NumPy** - no TensorFlow, PyTorch, or Keras.

## Learning Objectives

- Understand forward propagation step-by-step
- Implement backpropagation from scratch
- Calculate gradients manually using the chain rule
- See how activation functions and loss functions work
- Visualize what the network learns

---

## The Mathematics

### 1. Forward Propagation

For each layer `l`, we compute:

**Linear Transformation:**
```
Z[l] = A[l-1] @ W[l] + b[l]
```
Where:
- `A[l-1]` = activations from previous layer (shape: `m × n[l-1]`)
- `W[l]` = weight matrix (shape: `n[l-1] × n[l]`)
- `b[l]` = bias vector (shape: `1 × n[l]`)
- `Z[l]` = pre-activation values (shape: `m × n[l]`)

**Non-linear Activation:**
```
A[l] = f(Z[l])
```
Where `f` is an activation function (ReLU, sigmoid, tanh, etc.)

**Example with real numbers:**

Suppose we have:
- Input: `X = [1.0, 0.5]` (2 features)
- Hidden layer: 3 neurons
- Weights: `W1 = [[0.2, -0.3, 0.1], [0.4, 0.2, -0.1]]`
- Bias: `b1 = [0.1, 0.0, -0.2]`

```
Z1 = X @ W1 + b1
Z1 = [1.0, 0.5] @ [[0.2, -0.3, 0.1], [0.4, 0.2, -0.1]] + [0.1, 0.0, -0.2]
Z1 = [0.2 + 0.2, -0.3 + 0.1, 0.1 - 0.05] + [0.1, 0.0, -0.2]
Z1 = [0.5, -0.2, -0.15]

A1 = ReLU(Z1) = [0.5, 0.0, 0.0]
```

---

### 2. Activation Functions

#### ReLU (Rectified Linear Unit)
```
f(x) = max(0, x)
f'(x) = 1 if x > 0, else 0
```
- **Use**: Hidden layers (most common)
- **Range**: [0, ∞)
- **Pros**: No vanishing gradient, computationally efficient
- **Cons**: Dead neurons (outputs 0 forever if x < 0)

#### Sigmoid
```
f(x) = 1 / (1 + e^(-x))
f'(x) = f(x) * (1 - f(x))
```
- **Use**: Binary classification output layer
- **Range**: (0, 1)
- **Pros**: Smooth gradient, outputs probabilities
- **Cons**: Vanishing gradient for extreme values

#### Tanh
```
f(x) = (e^x - e^(-x)) / (e^x + e^(-x))
f'(x) = 1 - tanh²(x)
```
- **Use**: Hidden layers (centered data)
- **Range**: (-1, 1)
- **Pros**: Zero-centered (better than sigmoid)
- **Cons**: Vanishing gradient for extreme values

---

### 3. Loss Function (Mean Squared Error)

```
L(y, ŷ) = (1/m) * Σ(y - ŷ)²
```

Where:
- `y` = true labels
- `ŷ` = predictions
- `m` = number of samples

**Gradient:**
```
∂L/∂ŷ = (2/m) * (ŷ - y)
```

---

### 4. Backpropagation (The Chain Rule)

Backpropagation calculates how much each weight contributed to the error, working backwards from output to input.

**For output layer L:**
```
dL/dA[L] = 2(A[L] - y) / m          # Loss gradient
dL/dZ[L] = dL/dA[L] * f'(Z[L])      # Chain rule: multiply by activation derivative
```

**For each hidden layer l (from L-1 to 1):**
```
dL/dW[l] = A[l-1].T @ dL/dZ[l]      # Weight gradient
dL/db[l] = sum(dL/dZ[l], axis=0)    # Bias gradient
dL/dA[l-1] = dL/dZ[l] @ W[l].T      # Propagate to previous layer
dL/dZ[l-1] = dL/dA[l-1] * f'(Z[l-1])  # Apply activation derivative
```

**Step-by-step example:**

Network: Input (2) → Hidden (3) → Output (1)

1. **Forward pass** produces activations
2. **Compute output error**:
   - Predicted: 0.8, True: 1.0
   - Error: 0.8 - 1.0 = -0.2
3. **Backpropagate through output layer**:
   - `dL/dZ_out = -0.2 * sigmoid'(Z_out)`
   - Update output weights using this gradient
4. **Backpropagate through hidden layer**:
   - Propagate error: `dL/dZ_hidden = dL/dZ_out @ W_out.T * ReLU'(Z_hidden)`
   - Update hidden weights using this gradient

---

### 5. Gradient Descent

Update weights to minimize loss:

```
W_new = W_old - learning_rate * (∂L/∂W)
b_new = b_old - learning_rate * (∂L/∂b)
```

**Learning rate** controls step size:
- Too large: overshoot minimum, unstable training
- Too small: slow convergence

---

## Running the Code

### XOR Problem Example

```bash
cd experiments/01_manual_fnn
python example_xor.py
```

**What happens:**
1. Creates XOR dataset (4 samples)
2. Builds a 2→4→1 network (2 inputs, 4 hidden neurons, 1 output)
3. Trains for 5000 epochs
4. Visualizes decision boundary and loss curve
5. Shows detailed forward pass calculation

**Expected output:**
```
Epoch    0/5000 - Loss: 0.250000
Epoch  100/5000 - Loss: 0.235482
...
Epoch 5000/5000 - Loss: 0.000123

Testing on XOR inputs:
Input: [0 0] → Predicted: 0.0012 (rounded: 0) | True: 0 ✓
Input: [0 1] → Predicted: 0.9988 (rounded: 1) | True: 1 ✓
Input: [1 0] → Predicted: 0.9991 (rounded: 1) | True: 1 ✓
Input: [1 1] → Predicted: 0.0009 (rounded: 0) | True: 0 ✓

Accuracy: 100.0%
```

---

## Understanding the XOR Problem

XOR (Exclusive OR) is a non-linear problem that requires a hidden layer to solve:

| X1 | X2 | Output |
|----|----| ------ |
| 0  | 0  | 0      |
| 0  | 1  | 1      |
| 1  | 0  | 1      |
| 1  | 1  | 0      |

**Why can't a single perceptron solve it?**

A single perceptron draws a straight line to separate classes. XOR requires a curved decision boundary, which needs hidden layers.

**What does the network learn?**

The hidden layer creates intermediate representations:
- Neuron 1 might learn: "Is X1 active?"
- Neuron 2 might learn: "Is X2 active?"
- Neuron 3 might learn: "Are both active?"
- Output combines these: "Is exactly one active?"

---

## Code Structure

### `neural_network.py`
- `Activation` class: Activation functions + derivatives
- `NeuralNetwork` class: Complete implementation
  - `__init__`: Initialize weights (He initialization)
  - `forward`: Forward propagation
  - `backward`: Backpropagation (compute gradients)
  - `update_weights`: Gradient descent
  - `train`: Full training loop
  - `predict`: Make predictions

### `example_xor.py`
- XOR dataset creation
- Network training
- Decision boundary visualization
- Detailed forward pass walkthrough

---

## Weight Initialization: Why It Matters

**He Initialization** (used for ReLU):
```python
W = np.random.randn(n_in, n_out) * sqrt(2 / n_in)
```

**Why?**
- Random initialization breaks symmetry (all neurons learn different features)
- Proper scaling prevents vanishing/exploding gradients
- He init is optimized for ReLU activations

---

## Key Takeaways

1. **Forward pass**: Multiply by weights, add bias, apply activation
2. **Backward pass**: Chain rule to compute gradients
3. **Gradient descent**: Adjust weights to reduce loss
4. **Activation functions**: Introduce non-linearity (critical for learning complex patterns)
5. **Hidden layers**: Learn intermediate representations

---

## Extending This Implementation

**Ideas to try:**
1. Add more activation functions (LeakyReLU, ELU, Swish)
2. Implement different loss functions (Cross-Entropy, Huber)
3. Add regularization (L2, Dropout)
4. Implement mini-batch gradient descent
5. Add momentum or Adam optimizer
6. Try on real datasets (Iris, MNIST)
7. Visualize weight evolution during training

---

## Further Reading

- **Backpropagation**: [Calculus on Computational Graphs](http://colah.github.io/posts/2015-08-Backprop/)
- **Activation Functions**: [CS231n Stanford](http://cs231n.github.io/neural-networks-1/)
- **Initialization**: [He et al. 2015](https://arxiv.org/abs/1502.01852)
- **Optimization**: [An overview of gradient descent optimization algorithms](https://arxiv.org/abs/1609.04747)

---

## What You've Learned

By working through this implementation, you now understand:
- How matrix multiplication creates connections between layers
- Why activation functions are necessary
- How backpropagation computes gradients using the chain rule
- Why learning rate matters
- How neural networks learn to solve non-linear problems

**No more black boxes!** You know exactly what happens inside a neural network.
