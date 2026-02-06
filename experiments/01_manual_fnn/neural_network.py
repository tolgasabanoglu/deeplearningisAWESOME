"""
Manual Feedforward Neural Network Implementation
Built from scratch using only NumPy - no TensorFlow, PyTorch, or Keras

This implementation helps understand the fundamentals:
- Forward propagation
- Backpropagation
- Gradient descent
- Weight initialization
"""

import numpy as np


class Activation:
    """Collection of activation functions and their derivatives"""

    @staticmethod
    def sigmoid(x):
        """
        Sigmoid: σ(x) = 1 / (1 + e^(-x))
        Range: (0, 1)
        Use: Output layer for binary classification
        """
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))  # Clip to prevent overflow

    @staticmethod
    def sigmoid_derivative(x):
        """
        Derivative: σ'(x) = σ(x) * (1 - σ(x))
        """
        s = Activation.sigmoid(x)
        return s * (1 - s)

    @staticmethod
    def relu(x):
        """
        ReLU: f(x) = max(0, x)
        Range: [0, ∞)
        Use: Hidden layers (most common)
        """
        return np.maximum(0, x)

    @staticmethod
    def relu_derivative(x):
        """
        Derivative: f'(x) = 1 if x > 0, else 0
        """
        return (x > 0).astype(float)

    @staticmethod
    def tanh(x):
        """
        Tanh: f(x) = (e^x - e^(-x)) / (e^x + e^(-x))
        Range: (-1, 1)
        Use: Hidden layers (centered around 0)
        """
        return np.tanh(x)

    @staticmethod
    def tanh_derivative(x):
        """
        Derivative: f'(x) = 1 - tanh²(x)
        """
        return 1 - np.tanh(x) ** 2


class NeuralNetwork:
    """
    Manual Feedforward Neural Network

    Architecture:
    - Input layer (n features)
    - Hidden layers (customizable)
    - Output layer (1 or more neurons)

    Example:
        nn = NeuralNetwork(
            layer_sizes=[4, 8, 8, 1],  # 4 inputs, 2 hidden layers (8 neurons each), 1 output
            activations=['relu', 'relu', 'sigmoid'],
            learning_rate=0.01
        )
    """

    def __init__(self, layer_sizes, activations, learning_rate=0.01, random_seed=42):
        """
        Initialize neural network

        Args:
            layer_sizes: List of integers [input_size, hidden1, hidden2, ..., output_size]
            activations: List of activation function names for each layer
            learning_rate: Learning rate for gradient descent
            random_seed: Random seed for reproducibility
        """
        np.random.seed(random_seed)

        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes) - 1  # Number of weight matrices
        self.learning_rate = learning_rate
        self.activations = activations

        # Initialize weights and biases using He initialization
        self.weights = []
        self.biases = []

        for i in range(self.num_layers):
            # He initialization: scale by sqrt(2 / n_inputs)
            # Good for ReLU activations
            w = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * np.sqrt(2.0 / layer_sizes[i])
            b = np.zeros((1, layer_sizes[i + 1]))

            self.weights.append(w)
            self.biases.append(b)

        # Map activation function names to actual functions
        self.activation_funcs = []
        self.activation_derivs = []

        for act_name in activations:
            if act_name == 'sigmoid':
                self.activation_funcs.append(Activation.sigmoid)
                self.activation_derivs.append(Activation.sigmoid_derivative)
            elif act_name == 'relu':
                self.activation_funcs.append(Activation.relu)
                self.activation_derivs.append(Activation.relu_derivative)
            elif act_name == 'tanh':
                self.activation_funcs.append(Activation.tanh)
                self.activation_derivs.append(Activation.tanh_derivative)
            else:
                raise ValueError(f"Unknown activation: {act_name}")

    def forward(self, X):
        """
        Forward propagation

        Math for each layer l:
        1. Z[l] = A[l-1] @ W[l] + b[l]  (linear transformation)
        2. A[l] = activation(Z[l])       (non-linear activation)

        Args:
            X: Input data (n_samples, n_features)

        Returns:
            Output activations (n_samples, output_size)
        """
        self.cache = {'A0': X}  # Store activations for backprop

        A = X
        for i in range(self.num_layers):
            # Linear transformation: Z = A @ W + b
            Z = A @ self.weights[i] + self.biases[i]

            # Non-linear activation: A = f(Z)
            A = self.activation_funcs[i](Z)

            # Cache for backpropagation
            self.cache[f'Z{i+1}'] = Z
            self.cache[f'A{i+1}'] = A

        return A

    def backward(self, X, y):
        """
        Backpropagation - Calculate gradients

        Chain rule applied layer by layer:
        1. Output layer: dL/dW = dL/dA * dA/dZ * dZ/dW
        2. Hidden layers: propagate error backwards

        Math:
        - dL/dZ[l] = dL/dA[l] * dA[l]/dZ[l]  (gradient at layer l)
        - dL/dW[l] = A[l-1].T @ dL/dZ[l]     (weight gradient)
        - dL/db[l] = sum(dL/dZ[l])           (bias gradient)
        - dL/dA[l-1] = dL/dZ[l] @ W[l].T    (propagate to previous layer)

        Args:
            X: Input data (n_samples, n_features)
            y: True labels (n_samples, output_size)
        """
        m = X.shape[0]  # Number of samples

        # Storage for gradients
        dW = [None] * self.num_layers
        db = [None] * self.num_layers

        # Output layer gradient
        # For MSE loss: dL/dA = 2(A - y)
        A_last = self.cache[f'A{self.num_layers}']
        dA = 2 * (A_last - y) / m

        # Backpropagate through each layer
        for i in reversed(range(self.num_layers)):
            # Get cached values
            Z = self.cache[f'Z{i+1}']
            A_prev = self.cache[f'A{i}']

            # Gradient of activation: dA/dZ
            dZ = dA * self.activation_derivs[i](Z)

            # Weight gradient: dL/dW = A_prev.T @ dZ
            dW[i] = A_prev.T @ dZ

            # Bias gradient: dL/db = sum(dZ) across samples
            db[i] = np.sum(dZ, axis=0, keepdims=True)

            # Propagate gradient to previous layer: dL/dA_prev
            if i > 0:
                dA = dZ @ self.weights[i].T

        return dW, db

    def update_weights(self, dW, db):
        """
        Update weights using gradient descent

        W_new = W_old - learning_rate * dL/dW
        b_new = b_old - learning_rate * dL/db

        Args:
            dW: List of weight gradients
            db: List of bias gradients
        """
        for i in range(self.num_layers):
            self.weights[i] -= self.learning_rate * dW[i]
            self.biases[i] -= self.learning_rate * db[i]

    def train(self, X, y, epochs=1000, verbose=True):
        """
        Train the neural network

        Args:
            X: Training data (n_samples, n_features)
            y: Training labels (n_samples, output_size)
            epochs: Number of training iterations
            verbose: Print loss during training

        Returns:
            List of losses per epoch
        """
        losses = []

        for epoch in range(epochs):
            # Forward pass
            predictions = self.forward(X)

            # Calculate loss (Mean Squared Error)
            loss = np.mean((predictions - y) ** 2)
            losses.append(loss)

            # Backward pass
            dW, db = self.backward(X, y)

            # Update weights
            self.update_weights(dW, db)

            # Print progress
            if verbose and (epoch % 100 == 0 or epoch == epochs - 1):
                print(f"Epoch {epoch:4d}/{epochs} - Loss: {loss:.6f}")

        return losses

    def predict(self, X):
        """
        Make predictions

        Args:
            X: Input data (n_samples, n_features)

        Returns:
            Predictions (n_samples, output_size)
        """
        return self.forward(X)

    def get_architecture_summary(self):
        """Print network architecture"""
        print("\n=== Neural Network Architecture ===")
        print(f"Learning Rate: {self.learning_rate}")
        print(f"\nLayer structure:")
        for i in range(self.num_layers):
            print(f"  Layer {i+1}: {self.layer_sizes[i]:3d} -> {self.layer_sizes[i+1]:3d} "
                  f"[{self.activations[i]}] "
                  f"(Weights: {self.weights[i].shape}, Bias: {self.biases[i].shape})")
        print(f"\nTotal parameters: {sum(w.size + b.size for w, b in zip(self.weights, self.biases))}")
        print("=" * 40)
