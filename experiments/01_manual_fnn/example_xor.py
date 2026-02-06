"""
Example: Solving the XOR Problem with Manual Neural Network

XOR (Exclusive OR) is a classic non-linear problem that cannot be solved by a single perceptron.
It requires at least one hidden layer to learn the XOR function.

Truth table:
Input (X1, X2) | Output (y)
(0, 0)         | 0
(0, 1)         | 1
(1, 0)         | 1
(1, 1)         | 0
"""

import numpy as np
import matplotlib.pyplot as plt
from neural_network import NeuralNetwork


def create_xor_dataset():
    """Create XOR dataset"""
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])

    y = np.array([
        [0],  # 0 XOR 0 = 0
        [1],  # 0 XOR 1 = 1
        [1],  # 1 XOR 0 = 1
        [0]   # 1 XOR 1 = 0
    ])

    return X, y


def plot_decision_boundary(nn, X, y):
    """Visualize the decision boundary learned by the network"""
    # Create a mesh grid
    x_min, x_max = -0.5, 1.5
    y_min, y_max = -0.5, 1.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))

    # Predict for each point in the mesh
    Z = nn.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot
    plt.figure(figsize=(10, 4))

    # Decision boundary
    plt.subplot(1, 2, 1)
    plt.contourf(xx, yy, Z, levels=20, cmap='RdYlBu', alpha=0.8)
    plt.colorbar(label='Prediction')
    plt.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap='RdYlBu',
                edgecolors='black', s=200, linewidths=2)
    plt.title('Decision Boundary')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.grid(True, alpha=0.3)

    # Add annotations
    for i, (x1, x2) in enumerate(X):
        plt.annotate(f'({int(x1)}, {int(x2)}) → {int(y[i][0])}',
                    (x1, x2), xytext=(10, 10), textcoords='offset points',
                    fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    return plt


def main():
    print("=" * 60)
    print("XOR Problem - Manual Neural Network Implementation")
    print("=" * 60)

    # Create dataset
    X, y = create_xor_dataset()

    print("\nDataset:")
    print("Input (X):")
    print(X)
    print("\nOutput (y):")
    print(y.ravel())

    # Create neural network
    # Architecture: 2 inputs -> 4 hidden neurons -> 1 output
    nn = NeuralNetwork(
        layer_sizes=[2, 4, 1],  # 2 inputs, 4 hidden neurons, 1 output
        activations=['relu', 'sigmoid'],  # ReLU for hidden, sigmoid for output
        learning_rate=0.1,
        random_seed=42
    )

    nn.get_architecture_summary()

    # Train
    print("\n" + "=" * 60)
    print("Training...")
    print("=" * 60)
    losses = nn.train(X, y, epochs=5000, verbose=True)

    # Test
    print("\n" + "=" * 60)
    print("Testing on XOR inputs:")
    print("=" * 60)
    predictions = nn.predict(X)

    for i, (input_vals, true_val, pred_val) in enumerate(zip(X, y, predictions)):
        rounded_pred = round(pred_val[0])
        correct = "✓" if rounded_pred == true_val[0] else "✗"
        print(f"Input: {input_vals} → Predicted: {pred_val[0]:.4f} "
              f"(rounded: {rounded_pred}) | True: {int(true_val[0])} {correct}")

    # Calculate accuracy
    rounded_predictions = np.round(predictions)
    accuracy = np.mean(rounded_predictions == y) * 100
    print(f"\nAccuracy: {accuracy:.1f}%")

    # Plot results
    print("\n" + "=" * 60)
    print("Generating visualizations...")
    print("=" * 60)

    # Decision boundary
    plt = plot_decision_boundary(nn, X, y)

    # Loss curve
    plt.subplot(1, 2, 2)
    plt.plot(losses, linewidth=2)
    plt.title('Training Loss (MSE)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')

    plt.tight_layout()
    plt.savefig('xor_results.png', dpi=150, bbox_inches='tight')
    print("Saved: xor_results.png")
    plt.show()

    # Manual calculation example for one input
    print("\n" + "=" * 60)
    print("Manual Forward Propagation Example: Input [1, 0]")
    print("=" * 60)

    test_input = np.array([[1, 0]])
    nn.forward(test_input)

    print("\nLayer 0 (Input):")
    print(f"  A0 = {nn.cache['A0']}")

    for i in range(nn.num_layers):
        print(f"\nLayer {i+1} ({nn.activations[i]}):")
        print(f"  Weights shape: {nn.weights[i].shape}")
        print(f"  Weights:\n{nn.weights[i]}")
        print(f"  Bias: {nn.biases[i]}")
        print(f"  Z{i+1} = A{i} @ W{i+1} + b{i+1}")
        print(f"  Z{i+1} = {nn.cache[f'Z{i+1}']}")
        print(f"  A{i+1} = {nn.activations[i]}(Z{i+1})")
        print(f"  A{i+1} = {nn.cache[f'A{i+1}']}")

    print("\n" + "=" * 60)
    print("Success! The network learned XOR function.")
    print("=" * 60)


if __name__ == "__main__":
    main()
