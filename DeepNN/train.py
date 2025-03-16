import argparse
from dataset import load_data
from deepnn import DeepNNScratch as dnnworelu
from deepnnrelu import DeepNNScratch as dnnwithrelu
import numpy as np
import matplotlib.pyplot as plt

def plot_decision_boundary(model, X, y):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    
    grid = np.c_[xx.ravel(), yy.ravel()]
    preds = model.predict(grid).reshape(xx.shape)
    
    plt.contourf(xx, yy, preds, alpha=0.5, cmap="coolwarm")
    plt.scatter(X[:, 0], X[:, 1], c=y.ravel(), edgecolor="k", cmap="coolwarm")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("Decision Boundary")
    plt.show()

# Parse command-line argument for model selection
parser = argparse.ArgumentParser(description="Train a neural network model.")
parser.add_argument("--model", type=str, choices=["dnnworelu", "dnnwithrelu"], required=True,
                    help="Choose model: 'dnnworelu' for DeepNNScratch without ReLU, 'dnnwithrelu' for DeepNNScratch with ReLU")
args = parser.parse_args()

# Load dataset
X_train, X_test, y_train, y_test = load_data()

# Select model based on command-line argument
if args.model == "dnnworelu":
    print("Using Model: DeepNNScratch **without ReLU activation**")
    nn = dnnworelu(input_size=2, output_size=1, hidden_size=4, iteration=2000, learning_rate=0.01)
else:
    print("Using Model: DeepNNScratch **with ReLU activation**")
    nn = dnnwithrelu(input_size=2, output_size=1, hidden_size1=4, hidden_size2 = 4, iteration=2000, learning_rate=0.01)

# Train model
print("Training the model...")
nn.fit(X_train, y_train)

# Plot loss if applicable
if hasattr(nn, "plot_loss"):
    nn.plot_loss()

# Plot decision boundary
plot_decision_boundary(nn, X_train, y_train)

# Evaluate model
y_pred_train = nn.predict(X_train)
y_pred_test = nn.predict(X_test)

train_accuracy = np.mean(y_pred_train == y_train) * 100
test_accuracy = np.mean(y_pred_test == y_test) * 100

print(f"Training Accuracy: {train_accuracy:.2f}%")
print(f"Testing Accuracy: {test_accuracy:.2f}%")