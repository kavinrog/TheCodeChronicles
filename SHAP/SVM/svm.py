"""
Support Vector Machine (SVM) from Scratch

Author: Kavinder Roghit Kanthen
Date: 03-09-2025
Last Modified: 03-09-2025

Description:
svm.py - Implements a Support Vector Machine (SVM) classifier using Stochastic Gradient Descent (SGD) 
for binary classification. The model is trained using the hinge loss function and L2 regularization.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Support Vector Machine (SVM) implementation using SGD optimization
class SVM:
    
    def __init__(self, learning_rate=0.001, iteration=1000, lambda_param=0.01):
        self.learning_rate = learning_rate  # Learning rate for weight updates
        self.iteration = iteration  # Number of training iterations
        self.lambda_param = lambda_param  # Regularization parameter
        self.weights = None
        self.bias = None 
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)  # Initialize weights
        self.bias = 0  # Initialize bias
        
        y_transformed = np.where(y <= 0, -1, 1)  # Ensure labels are {-1, 1}

        for _ in range(self.iteration):
            for idx, xi in enumerate(X): 
                condition = y_transformed[idx] * (np.dot(xi, self.weights) + self.bias) >= 1
                if condition:
                    self.weights -= self.learning_rate * (2 * self.lambda_param * self.weights)  # Regularization
                else:
                    self.weights -= self.learning_rate * (2 * self.lambda_param * self.weights - y_transformed[idx] * xi)
                    self.bias -= self.learning_rate * y_transformed[idx]  # Bias update when misclassified
                    
    def predict(self, X):
        approx = np.dot(X, self.weights) + self.bias  # Linear decision function
        return np.sign(approx)  # Assign class based on sign

# Generate synthetic dataset
np.random.seed(42)
num_samples = 100

X_class1 = np.random.randn(num_samples // 2, 2) - 2  # Class -1 points
y_class1 = np.full((num_samples // 2,), -1)

X_class2 = np.random.randn(num_samples // 2, 2) + 2  # Class 1 points
y_class2 = np.full((num_samples // 2,), 1)

X = np.vstack((X_class1, X_class2))  # Combine both classes
y = np.hstack((y_class1, y_class2))  # Labels

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train SVM model
svm = SVM()
svm.fit(X_train, y_train)

# Predict on test data
y_pred = svm.predict(X_test)

def plot_decision_boundary(X, y, model):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    
    # Predict on the mesh grid
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot the decision boundary
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')  # Color-filled boundary
    plt.scatter(X[y == -1][:, 0], X[y == -1][:, 1], color='red', label='Class -1')
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='blue', label='Class 1')
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("SVM Decision Boundary Visualization")
    plt.legend()
    plt.show()

# Call function to plot decision boundary
plot_decision_boundary(X, y, svm)