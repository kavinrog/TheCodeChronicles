"""
Logistic Regression from Scratch

Author: Kavinder Roghit Kanthen
Date: 03-06-2025
Last Modified: 03-06-2025

Description:
logistic_regression.py - Implements a logistic regression model for binary classification using the sigmoid function."""
import numpy as np 
import matplotlib.pyplot as plt 

# Logistic Regression Class
class LogisticRegression:
    
    def __init__(self, learning_rate=0.001, iteration=1000):
        """
        Initializes the logistic regression model.

        Parameters:
        learning_rate : float - Step size for gradient descent
        iteration : int - Number of iterations for training
        """
        self.learning_rate = learning_rate
        self.iteration = iteration
        self.weights = None
        self.bias = None 
    
    def sigmoid(self, z):
        """
        Computes the sigmoid activation function.

        Parameters:
        z : array-like - Input value(s)

        Returns:
        Sigmoid output between 0 and 1.
        """
        return 1 / (1 + np.exp(-z))
        
    def fit(self, X, y):
        """
        Trains the logistic regression model using gradient descent.

        Parameters:
        X : numpy array - Training data (features)
        y : numpy array - Labels (0 or 1)
        """
        n_samples, n_features = X.shape
        # Initialize weights and bias to zero
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Gradient descent
        for _ in range(self.iteration):
            # Compute the linear model: X * weights + bias
            linear_model = np.dot(X, self.weights) + self.bias
            # Apply sigmoid function to get probability predictions
            y_pred = self.sigmoid(linear_model)
            
            # Compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))  # Gradient for weights
            db = (1 / n_samples) * np.sum(y_pred - y)  # Gradient for bias
            
            # Update weights and bias
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
    def predict(self, X):
        """
        Makes predictions using the trained logistic regression model.

        Parameters:
        X : numpy array - Input data (features)

        Returns:
        List of predicted labels (0 or 1).
        """
        linear_model = np.dot(X, self.weights) + self.bias
        y_pred = self.sigmoid(linear_model)
        return [1 if i > 0.5 else 0 for i in y_pred]  # Convert probabilities to 0 or 1
            
# Generate synthetic dataset
np.random.seed(42)
n_samples = 100
X = np.random.randn(n_samples, 2)  # Generate 100 samples with 2 features
y = np.array([1 if x1 + x2 > 0 else 0 for x1, x2 in X])  # Assign labels based on x1 + x2 > 0

# Initialize and train logistic regression model
model = LogisticRegression(learning_rate=0.01, iteration=1000)
model.fit(X, y)

# Predict labels for the dataset
y_pred = model.predict(X)

# Scatter plot of dataset with actual labels
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, edgecolors='k')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

# Create a grid of points for decision boundary visualization
xx, yy = np.meshgrid(np.linspace(-3, 3, 100), np.linspace(-3, 3, 100))
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])  # Predict labels for each grid point
Z = np.array(Z).reshape(xx.shape)  # Reshape to match grid shape
plt.contourf(xx, yy, Z, alpha=0.3)  # Plot decision boundary

plt.show()  # Show plot