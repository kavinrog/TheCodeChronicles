"""
Linear Regression from Scratch

Author: Kavinder Roghit Kanthen
Date: 03-06-2025
Last Modified: 03-06-2025

Description:
linear_regression.py - Implements a simple linear regression model using gradient descent for predicting continuous values.
"""
import numpy as np
import matplotlib.pyplot as plt

# Define the Linear Regression class
class LinearRegression:
    
    def __init__(self, learning_rate=0.001, iterations=1000):
        """
        Initializes the Linear Regression model.

        Parameters:
        learning_rate : float - Step size for gradient descent
        iterations : int - Number of iterations for training
        """
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights = None
        self.bias = None
        
    def fit(self, X, y):
        """
        Trains the Linear Regression model using Gradient Descent.

        Parameters:
        X : numpy array - Training data (features)
        y : numpy array - Labels (target values)
        """
        n_samples, n_features = X.shape
        # Initialize weights and bias to zero
        self.weights = np.zeros(n_features)
        self.bias = 0 

        # Gradient Descent Optimization
        for i in range(self.iterations):
            # Compute predicted values (y = Xw + b)
            y_predicted = np.dot(X, self.weights) + self.bias
        
            # Compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))  # Derivative w.r.t weights
            db = (1 / n_samples) * np.sum(y_predicted - y)  # Derivative w.r.t bias
            
            # Update weights and bias
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
    def predict(self, X):
        """
        Predicts output values using the trained model.

        Parameters:
        X : numpy array - Input data (features)

        Returns:
        Predicted target values.
        """
        return np.dot(X, self.weights) + self.bias

# Generate synthetic dataset
np.random.seed(42)  # Ensure reproducibility
n_sample = 100  # Number of data points
n_feature = 1  # Number of features (1D input)

# Generate random feature values
X = 2 * np.random.randn(n_sample, n_feature)

# Generate target values with some noise
y = (7 + 3 * X + np.random.randn(n_sample, n_feature)).ravel()  # Flatten to 1D array

# Initialize and train the linear regression model
model = LinearRegression()
model.fit(X, y)

# Predict target values using the trained model
y_pred = model.predict(X)
print(y_pred)  # Print predicted values

# Plot the actual data and the regression line
plt.scatter(X, y, color='blue', label='Actual data')  # Scatter plot of actual data
plt.plot(X, y_pred, color='red', label='Predicted line')  # Regression line
plt.xlabel('X')  # Label x-axis
plt.ylabel('y')  # Label y-axis
plt.legend()  # Show legend
plt.show()  # Display the plot