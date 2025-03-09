import numpy as np

class SVM:
    
    def __init__(self, learning_rate=0.001, iteration=1000, lambda_param=0.01):
        self.learning_rate = learning_rate
        self.iteration = iteration
        self.lambda_param = lambda_param
        self.weights = None
        self.bias = None 
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0 
        
        # Ensure labels are in {-1, 1}
        y_transformed = np.where(y <= 0, -1, 1)

        for _ in range(self.iteration):
            for idx, xi in enumerate(X):  # xi is a single sample
                condition = y_transformed[idx] * (np.dot(xi, self.weights) + self.bias) >= 1
                if condition:
                    self.weights -= self.learning_rate * (2 * self.lambda_param * self.weights)
                else:
                    self.weights -= self.learning_rate * (2 * self.lambda_param * self.weights - y_transformed[idx] * xi)
                    self.bias -= self.learning_rate * y_transformed[idx]
                    
    def predict(self, X):
        approx = np.dot(X, self.weights) + self.bias
        return np.sign(approx)