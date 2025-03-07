import numpy as np

class LinearRegression:
    
    def __init__(self,learning_rate = 0.001, iterations = 1000):
        self.learning_rate= learning_rate
        self.iterations = iterations
        self.weights = None
        self.bias = None
        
    def fit(self, X, y):
        n_samples , n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0 
        for i in range(self.iterations):
            y_predicted = np.dot(X, self.weights)+self.bias
        
            dw = (1/n_samples) * np.dot(X.T, (y_predicted- y))
            db = (1/n_samples) * np.sum(y_predicted- y)
            
            self.weights = self.weights + self.learning_rate * dw
            self.bias = self.bias + self.learning_rate * db
            
    def predict(self, X):
        return np.dot(X, self.weights) + self.bias
    
            
            
            
        