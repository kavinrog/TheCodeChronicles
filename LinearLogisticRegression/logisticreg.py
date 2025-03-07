import numpy as np 
import matplotlib.pyplot as plt 

class LogisticRegression:
    
    def __init__(self, learning_rate = 0.001, iteration = 1000):
        self.learning_rate = learning_rate
        self.iteration = iteration
        self.weights = None
        self.bias = None 
    
    def sigmoid(self, z):
        return 1 / (1+np.exp(-z))
        
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for _ in range(self.iteration):
            linear_model = np.dot(self.weights, X) + self.bias
            y_pred = self.sigmoid(linear_model)
            
            dw = (1/n_samples)*np.dot(X.T, y_pred- y) + self.bias
            db = (1/n_samples)*np.sum(y_pred - y)
            
            self.weights-= self.learning_rate * dw
            self.bias-= self.learning_rate * db
            
    def predict(self, X):
        linear_model = np.dot(self.weights, X) + self.bias
        y_pred = self.sigmoid(linear_model)
        return y_pred
            
            
                        
            
        