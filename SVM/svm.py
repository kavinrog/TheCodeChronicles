import numpy as np 


class SVM:
    
    def __init__(self, learning_rate = 0.001, iteration = 1000, lambda_param = 0.01):
        self.learning_rate = learning_rate
        self.iteration = iteration
        self.lambda_param = lambda_param
        self.weights = None
        self.bias = None 
    
    def fit(self, X, y):
        n_samples, n_feature = X.shape
        self.weights = np.zeros(n_feature)
        self.bias = 0 
        
        y_pred = np.where(y<=0,-1,1)
        for _ in range(self.iteration):
            for idx, ix in enumerate(X):
                condition = y_pred[idx] * (np.dot(X, self.weights)+self.bias) >= 1
                if condition:
                    self.weights -= self.learning_rate * (2 * self.lambda_param * self.weights)
                else:
                    self.weights -= self.learning_rate * (2 * self.lambda_param * self.weights - np.dot(ix, y_pred[idx]))
                    self.bias -= self.learning_rate * y_pred[idx]
                    
    def predict(self, X):
        approx = np.dot(X, self.weights) + self.bias
        return np.sign(approx)
                    
        