import numpy as np
import matplotlib.pyplot as plt

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
            
            self.weights = self.weights - self.learning_rate * dw
            self.bias = self.bias - self.learning_rate * db
            
    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

np.random.seed(42)
n_sample = 100
n_feature =  1    
X = 2 * np.random.randn(n_sample, n_feature)
y = (7 + 3 * X + np.random.randn(n_sample, n_feature)).ravel()

model = LinearRegression()
model.fit(X,y)
y_pred = model.predict(X)
print(y_pred)

# Plot the results
plt.scatter(X, y, color='blue', label='Actual data')
plt.plot(X, y_pred, color='red', label='Predicted line')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()
            
        