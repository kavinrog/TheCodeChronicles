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
            linear_model = np.dot(X, self.weights) + self.bias
            y_pred = self.sigmoid(linear_model)
            
            dw = (1/n_samples)*np.dot(X.T, y_pred- y) 
            db = (1/n_samples)*np.sum(y_pred - y)
            
            self.weights-= self.learning_rate * dw
            self.bias-= self.learning_rate * db
            
    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_pred = self.sigmoid(linear_model)
        return [1 if i > 0.5 else 0 for i in y_pred]
            
np.random.seed(42)
n_samples = 100
X = np.random.randn(n_samples, 2)
y = np.array([1 if x1 + x2 > 0 else 0 for x1, x2 in X])           
                        
model = LogisticRegression(learning_rate=0.01, iteration=1000)
model.fit(X, y)

y_pred = model.predict(X)

plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, edgecolors='k')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

xx, yy = np.meshgrid(np.linspace(-3, 3, 100), np.linspace(-3, 3, 100))
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = np.array(Z).reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=0.3)

plt.show() 