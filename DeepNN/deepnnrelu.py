import numpy as np 
import matplotlib.pyplot as plt

class DeepNNScratch:
    def __init__(self, input_size, output_size, hidden_size1, hidden_size2, iteration = 1000, learning_rate = 0.001):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.iteration = iteration
        self.learning_rate = learning_rate
        
        self.W1 = np.random.randn(self.input_size, self.hidden_size1)
        self.b1 = np.zeros((1, self.hidden_size1))
        self.W2 = np.random.randn(self.hidden_size1, self.hidden_size2)
        self.b2 = np.zeros((1, self.hidden_size2))
        self.W3 = np.random.rand(self.hidden_size2, self.output_size)
        self.b3 = np.zeros((1, self.output_size))
    
    def relu(self, z):
        return np.maximum(0,z)
    
    def derivative_relu(self, z):
        return (z > 0).astype(float)
        
    def sigmoid(self, x):
        return (1 /(1 + np.exp(-x)))
    
    def derivation_sigmoid(self, z):
        return z * (1-z)
    
    def forward(self, X):
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = self.relu(self.Z1)
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = self.relu(self.Z2)
        self.Z3 = np.dot(self.A2 , self.W3) + self.b3
        self.A3 = self.sigmoid(self.Z3)
        return self.A3
    
    def backward(self, X, y, output):
        n_samples = X.shape[0]
        dZ3 = output - y 
        dW3 = (1/n_samples) * np.dot(self.A2.T, dZ3)
        db3 = (1/n_samples) * np.sum(dZ3, axis = 0, keepdims= True)
        
        dZ2 = np.dot(dZ3, self.W3.T) * self.derivative_relu(dZ3)
        dW2 = (1/n_samples) * np.dot(self.A1.T, dZ2)
        db2 = (1/n_samples) * np.sum(dZ2, axis=0, keepdims=True)
        
        dZ1 = np.dot(dZ2, self.W2.T) * self.derivative_relu(dZ2)
        dW1 = (1/n_samples) * np.dot(X.T, dZ1)
        db1= (1/n_samples) * np.sum(dZ1, axis= 0 , keepdims= True)
        
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
        self.W3 -= self.learning_rate * dW3
        self.b3 -= self.learning_rate * db3
        
    def fit(self, X, y):
        self.loss_history = []
        for _ in range(self.iteration):
            output = self.forward(X)
            self.loss_history.append(np.mean(output-y)**2)
            self.backward(X, y, output)
    
    def plot_loss(self):
        plt.plot(self.loss_history)
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.title("Training Loss Over Time")
        plt.show()
    
    def predict(self, X):
        pred = self.forward(X)
        return np.where(pred > 0.5, 1, 0)
        