import numpy as np 
import matplotlib.pyplot as plt

class DeepNNScratch:
    def __init__(self, input_size, output_size, hidden_size, iteration = 1000, learning_rate = 0.001):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.iteration = iteration
        self.learning_rate = learning_rate
        
        self.W1 = np.random.randn(self.input_size, self.hidden_size)
        self.b1 = np.zeros((1, self.hidden_size))
        self.W2 = np.random.randn(self.hidden_size, self.output_size)
        self.b2 = np.zeros((1, self.output_size))
    
    def sigmoid(self, x):
        return (1 /(1 + np.exp(-x)))
    
    def derivation_sigmoid(self, z):
        return z * (1-z)
    
    def forward(self, X):
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = self.sigmoid(self.Z1)
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = self.sigmoid(self.Z2)
        return self.A2
    
    def backward(self, X, y, output):
        n_samples = X.shape[0]
        
        dZ2 = output - y 
        dW2 = (1/n_samples) * np.dot(self.A1.T, dZ2)
        db2 = (1/n_samples) * np.sum(dZ2, axis = 0, keepdims= True)
        
        dZ1 = np.dot(dZ2, self.W2.T) * self.derivation_sigmoid(self.A1)
        dW1 = (1/n_samples) * np.dot(X.T, dZ1)
        db1 = (1/n_samples) * np.sum(dZ1, axis=0 , keepdims= True)
        
        self.W1 -= self.learning_rate * dW1        
        self.b1 -= self.learning_rate * db1      
        self.W2 -= self.learning_rate * dW2       
        self.b2 -= self.learning_rate * db2   
        
    def fit(self, X, y):
        self.loss_history = []  

        for _ in range(self.iteration):
            output = self.forward(X)
            loss = np.mean((y - output) ** 2)  
            self.loss_history.append(loss)
            self.backward(X, y, output)

    def plot_loss(self):
        plt.plot(self.loss_history)
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.title("Training Loss Over Time")
        plt.show()
    
    def predict(self, X):
        output = self.forward(X)
        return np.where(output > 0.5, 1, 0)
        
        
           
        
       
    
        
        