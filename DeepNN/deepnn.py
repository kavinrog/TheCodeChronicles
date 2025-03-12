import numpy as np 

class DeepNNScratch:
    def __init__(self, input_size, output_size, hidden_size, iteration = 1000, learning_rate = 0.001):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.iteration = iteration
        self.learning_rate = learning_rate
        
        self.W1 = np.random.randn(self.input_size, self.hidden_size)
        self.b1 = np.zeros(1, self.hidden_size)
        self.W2 = np.random.randn(self.hidden_size, self.output_size)
        self.b2 = np.zeros(1, self.output_size)
    
    def sigmoid(self, x):
        return (1 / 1 + np.exp(-x))
    
    def derivation_sigmoid(self, z):
        return z * (1-z)
    
    def forward(self, X):
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = self.sigmoid(self.Z1)
        self.Z2 = np.dot(X, self.W2) + self.b2
        self.A2 = self.sigmoid(self.Z2)
        return self.A2
       
    
        
        