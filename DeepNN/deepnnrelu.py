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
    
   