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
    
    