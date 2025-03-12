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
        
        