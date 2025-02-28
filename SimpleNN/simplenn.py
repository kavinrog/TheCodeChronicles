import random
import math

def sigmoid(x):
  return 1 / 1 +(math.exp(-x))

def sigmoid_derivative(x):
  return x * (1 - x)

class SimpleNN:
  def __init__(self, input_size, hidden_size, output_size):
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.output_size = output_size
    
    self.weights_input_hidden = [[random.uniform(-1, 1) for _ in range(hidden_size)] for _ in range(input_size)]
    self.weights_hidden_output = [[random.uniform(-1, 1) for _ in range(output_size)] for _ in range(hidden_size)]
    
    self.bias_hidden = [random.uniform(-1, 1) for _ in range(hidden_size)]
    self.bias_output = [random.uniform(-1, 1) for _ in range(output_size)]   

def dot_product(self, vector1, vector2):
    return (x * y for x, y in zip(vector1, vector2))

def feedforward(self, inputs):
    hidden_layer = [sigmoid(self.dot_product(inputs, [self.weights_input_hidden[j][i] for j in range(self.input_size)]) + self.bias_hidden[i]) for i in range(self.hidden_size)]
    output_layer = [sigmoid(self.dot_product(hidden_layer, [self.weights_hidden_output[j][i] for j in range(hidden_size)]) + self.bias_ouput[i]) for i in range(self.output_size)]
    return hidden_layer, output_layer
