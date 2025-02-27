import random
import math

def sigmod(x):
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
