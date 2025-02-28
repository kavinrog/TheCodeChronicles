import random
import math

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

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
        return sum(x * y for x, y in zip(vector1, vector2))

    def feedforward(self, inputs):
        hidden_layer = [sigmoid(self.dot_product(inputs, [self.weights_input_hidden[j][i] for j in range(self.input_size)]) + self.bias_hidden[i]) for i in range(self.hidden_size)]
        output_layer = [sigmoid(self.dot_product(hidden_layer, [self.weights_hidden_output[j][i] for j in range(self.hidden_size)]) + self.bias_output[i]) for i in range(self.output_size)]
        return hidden_layer, output_layer

    def training(self, inputs, targets, learning_rate = 0.001, epochs = 1000):
        for epoch in range(epochs):
            for input_vector, target_vector in zip(inputs, targets):
                hidden_layer, output_layer = self.feedforward(input_vector)
                output_error = [target_vector[i] - output_layer[i] for i in range(self.output_size)]
                output_delta = [output_error[i] * sigmoid_derivative(output_layer[i]) for i in range(self.output_size)]
                
                hidden_error = [sum(self.weights_hidden_output[i][j] * output_delta[j] for j in range(self.output_size)) for i in range(self.hidden_size)]
                hidden_delta = [hidden_error[i] * sigmoid_derivative(hidden_layer[i]) for i in range(self.hidden_size)]
                
                for i in range(self.input_size):
                    for j in range(self.hidden_size):
                        self.weights_input_hidden[i][j] += learning_rate * hidden_delta[j] * input_vector[i]
                    self.bias_hidden[j] += learning_rate * hidden_delta[j]
                    
                for i in range(self.hidden_size):
                    for j in range(self.output_size):
                        self.weights_hidden_output[i][j] += learning_rate * output_delta[j] * hidden_layer[i]
                    self.bias_output[j] += learning_rate * output_delta[j]
            
inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
targets = [[0], [1], [1], [0]]

nn = SimpleNN(2, 2, 1)
nn.training(inputs, targets)

for input_vector in inputs:
    _, output = nn.feedforward(input_vector)
    print(f"Input: {input_vector} => Output: {round(output[0], 3)}")