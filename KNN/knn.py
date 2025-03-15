import numpy as np 
from collections import Counter

class KNN:
    def __init__(self, k = 3):
        self.k = k 
        
    def fit(self, X, Y):
        self.X_train = X 
        self.Y_train = Y
        
    