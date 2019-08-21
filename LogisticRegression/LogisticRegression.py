import random
from math import log
import numpy as np

class LogisticRegression:
    def __init__(self, dim):
        self.dim = dim
        self.params = np.random.uniform(0.0, 1.0, dim+1)
        
    
    def fit(self, X, y, learning_rate, epochs):
        for e in range(epochs):
            for i in range(self.dim+1):
                d_i = self.__d_param_i__(X, y, i)
                self.params[i] = self.params[i] - learning_rate * d_i
            
        return self.get_cost(X, y)
    
        
    def predict(self, X):
        predictions = [] 
        for x in X:
            predictions.append(self.__predict_point__(x))
            
        return predictions
         
        
    def __predict_point__(self, x):
        prediction = self.params[0]
        for i in range(self.dim):
            #print(self.params, x, i)
            prediction += self.params[i+1] * x[i]  
            
        return self.__sigmoid__(prediction)
        
        
    def __d_param_i__(self, X, y, i):
        s=0
        
        for x, t in zip(X, y):
            prediction=self.__predict_point__(x)
            temp = prediction - t
            if i != 0:
                temp *= x[i-1]
            s += temp
        return s
    
    
    def __sigmoid__(self, x):
        return 1. / (1. + np.exp(-x))
            
        
    def get_cost(self, X, y):
        n = len(X)
        s = 0
        
        for x, t in zip(X, y):
            prediction = self.__predict_point__(x)
            s += t*(log(prediction)) + (1-t)*(log(1-prediction))
        return -1/n*s
    
    def get_params(self):
        return self.params