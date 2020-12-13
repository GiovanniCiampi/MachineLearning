import random
from math import log
import numpy as np

class LogisticRegression:
    def __init__(self, dim):
        self.dim = dim
        self.params = np.array([0.01 for i in range(dim+1)])
        
    
    def fit(self, X, y, learning_rate, epochs, add_bias=True):
        y = np.array(y)
        if add_bias:
            X = self.__add_bias(X)
            
        for e in range(epochs):
            gradients = self.__get_gradient(X, y)
            self.params = self.params - learning_rate*gradients
            
        return self.get_cost(X, y)

    
    def __get_gradient(self, X, y):
        n = X.shape[0]
        all_preds = self.predict(X, add_bias = False)
        return (1 / n) * np.dot(X.T,  all_preds - y)

        
    def predict(self, X, add_bias=True, thresholded=False, t=.5):
        if add_bias:
            X = self.__add_bias(X)
        preds = np.array([self.__predict_point(x) for x in X])
        if thresholded:
            return [1 if p>=t else 0 for p in preds]
        return preds
         
        
    def __predict_point(self, x):
        net = np.dot(x, self.params)
        return self.__sigmoid(net)
    
    
    def __sigmoid(self, x):
        return 1. / (1. + np.exp(-x))
    
    def __add_bias(self, X):
        return np.insert(X.reshape(-1, 1), 0, np.ones(X.shape[0]), axis=1)
            
        
    def get_cost(self, X, y, add_bias=False):
        n = X.shape[0]
        s = 0
        if add_bias:
            X=self.__add_bias(X)
        
        for x, t in zip(X, y):
            prediction = self.__predict_point(x)
            prediction = self.__get_math_safe_prediction(prediction)
            s += t*(log(prediction)) + (1-t)*(log(1-prediction))
        return -1/n*s
    
    
    def get_params(self):
        return self.params
    
    
    def __get_math_safe_prediction(self, p, adj=0.00001):
        if p == 0: 
            return p+adj
        elif p ==1: 
            return p-adj
        else: 
            return p
