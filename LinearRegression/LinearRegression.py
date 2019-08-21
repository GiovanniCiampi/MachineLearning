import random

class LinearRegression:
    def __init__(self):
        self.m = random.uniform(0.0, 1.0)
        self.a = random.uniform(0.0, 1.0)
    
    def fit(self, data, learning_rate, epochs):
        for i in range(epochs):
            self.m = self.m - learning_rate * self.__dm__(data)
            self.a = self.a - learning_rate * self.__da__(data)
            
        return self.get_mse(data)
        
    def predict(self, x):
        return self.m * x + self.a
        
        
    def __dm__(self, data):
        n = len(data)
        s = 0
        
        for point in data:
            s += (self.predict(point[0]) - point[1]) * point[0]
        return 2/n*s
        
        
    def __da__(self, data):
        n = len(data)
        s = 0
        
        for point in data:
            s += self.predict(point[0]) - point[1]   
        return 2/n*s
        
    def get_mse(self, data):
        n = len(data)
        s = 0
        
        for point in data:
            s += (self.predict(point[0]) - point[1])**2
        return s/n
    
    def get_params(self):
        return self.m, self.a