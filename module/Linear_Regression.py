import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd

class LinearRegression:

    def __init__(self):
        
        self.betta = None
        self._loss = None

    def fit_least_squares(self, X, Y):
        
        X = np.insert(X,0,1,axis=1)
        self.betta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)
        
    def fit_svd(self, X, Y):
        
        X = np.insert(X,0,1,axis=1)
        # Singular Value Decomposition
        U, S, V_T = np.linalg.svd(X, full_matrices = False)
        self.betta = V_T.T.dot(np.linalg.inv(np.diag(S))).dot(U.T).dot(Y)
        
    def fit_gradient_descent(self, X, Y, n_iter = 10000, learning_rate = 0.01):
        
        X = np.insert(X,0,1,axis=1)
        betta_old = np.random.randn(2,1)
        self.betta = betta_old + 1
        iteration = 0
        while np.linalg.norm(betta_old - self.betta) > 0.15 or iteration < n_iter:
            betta_old = self.betta
            grad = 2/len(Y) * X.T.dot(X.dot(betta_old) - Y)
            self.betta = betta_old - learning_rate * grad
            iteration += 1
            
    def fit_maximum_likelihood(self, X, Y):
        # Same as Least Squares.
        X = np.insert(X,0,1,axis=1)
        self.betta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)
        

    def predict(self, X):
        
        self.Y_pred = self.betta[0] + self.betta[1] * X
        
        return self.Y_pred
    
    def loss(self, Y):
        
        self._loss = np.sum((self.Y_pred - Y)**2) / len(Y)
        
        return self._loss
    
    def plot(self, X, Y, Y_pred):
        fig, axes = plt.subplots()
        axes.scatter(X,Y)
        axes.plot(X,Y_pred,'r')
