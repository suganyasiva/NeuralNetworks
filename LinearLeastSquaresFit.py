# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 21:28:25 2017

@author: Suganya
"""
import random
import numpy as np
import matplotlib.pyplot as plt

def pseudoInverse(X):
    XXT = np.matmul(X, X.T)
    inverse = np.linalg.inv(XXT)
    pseudoInverse = np.dot(X.T, inverse)
    #print(pseudoInverse)
    return pseudoInverse

def plotPoints(x,y,Y):
    plt.ylabel('Y')
    plt.xlabel('X')
    plt.plot(x, y, 'or', color = 'red', label='Points')
    plt.plot(x, Y, color = 'blue', label='Linear Least Squares Fit')
    plt.legend(bbox_to_anchor=(1,1), loc='upper left')
    plt.show()

x = [i for i in range(1,51)]
y = [i + random.uniform(-1,1) for i in range(1,51)]
A = [[1 for i in range(0,50)],x]
pseudoInverse = pseudoInverse(np.array(A))
W = np.matmul(np.array(y), pseudoInverse)
Y = [W[0] + W[1]*x[i] for i in range(0,50)]
plotPoints(x,y,Y)

