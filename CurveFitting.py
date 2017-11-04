# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 10:18:18 2017

@author: Suganya
"""

import random
import math
import matplotlib.pyplot as plt
import numpy as np

def feedForward(w1, w2, b1, b2, xi):
    ai = np.add(np.dot(np.asarray(w1)[:,None],np.array([xi])), b1)
    zi = [np.tanh(j) for j in ai]
    yi = np.add(np.dot(np.asarray(w2),np.asarray(zi)[:,None]), b2)
    return ai.tolist(), zi, yi.tolist()[0]

def feedForwardBatch(x, w1, w2, b1, b2):
    a = []
    z = []
    y = []
    for i in x:
        ai = np.add(np.dot(np.asarray(w1)[:,None],np.array([i])), b1)
        a.append(ai.tolist())
        zi = [np.tanh(j) for j in ai]
        z.append(zi)
        yi = np.add(np.dot(np.asarray(w2),np.asarray(zi)[:,None]), b2)
        y.append(yi.tolist()[0])
    return a, z, y
    
def backPropagation(x, d, y, w1, w2, b1, b2, a, z, i):
    deltab2i = [((-2/n) * (d[i] - y))]
    deltab1i = [(-2/n) * (d[i] - y)*w2[u]*(1 - np.tanh(a[u])**2) for u in range(0,numberOfNeurons)]
    deltaw2i = [(-2/n)*u*(d[i] - y) for u in z]
    deltaw1i = [((-2*x[i])/n)*(d[i] - y)* w2[u]*(1 - np.tanh(a[u])**2) for u in range(0,numberOfNeurons)]
    return deltab1i, deltab2i, deltaw1i, deltaw2i

def plotPoints(x,d):
    plt.ylabel('Desired Output')
    plt.xlabel('X')
    plt.plot(x, d, 'or', color = 'red', label='Points')
    plt.legend(bbox_to_anchor=(1,1), loc='upper left')
    plt.show()
    
def plotPointsAndFit(x, y, d):
    plt.ylabel('Desired Output')
    plt.xlabel('X')
    plt.plot(x, d, 'or', color = 'red', label='Points')
    plt.plot(x, y, 'or', color = 'blue', label='Fit')
    plt.legend(bbox_to_anchor=(1,1), loc='upper left')
    plt.show()

def meanSquaredError(d, y):
    total = 0
    for i in range(0,n):
        total += (d[i] - y[i])**2
    mse = total / n
    return mse
    
n = 300
eta = 0.2
numberOfNeurons = 24
    
x = [random.uniform(0,1) for i in range(0,n)]
v = [random.uniform(-0.1,0.1) for i in range(0,n)]
d = [math.sin(20*x[i])+(3*x[i])+v[i] for i in range(0,n)]
plotPoints(x,d)

w1 = [random.uniform(-10,10) for i in range(0,numberOfNeurons)]
w2 = [random.uniform(-7,8) for i in range(0,numberOfNeurons)]
b1 = [random.uniform(-7,4) for i in range(0,numberOfNeurons)]
b2 = [random.uniform(-1,1) for i in range(0,1)]
 
epoch = 1
epochs = []
MeanSquaredErrors = []

while(True):
    y = []
    for i in range(0, n):
        a, z, yi = feedForward(w1, w2, b1, b2, x[i])
        deltab1, deltab2, deltaw1, deltaw2 = backPropagation(x, d, yi, w1, w2, b1, b2, a, z, i)
        w1 = np.subtract(np.asarray(w1), np.dot(eta, np.asarray(deltaw1))).tolist()
        w2 = np.subtract(np.asarray(w2), np.dot(eta, np.asarray(deltaw2))).tolist()
        b1 = np.subtract(np.asarray(b1), np.dot(eta, np.asarray(deltab1))).tolist()
        b2 = np.subtract(np.asarray(b2), np.dot(eta, np.asarray(deltab2))).tolist()
        y.append(yi)
    mse = meanSquaredError(d, y)
    epochs.append(epoch)
    epoch += 1
    MeanSquaredErrors.append(mse)
    print("epoch", epoch)
    print("mse", mse)
    if(mse < 0.01):
        break

plotPoints(epochs,MeanSquaredErrors)
a, z, y1 = feedForwardBatch(x, w1, w2, b1, b2)
plotPointsAndFit(x,y1,d)
