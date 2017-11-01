# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 19:40:48 2017

@author: Suganya
"""

import random
import numpy
import matplotlib.pyplot as plt

def plotGraph(S0, S1, weights):
    w0 = weights[0]
    w1 = weights[1]
    w2 = weights[2]

    #Solving eqn w0+w1x1+w2x2 = 0
    #if x1 = 1
    p1 = [1]
    x2 = (-w0 - w1) / w2
    p1.append(x2)
    #if x2 = 1
    p2 = [1]
    x1 = (-w0 - w2) / w1
    p2.insert(0, x1)
    #if x1 = -1
    p3 = [-1]
    x2 = (-w0 + w1) / w2
    p3.append(x2)
    #if x2 = -1
    p4 = [-1]
    x1 = (-w0 + w2) / w1
    p4.insert(0, x1)
    
    data = numpy.array([p1,p2,p3,p4])
    k, l = data.T
    
    plt.ylabel('x2')
    plt.xlabel('x1')
    
    if(len(S1) != 0):
        data = numpy.array(S1)
        x, y = data.T
        plt.scatter(x,y,color='blue', label='S1')
    if(len(S0) != 0):
        data = numpy.array(S0)
        x, y = data.T
        plt.scatter(x,y,color='green', label='S0')
        
    plt.plot(k, l, color = 'red', label='Boundary')
    plt.legend(bbox_to_anchor=(1,1), loc='upper left')
    plt.show()
    
def plotEpochNumberVersusMisclassifications(epochCounts, misClassifications):
    plt.ylabel('No of Misclassifications')
    plt.xlabel('Epoch Number')
    plt.plot(epochCounts, misClassifications, color = 'red')
    plt.show()
    
def classify(S, w):
    set1 = []
    set0 = []
    count = 0
    weightsTranspose = w.transpose()
    for i in S:
        count += 1
        X = [1] + list(i)
        weightedSum = numpy.matmul(X, weightsTranspose)
        if(weightedSum >= 0):
            set1.append(list(i))
        elif(weightedSum < 0):
            set0.append(list(i))
    return set0, set1
    
def generateWeights():
    w0 = float("%.2f" % random.uniform(-1/4,1/4))
    w1 = float("%.2f" % random.uniform(-1,1))
    w2 = float("%.2f" % random.uniform(-1,1))
    weights = numpy.array([ float("%.2f" % elem) for elem in[w0, w1,w2]])
    print('Optimal Weights [w0, w1, w2] = ' , list(weights))
    return weights
    
def generateWeightsPrime():
    w0Prime = float("%.2f" % random.uniform(-1,1))
    w1Prime = float("%.2f" % random.uniform(-1,1))
    w2Prime = float("%.2f" % random.uniform(-1,1))
    weightsPrime = numpy.array([ float("%.2f" % elem) for elem in [w0Prime, w1Prime,w2Prime]])
    print('[W0\', W1\', W2\'] = ' , list(weightsPrime))
    return weightsPrime
    
def generateInputs(n):
    S = [] 
    for i in range(0,n):
        S.append([float("%.2f" % random.uniform(-1,1)), float("%.2f" % random.uniform(-1,1))])
    return S
    
def calculateMisclassifications(s, sPrime, desired, actual, w, trainingParameter):
    misClassifications = 0
    omega = w
    if(len(s) != 0):
        for i in s:
            if(i not in sPrime):
                X = [1] + i
                omega = numpy.array([ float("%.2f" % elem) for elem in list(numpy.array(w).transpose() + numpy.array(X).transpose() * (desired - actual) * trainingParameter)])
                misClassifications += 1
                w = omega
    return misClassifications, omega

def perceptronTrainingAlgorithm(S, S0, S1, weightsPrime, trainingParameter):
    epochCounts = []
    misClassifications = []
    S0Prime, S1Prime = classify(S, weightsPrime)
    s0Misclassifications, o = calculateMisclassifications(S0, S0Prime, 0, 1, numpy.array(weightsPrime), trainingParameter)
    s1Misclassifications, o = calculateMisclassifications(S1, S1Prime, 1, 0, numpy.array(o), trainingParameter)
    totalMisclassifications = s0Misclassifications + s1Misclassifications
    print('Number of Misclassifications using W\' (epoch 1): ' + str(totalMisclassifications))
    print('[W0\'\',W1\'\',W2\'\']', [ float("%.2f" % elem) for elem in list(o)])
    epochCounts.append(1)
    misClassifications.append(totalMisclassifications)
    
    epoch =2
    while(totalMisclassifications != 0):
        S0Prime = []
        S1Prime = []
        S0Prime, S1Prime = classify(S, o)
        s0Misclassifications, o = calculateMisclassifications(S0, S0Prime, 0, 1, numpy.array(o), trainingParameter)
        s1Misclassifications, o = calculateMisclassifications(S1, S1Prime, 1, 0, numpy.array(o), trainingParameter)
        totalMisclassifications = s0Misclassifications + s1Misclassifications
        epochCounts.append(epoch)
        misClassifications.append(totalMisclassifications)
        plotGraph(S0, S1, o)
        epoch += 1
    print("Number of Epochs: ", epoch - 1)
    print('Final Weights = ', [ float("%.2f" % elem) for elem in list(o)])
    plotEpochNumberVersusMisclassifications(epochCounts, misClassifications)    #Step k
    
def runProgramForInputN(n):
    S = [] 
    S0 = []
    S1 = []
    weights = generateWeights()   #Steps b,c,d

    S = generateInputs(n)          #Step f
    
    S0, S1 = classify(S, weights)  #Steps g,h      
    plotGraph(S0, S1, weights)     #Step i
    
    #Step j
    trainingParameter = 1
    weightsPrime = generateWeightsPrime()
    
    print('For trainingParameter = 1: ')
    perceptronTrainingAlgorithm(S, S0, S1, weightsPrime, trainingParameter)
    
    #Step l
    trainingParameter = 10
    print('For trainingParameter = 10: ')
    perceptronTrainingAlgorithm(S, S0, S1, weightsPrime, trainingParameter)
    
    #Step m
    trainingParameter = 0.1
    print('For trainingParameter = 0.1: ')
    perceptronTrainingAlgorithm(S, S0, S1, weightsPrime, trainingParameter)
            
    

n = 100
print("For N value = ", n)
runProgramForInputN(n)

#Step p
n = 1000
print("For N value = ", n)
runProgramForInputN(n)





    
