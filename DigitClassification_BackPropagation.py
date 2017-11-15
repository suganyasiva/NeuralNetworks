# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 16:32:54 2017

@author: Suganya
"""

import numpy as np
from numpy import linalg as la
import idx2numpy
import csv
from sklearn.preprocessing import normalize
import sys
import matplotlib.pyplot as plt

desired_outputs = {0 : [1,0,0,0,0,0,0,0,0,0],
                   1 : [0,1,0,0,0,0,0,0,0,0], 
                   2 : [0,0,1,0,0,0,0,0,0,0], 
                   3 : [0,0,0,1,0,0,0,0,0,0], 
                   4 : [0,0,0,0,1,0,0,0,0,0], 
                   5 : [0,0,0,0,0,1,0,0,0,0], 
                   6 : [0,0,0,0,0,0,1,0,0,0], 
                   7 : [0,0,0,0,0,0,0,1,0,0], 
                   8 : [0,0,0,0,0,0,0,0,1,0], 
                   9 : [0,0,0,0,0,0,0,0,0,1] }

def plotPoints(epochs,trainingMisclassifications, testMisclassifications, xlabel, ylabel, labelbox1, labelbox2):
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(epochs, trainingMisclassifications, 'or', color = 'magenta', label=labelbox1)
    plt.plot(epochs, testMisclassifications, 'or', color = 'green', label=labelbox2)
    plt.legend(bbox_to_anchor=(1,1), loc='upper left')
    plt.show()

def prepData():
    img_data = []
    img_label = []
    f = open ( 'mnist_training_images.csv', 'r' )
    for line in f.readlines():
        cleanline = line.replace("\n","")
        row = cleanline.split(",")
        row = [int(i) for i in row]
        img_data.append(row)
    f.close()
    f = open ( 'mnist_training_labels.csv', 'r' )
    for line in f.readlines():
        cleanline = line.replace("\n","")
        img_label.append(int(cleanline))
    f.close()
    
    img_test_data = []
    img_test_label = [] 
    f = open ( 'mnist_test_images.csv', 'r' )
    for line in f.readlines():
        cleanline = line.replace("\n","")
        row = cleanline.split(",")
        row = [int(i) for i in row]
        img_test_data.append(row)
    f.close()
    f = open ( 'mnist_test_labels.csv', 'r' )
    for line in f.readlines():
        cleanline = line.replace("\n","")
        img_test_label.append(int(cleanline))   
    f.close()
    return img_data, img_label, img_test_data, img_test_label
    
def feedForward(x, w1,w2,b1,b2):
    temp1 = np.dot(w1.T,np.asarray(x).reshape((784,1)))
    a = np.array([temp1[i] + b1[i] for i in range(0,layer1_neurons)]).reshape((layer1_neurons,1))
    z = np.tanh(a)
    g = np.dot(w2.T,z) + b2
    y = np.tanh(g)
    return temp1, a,z,g,y
    
def activationPrime(k):
    return np.array(1 - np.tanh(k)**2)
    
def backPropagation(x, d, y, w1, w2, b1, b2, a, z, g, b1Inputs, b2Inputs):
    #use n below
    e = ((-2/n) * np.subtract(d,y.reshape((10,)))).reshape((10,1))                  
    deltab2i = np.asarray([e[i]*b2Inputs[i]*activationPrime(g)[i] for i in range(0,len(b2Inputs))]).reshape((layer2_neurons,1))
    deltab1i = np.dot(w2, np.asarray([e[i]*activationPrime(g)[i]*activationPrime(a)[i] for i in range(0,len(activationPrime(g)))]).reshape((10,1)))
    deltaw2i = np.dot(np.asarray([e[i]*activationPrime(g)[i] for i in range(0,len(activationPrime(g)))]).reshape((10,1)), z.T)
    deltaw1i = np.dot(np.dot(w2, np.asarray([e[i]*activationPrime(g)[i]*activationPrime(a)[i] for i in range(0,len(activationPrime(g)))]).reshape((10,1))), np.asarray(x).reshape((1,784))).T
    return deltab1i, deltab2i, deltaw1i, deltaw2i
    
def distanceFunction(desired, y):
    total = 0
    size = len(desired)
    for i in range(0,size):
        total += la.norm(np.asarray(desired[i]) - np.asarray(y[i]))**2
    distance = total / size
    return distance, total
    

n = 60000
input_neurons = 784
no_of_layers = 2
layer1_neurons = 10
layer2_neurons = 10
eta = 5

w1 = np.random.uniform(0.001, 1, size = (input_neurons, layer1_neurons))
b1 = np.random.uniform(0.001, 1, size = (layer1_neurons,1))
w2 = np.random.uniform(0.001, 1, size = (layer1_neurons,layer2_neurons))
b2 = np.random.uniform(0.001, 1, size = (layer2_neurons,1))

b1Inputs = np.asarray([1 for i in range(0,layer1_neurons)]).reshape((layer1_neurons, 1))
b2Inputs = np.asarray([1 for i in range(0,layer2_neurons)]).reshape((layer2_neurons, 1))

trainImages = idx2numpy.convert_from_file('train-images.idx3-ubyte')
trainLabels = idx2numpy.convert_from_file('train-labels.idx1-ubyte')
testImages = idx2numpy.convert_from_file('t10k-images.idx3-ubyte')
testLabels = idx2numpy.convert_from_file('t10k-labels.idx1-ubyte')


images = np.concatenate([trainImages.reshape(60000,784)])
labels = np.concatenate([trainLabels])

csvfile = open('mnist_training_imgs.csv', 'w')
writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
for row in images:
    writer.writerow(row)
csvfile.close()
    
with open('mnist_training_imgs.csv') as input, open('mnist_training_images.csv', 'w') as output:
    non_blank = (line for line in input if line.strip())
    output.writelines(non_blank)
input.close()
output.close()


with open('mnist_training_labels.csv', 'w') as csvfile:
    writer = csv.writer(csvfile, delimiter='\n', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(labels)
csvfile.close()

 
images = np.concatenate([testImages.reshape(10000,784)])
labels = np.concatenate([testLabels])
 
csvfile = open('mnist_test_imgs.csv', 'w')
writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
for row in images:
    writer.writerow(row)
csvfile.close()
    
with open('mnist_test_imgs.csv') as input, open('mnist_test_images.csv', 'w') as output:
    non_blank = (line for line in input if line.strip())
    output.writelines(non_blank)
input.close()
output.close()

with open('mnist_test_labels.csv', 'w') as csvfile:
    writer = csv.writer(csvfile, delimiter='\n', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(labels)
csvfile.close()

img_data, img_label, img_test_data, img_test_label = prepData()

epoch = 0
epochs = []
Distances = []
DistancesTest = []
trainingMisclassifications = []
testMisclassifications = []

img_data_norm = normalize(img_data)
img_test_data_norm = normalize(img_test_data)
testDataLength = len(img_test_data_norm)
while(True):
    y = []
    desired = []
    yTest = []
    desiredTest = []
    list_a = []
    epoch += 1
    misclassifications = 0
    print("epoch", epoch)
    for i in range(0,n):
        temp1, a,z,g,yi = feedForward(np.asarray(img_data_norm[i])[:,None], w1,w2,b1,b2)
        list_a.append(a)
        d = desired_outputs[img_label[i]]
        deltab1, deltab2, deltaw1, deltaw2 = backPropagation(img_data_norm[i], np.asarray(d), yi, w1, w2, b1, b2, a, z, g, b1Inputs, b2Inputs)
        w1 = np.subtract(np.asarray(w1), np.dot(eta, np.asarray(deltaw1)))
        w2 = np.subtract(np.asarray(w2), np.dot(eta, np.asarray(deltaw2)))
        b1 = np.subtract(np.asarray(b1), np.dot(eta, np.asarray(deltab1)))
        b2 = np.subtract(np.asarray(b2), np.dot(eta, np.asarray(deltab2)))
        idx = list(yi.reshape((10,))).index(max(list(yi.reshape((10,)))))
        if idx != img_label[i]:
            misclassifications += 1
        yj = [1 if i == idx else 0 for i in range(0,10)]
        y.append(yj)
        desired.append(d)
    distance, total = distanceFunction(desired, y)
    epochs.append(epoch)
    Distances.append(distance)
    print("distance", distance)
    print("total", total)
    print("misclassifications", misclassifications)
    trainingMisclassifications.append(misclassifications)
    trainingAccuracy = (n-misclassifications)*100/n
    print("Training Accuracy", trainingAccuracy)
    
    misclassifications = 0
    for i in range(0,testDataLength):
        temp2, a,z,g,yi = feedForward(np.asarray(img_test_data_norm[i])[:,None], w1,w2,b1,b2)
        d = desired_outputs[img_test_label[i]]
        idx = list(yi.reshape((10,))).index(max(list(yi.reshape((10,)))))
        if idx != img_test_label[i]:
            misclassifications += 1
        yj = [1 if i == idx else 0 for i in range(0,10)]
    desiredTest.append(d)
    yTest.append(yj)
    distanceTest, total = distanceFunction(desiredTest, yTest)
    DistancesTest.append(distanceTest)
    testAccuracy = (testDataLength-misclassifications)*100/testDataLength
    testMisclassifications.append(misclassifications)
    print("Test Accuracy", testAccuracy)
    
    if(testAccuracy > 95):
        break
    
plotPoints(epochs,trainingMisclassifications, testMisclassifications, 'No of Epochs', 'Misclassifications', 'Training Misclassifications', 'Test Misclassifications')
plotPoints(epochs,Distances, DistancesTest, 'No of Epochs', 'Energies', 'Training Energy', 'Test Energy')