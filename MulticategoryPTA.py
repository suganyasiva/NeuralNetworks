# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 21:35:16 2017

@author: Suganya
"""
import idx2numpy
import numpy as np
import numpy
import csv
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
    


"""
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
"""

def trainModel(W, n, eta, epsilon, img_data, img_label):
    epoch = 0
    errors = []
    induced_local_fields = []
    while(True):
        errors.insert(epoch, 0)
        #print(epoch,':', errors[epoch])
        for i in range(0,n):  
            v = numpy.matmul(W,numpy.array(img_data[i]))
            ilf = list(v)
            induced_local_fields.append(ilf)
            label = ilf.index(max(list(ilf)))
            if(label != img_label[i]):
                errors[epoch] = errors[epoch] + 1
    
        #print(epoch,':', errors[epoch])    
        epoch = epoch + 1    
    
        for i in range(0,n):
            step = [1 if k >= 0 else 0 for k in induced_local_fields[i]]        #step function for each component in induced_local_field value
            dx = numpy.array(desired_outputs[img_label[i]])
            eta_dx = numpy.multiply(eta, numpy.subtract(dx, numpy.array(step)))
            xi = numpy.array(img_data[i])
            delta = numpy.dot(eta_dx[:,None], xi[None,:])
            W = numpy.add(W, delta)
        
        induced_local_fields = []
        if(errors[epoch - 1]/n <= epsilon):
            break
    return W, errors
        
def testModel(W, img_test_data, img_test_label):
    errors = 0  
    induced_local_fields = []
    for i in range(0,10000):  
        v = numpy.matmul(W,numpy.array(img_test_data[i]))
        ilf = list(v)
        induced_local_fields.append(ilf)
        label = ilf.index(max(list(ilf)))
        if(label != img_test_label[i]):
            errors = errors + 1
    
    print('Number of misclassifications on Test data :',errors)
    print('Percentage of misclassified test data samples: ', errors*100/10000)
    
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
    
def plotEpochNumberVersusMisclassifications(errors):
    plt.ylabel('No of Misclassifications')
    plt.xlabel('Epoch Number')
    plt.plot(list(range(0,len(errors))), errors, color = 'red')
    plt.show()
    
def trainAndTest(W, n, eta, epsilon, img_data, img_label, img_test_data, img_test_label):
    print("For N = ", n)
    print("Eta = ", eta)
    print("Epsilon = ", epsilon)
    #Step d
    W, errors = trainModel(W, n, eta, epsilon, img_data, img_label)
    plotEpochNumberVersusMisclassifications(errors)
    #Step e
    testModel(W, img_test_data, img_test_label)

 
W = numpy.random.uniform(-1, 1.0, size = (10,784))
W1 = W
img_data, img_label, img_test_data, img_test_label = prepData()
    
    
n = 50
eta = 1
epsilon = 0
trainAndTest(W, n, eta, epsilon, img_data, img_label, img_test_data, img_test_label)

n = 1000
eta = 1
epsilon = 0
W = W1
trainAndTest(W, n, eta, epsilon, img_data, img_label, img_test_data, img_test_label)


n = 60000
eta = 1
epsilon = 0.12
W = W1
trainAndTest(W, n, eta, epsilon, img_data, img_label, img_test_data, img_test_label)
"""
"""
n = 60000
eta = 1
epsilon = 0.2
trainAndTest(W, n, eta, epsilon, img_data, img_label, img_test_data, img_test_label)
    
W = numpy.random.uniform(-1, 1.0, size = (10,784))
trainAndTest(W, n, eta, epsilon, img_data, img_label, img_test_data, img_test_label)

W = numpy.random.uniform(-1, 1.0, size = (10,784))
trainAndTest(W, n, eta, epsilon, img_data, img_label, img_test_data, img_test_label)