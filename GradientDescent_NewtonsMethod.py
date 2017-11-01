# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 23:05:43 2017

@author: Suganya
"""

import numpy
from scipy.spatial import distance
import matplotlib.pyplot as plt
import time

def plotTrajectory(x,y, f):
    plt.ylabel('Y')
    plt.xlabel('X')
    plt.plot(x, y, color = 'red', label='Trajectory')
    plt.legend(bbox_to_anchor=(1,1), loc='upper left')
    plt.show()
    plt.ylabel('Energies')
    plt.xlabel('Epochs')
    plt.scatter(range(0,len(f)), f, color = 'green', label = 'Energies')
    plt.legend(bbox_to_anchor=(1,1), loc='upper left')
    plt.show()
    
def gradientDescent(w, dist, epsilon, eta):
    while(dist > epsilon):
        if(w[0]+w[1] < 1 and w[0] > 0 and w[1] > 0):
            wPrev = w
            f = -numpy.log(1-w[0]-w[1]) - numpy.log(w[0]) - numpy.log(w[1])
            flist.append(f)
            g = numpy.array([(1/(1-w[0]-w[1])) - (1/w[0]), (1/(1-w[0]-w[1])) - (1/w[1])])
            etaG = numpy.multiply(eta,g)
            w = numpy.subtract(w, etaG)
            xlist.append(w[0])
            ylist.append(w[1])
            dist = distance.euclidean(wPrev, w)
        else:
            break
    plotTrajectory(xlist,ylist, flist)
    
def NewtonsMethod(w, dist, epsilon, eta):
    xlist = []
    ylist = []
    flist = []
    while(dist > epsilon):
        if(w[0]+w[1] < 1 and w[0] > 0 and w[1] > 0):
            wPrev = w
            f = -numpy.log(1-w[0]-w[1]) - numpy.log(w[0]) - numpy.log(w[1])
            flist.append(f)
            g = numpy.array([(1/(1-w[0]-w[1])) - (1/w[0]), (1/(1-w[0]-w[1])) - (1/w[1])])
            H = numpy.array([[(1/((1-w[0]-w[1])**2)) + (1/(w[0]**2)), (1/((1-w[0]-w[1])**2))],[(1/((1-w[0]-w[1])**2)), (1/((1-w[0]-w[1])**2)) + (1/(w[1]**2))]])
            HInverse = numpy.linalg.inv(H)
            etaHInverseG = numpy.dot(numpy.multiply(eta,HInverse), g)
            w = numpy.subtract(w, etaHInverseG)
            xlist.append(w[0])
            ylist.append(w[1])
            dist = distance.euclidean(wPrev, w)
        else:
            break
    plotTrajectory(xlist,ylist, flist)
    
    
w = numpy.array([0.1,0.4])
eta = 0.01
epsilon = 0.0001
dist = 1
xlist = []
ylist = []
flist = []

print("Initial Point: ", w)
print("Learning Parameter (eta): ", eta)

t0 = time.time()
print("Gradient Descent")
gradientDescent(w, dist, epsilon, eta)
t1 = time.time()
print("Newton's Method")
epsilon = 0.0001
NewtonsMethod(w, dist, epsilon, eta)
t2 = time.time()

print("Time taken for Gradient Descent: ", t1-t0, " seconds")
print("Time taken for Newton's method: ", t2-t1, " seconds")