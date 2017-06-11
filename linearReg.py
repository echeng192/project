"""
Created on Fri Jun 09 10:20:05 2017
@author: Paul
"""

import numpy as np

def sigmoid(inX):
    return 1.0/(1+np.exp(-inX))

def gradDecsent(dataIn,classLabels):
    dataMat = np.mat(dataIn)
    labelMat = np.mat(classLabels).transpose()
    m,n = np.shape(dataMat)
    alpha = 0.001
    maxClycles =500
    weights = np.ones((n,1))
    for k in range(maxClycles):
        h = dataMat * weights
        error = ( h -labelMat)
        weights = weights - alpha * dataMat.transpose() * error
    return weights

def classify(inVect,weights):
    prob = sigmoid(sum(np.mat(inVect) * weights))
    if prob > 0.5:
        return 1
    else:
        return 0
def predictValue(inX, weights):
    print inX, weights
    return (np.mat(inX) * weights)
def test():
    dataset = [[1, 1, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1], [0, 0, 0], [0, 0, 1]]
    labels = [1, 1, 2, 3, 0, 0]
    weight = gradDecsent(dataset,labels)
    a = predictValue([7, 8, 9.1],weight)
    print a

test()