"""
Created on Fri Jun 09 10:20:05 2017
@author: Paul
"""

import numpy as np

def sigmoid(inX):
    return 1.0/(1+np.exp(-inX))
def readData():
    dataMat = []
    labelMat = []
    fr = open('data/testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    
    return dataMat,labelMat

def gradAscent(dataIn,classLabels):
    dataMat = np.mat(dataIn)
    labelMat = np.mat(classLabels).transpose()
    m,n = np.shape(dataMat)
    alpha = 0.01
    maxClycles =500
    weights = np.ones((n,1))
    for k in range(maxClycles):
        h = sigmoid(dataMat * weights)
        error = (labelMat - h)
        weights = weights + alpha * dataMat.transpose() * error
    return weights

def gradAscent0(dataIn, classLabels):
    m,n = np.shape(dataIn)
    alpha = 0.01
    weights = np.ones(n)
    for i in range(m):
        h = sigmoid(dataIn[i] * weights)
        error = classLabels[i] - h
        weights = weights + alpha  * error * dataIn[i]
    return weights
def gradAscent1(dataMat, classLabels):
    dataIn = np.array(dataMat)
    m,n = np.shape(dataIn)
    alpha = 0.01
    weights = np.ones(n)
    maxClycles =800
    for i in range(maxClycles):
        dataIndex = range(m)
        for j in range(m):
            alpha = 4/(1.0+i+j) + 0.01
            randIndex = int(np.random.uniform(0,len(dataIndex)))
            h = sigmoid(sum(dataIn[randIndex]*weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha  * error * dataIn[randIndex]
            del(dataIndex[randIndex])
    return weights
def classify(inVect,weights):
    prob = sigmoid(sum(np.mat(inVect) * weights))
    print prob
    if prob > 0.5:
        return 1
    else:
        return 0
def classify0(inVect,weights):
    prob = sigmoid(sum(inVect * weights))
    print prob
    if prob > 0.5:
        return 1
    else:
        return 0
def plotBestFit(weights):
    import matplotlib.pyplot as plt
    dataMat,labelMat = readData()
    dataArr = np.array(dataMat)
    n = np.shape(dataArr)[0]
    xcord1= []; xcord2 =[]
    ycord1 =[]; ycord2 = []
    for i in range(n):
        if int(labelMat[i]) ==1:
            xcord1.append(dataArr[i,1])
            ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1])
            ycord2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1,ycord1,s=30,c='red',marker='s')
    ax.scatter(xcord2,ycord2,s=30,c='green')
    x = np.arange(-3.0,3.0,1)
    # y = ((-weights[0] - weights[1]*x)/weights[2]).getA()[0]
    y = ((-weights[0] - weights[1]*x)/weights[2])
    print x,y
    ax.plot(x,y)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()

def test():
    dataset,labels = readData()
    weight = gradAscent1(dataset,labels)
    weight0 = gradAscent0(dataset,labels)
    weight1 = gradAscent1(dataset,labels)
    print weight,weight0,weight1
    a = classify0([1,0.196949, 1.444165],weight)
    print a
    plotBestFit(weight)

test()