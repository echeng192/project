# -*- coding: utf-8 -*-
"""
Created on Fri Jun 06 21:20:05 2017
@author: Paul
KNN  classify function
"""
import numpy as np
import urllib2
def classifyKNN(inputData,trainData,labels,k=3):
    datasetSize = trainData.shape[0]
    # diffSet = np.tile(inputData,(datasetSize,1)) - trainData
    # sqdiffSet = diffSet ** 2
    # sqDistance = sqdiffSet.sum(axis=1)
    # distances = sqDistance ** 0.5
    diffSet = np.tile(inputData,(datasetSize,1))
    distances = np.linalg.norm(diffSet - trainData,axis=1)
    sortDistances = distances.argsort()
    classCount = {}
    for i in range(k):
        label = labels[sortDistances[i]]
        classCount[label] = classCount.get(label,0) +1
    result = [(classCount[t],t) for t in classCount]
    result.sort(reverse=True)

    return result[0][1]

def classifyKNN_standard(inputData,trainData,labels,k=3):
    datasetSize = trainData.shape[0]
    # diffSet = np.tile(inputData,(datasetSize,1)) - trainData
    # sqdiffSet = diffSet ** 2
    # sqDistance = sqdiffSet.sum(axis=1)
    # distances = sqDistance ** 0.5
    inputSet = np.tile(inputData,(datasetSize,1))
    trainMax = np.max(trainData,axis=0)
    trainMin = np.min(trainData,axis=0)
    maxSet = np.tile(trainMax,(datasetSize,1))
    minSet = np.tile(trainMin,(datasetSize,1))
    trainStandard = (trainData - minSet)/(maxSet-minSet)
    inputStandard = (inputSet - minSet)/(maxSet-minSet)
    distances = np.linalg.norm(inputStandard - trainStandard,axis=1)
    sortDistances = distances.argsort()
    classCount = {}
    for i in range(k):
        label = labels[sortDistances[i]]
        classCount[label] = classCount.get(label,0) +1
    result = [(classCount[t],t) for t in classCount]
    result.sort(reverse=True)

    return result[0][1]

def classifyKNN_cos(inputData,trainData,labels,k=3):
    datasetSize = trainData.shape[0]
    A = np.tile(inputData,(datasetSize,1)) 
    AB = (A * trainData)
    sumAB = AB.sum(axis=1)
    denom = np.linalg.norm(A,axis=1) * np.linalg.norm(trainData,axis=1)
    cos = (sumAB+1)/(denom+1)
    sim = (cos*0.5 + 0.5)
    sortDistances = sim.argsort()
    classCount = {}
    for i in range(k):
        label = labels[sortDistances[k-1-i]]
        classCount[label] = classCount.get(label,0) +1
    result = [(classCount[t],t) for t in classCount]
    result.sort(reverse=True)

    return result[0][1]
def classifyKNN_pearson(inputData,trainData,labels,k=3):
    datasetSize = trainData.shape[0]
    diffSet = np.tile(inputData,(datasetSize,1))
    XY = (diffSet * trainData).sum(axis=1)
    sumX = diffSet.sum(axis=1) 
    sumY = trainData.sum(axis=1)
    seqX = (diffSet **2).sum(axis=1)
    seqY = (trainData **2).sum(axis=1)
    n = trainData.shape[1]
    distances = (XY - sumX * sumY/n)/(((seqX - sumX**2/n)**0.5) *((seqY - sumY**2/n)**0.5))
    sortDistances = distances.argsort()
    classCount = {}
    for i in range(k):
        label = labels[sortDistances[k-1-i]]
        classCount[label] = classCount.get(label,0) +1
    result = [(classCount[t],t) for t in classCount]
    result.sort(reverse=True)

    return result[0][1]
def test(classify=classifyKNN):
    trainData,labels = createData()
    inputData= np.array([6.7, 3.0, 5.1, 1.8])
    group = np.array(trainData)
    a = classify(inputData,group,labels)
    # b = classify(np.array([0.1,0.1]),group,labels)
    print a

def downloadData():
    url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
    res = urllib2.urlopen(url,timeout=30)
    content = res.read()
    rows = content.split('\n')
    data = []
    labels = []
    for row in rows:
        if row =="": continue
        temp = row.split(',')
        data.append([float(temp[i]) for i in range(4)])
        labels.append(temp[4])
    return data,labels

def createData():
    data =[[5.1, 3.5, 1.4, 0.2], [4.9, 3.0, 1.4, 0.2], [4.7, 3.2, 1.3, 0.2], [4.6, 3.1, 1.5, 0.2], [5.0, 3.6, 1.4, 0.2], [5.4, 3.9, 1.7, 0.4], [4.6, 3.4, 1.4, 0.3], [5.0, 3.4, 1.5, 0.2], [4.4, 2.9, 1.4, 0.2], [4.9, 3.1, 1.5, 0.1], [5.4, 3.7, 1.5, 0.2], [4.8, 3.4, 1.6, 0.2], [4.8, 3.0, 1.4, 0.1], [4.3, 3.0, 1.1, 0.1], [5.8, 4.0, 1.2, 0.2], [5.7, 4.4, 1.5, 0.4], [5.4, 3.9, 1.3, 0.4], [5.1, 3.5, 1.4, 0.3], [5.7, 3.8, 1.7, 0.3], [5.1, 3.8, 1.5, 0.3], [5.4, 3.4, 1.7, 0.2], [5.1, 3.7, 1.5, 0.4], [4.6, 3.6, 1.0, 0.2], [5.1, 3.3, 1.7, 0.5], [4.8, 3.4, 1.9, 0.2], [5.0, 3.0, 1.6, 0.2], [5.0, 3.4, 1.6, 0.4], [5.2, 3.5, 1.5, 0.2], [5.2, 3.4, 1.4, 0.2], [4.7, 3.2, 1.6, 0.2], [4.8, 3.1, 1.6, 0.2], [5.4, 3.4, 1.5, 0.4], [5.2, 4.1, 1.5, 0.1], [5.5, 4.2, 1.4, 0.2], [4.9, 3.1, 1.5, 0.1], [5.0, 3.2, 1.2, 0.2], [5.5, 3.5, 1.3, 0.2], [4.9, 3.1, 1.5, 0.1], [4.4, 3.0, 1.3, 0.2], [5.1, 3.4, 1.5, 0.2], [5.0, 3.5, 1.3, 0.3], [4.5, 2.3, 1.3, 0.3], [4.4, 3.2, 1.3, 0.2], [5.0, 3.5, 1.6, 0.6], [5.1, 3.8, 1.9, 0.4], [4.8, 3.0, 1.4, 0.3], [5.1, 3.8, 1.6, 0.2], [4.6, 3.2, 1.4, 0.2], [5.3, 3.7, 1.5, 0.2], [5.0, 3.3, 1.4, 0.2], [7.0, 3.2, 4.7, 1.4], [6.4, 3.2, 4.5, 1.5], [6.9, 3.1, 4.9, 1.5], [5.5, 2.3, 4.0, 1.3], [6.5, 2.8, 4.6, 1.5], [5.7, 2.8, 4.5, 1.3], [6.3, 3.3, 4.7, 1.6], [4.9, 2.4, 3.3, 1.0], [6.6, 2.9, 4.6, 1.3], [5.2, 2.7, 3.9, 1.4], [5.0, 2.0, 3.5, 1.0], [5.9, 3.0, 4.2, 1.5], [6.0, 2.2, 4.0, 1.0], [6.1, 2.9, 4.7, 1.4], [5.6, 2.9, 3.6, 1.3], [6.7, 3.1, 4.4, 1.4], [5.6, 3.0, 4.5, 1.5], [5.8, 2.7, 4.1, 1.0], [6.2, 2.2, 4.5, 1.5], [5.6, 2.5, 3.9, 1.1], [5.9, 3.2, 4.8, 1.8], [6.1, 2.8, 4.0, 1.3], [6.3, 2.5, 4.9, 1.5], [6.1, 2.8, 4.7, 1.2], [6.4, 2.9, 4.3, 1.3], [6.6, 3.0, 4.4, 1.4], [6.8, 2.8, 4.8, 1.4], [6.7, 3.0, 5.0, 1.7], [6.0, 2.9, 4.5, 1.5], [5.7, 2.6, 3.5, 1.0], [5.5, 2.4, 3.8, 1.1], [5.5, 2.4, 3.7, 1.0], [5.8, 2.7, 3.9, 1.2], [6.0, 2.7, 5.1, 1.6], [5.4, 3.0, 4.5, 1.5], [6.0, 3.4, 4.5, 1.6], [6.7, 3.1, 4.7, 1.5], [6.3, 2.3, 4.4, 1.3], [5.6, 3.0, 4.1, 1.3], [5.5, 2.5, 4.0, 1.3], [5.5, 2.6, 4.4, 1.2], [6.1, 3.0, 4.6, 1.4], [5.8, 2.6, 4.0, 1.2], [5.0, 2.3, 3.3, 1.0], [5.6, 2.7, 4.2, 1.3], [5.7, 3.0, 4.2, 1.2], [5.7, 2.9, 4.2, 1.3], [6.2, 2.9, 4.3, 1.3], [5.1, 2.5, 3.0, 1.1], [5.7, 2.8, 4.1, 1.3], [6.3, 3.3, 6.0, 2.5], [5.8, 2.7, 5.1, 1.9], [7.1, 3.0, 5.9, 2.1], [6.3, 2.9, 5.6, 1.8], [6.5, 3.0, 5.8, 2.2], [7.6, 3.0, 6.6, 2.1], [4.9, 2.5, 4.5, 1.7], [7.3, 2.9, 6.3, 1.8], [6.7, 2.5, 5.8, 1.8], [7.2, 3.6, 6.1, 2.5], [6.5, 3.2, 5.1, 2.0], [6.4, 2.7, 5.3, 1.9], [6.8, 3.0, 5.5, 2.1], [5.7, 2.5, 5.0, 2.0], [5.8, 2.8, 5.1, 2.4], [6.4, 3.2, 5.3, 2.3], [6.5, 3.0, 5.5, 1.8], [7.7, 3.8, 6.7, 2.2], [7.7, 2.6, 6.9, 2.3], [6.0, 2.2, 5.0, 1.5], [6.9, 3.2, 5.7, 2.3], [5.6, 2.8, 4.9, 2.0], [7.7, 2.8, 6.7, 2.0], [6.3, 2.7, 4.9, 1.8], [6.7, 3.3, 5.7, 2.1], [7.2, 3.2, 6.0, 1.8], [6.2, 2.8, 4.8, 1.8], [6.1, 3.0, 4.9, 1.8], [6.4, 2.8, 5.6, 2.1], [7.2, 3.0, 5.8, 1.6], [7.4, 2.8, 6.1, 1.9], [7.9, 3.8, 6.4, 2.0], [6.4, 2.8, 5.6, 2.2], [6.3, 2.8, 5.1, 1.5], [6.1, 2.6, 5.6, 1.4], [7.7, 3.0, 6.1, 2.3], [6.3, 3.4, 5.6, 2.4], [6.4, 3.1, 5.5, 1.8], [6.0, 3.0, 4.8, 1.8], [6.9, 3.1, 5.4, 2.1], [6.7, 3.1, 5.6, 2.4], [6.9, 3.1, 5.1, 2.3], [5.8, 2.7, 5.1, 1.9], [6.8, 3.2, 5.9, 2.3], [6.7, 3.3, 5.7, 2.5], [6.7, 3.0, 5.2, 2.3], [6.3, 2.5, 5.0, 1.9], [6.5, 3.0, 5.2, 2.0], [6.2, 3.4, 5.4, 2.3], [5.9, 3.0, 5.1, 1.8]]
    labels = ['Iris-setosa', 'Iris-setosa', 'Iris-setosa', 'Iris-setosa', 'Iris-setosa', 'Iris-setosa', 'Iris-setosa', 'Iris-setosa', 'Iris-setosa', 'Iris-setosa', 'Iris-setosa', 'Iris-setosa', 'Iris-setosa', 'Iris-setosa', 'Iris-setosa', 'Iris-setosa', 'Iris-setosa', 'Iris-setosa', 'Iris-setosa', 'Iris-setosa', 'Iris-setosa', 'Iris-setosa', 'Iris-setosa', 'Iris-setosa', 'Iris-setosa', 'Iris-setosa', 'Iris-setosa', 'Iris-setosa', 'Iris-setosa', 'Iris-setosa', 'Iris-setosa', 'Iris-setosa', 'Iris-setosa', 'Iris-setosa', 'Iris-setosa', 'Iris-setosa', 'Iris-setosa', 'Iris-setosa', 'Iris-setosa', 'Iris-setosa', 'Iris-setosa', 'Iris-setosa', 'Iris-setosa', 'Iris-setosa', 'Iris-setosa', 'Iris-setosa', 'Iris-setosa', 'Iris-setosa', 'Iris-setosa', 'Iris-setosa', 'Iris-versicolor', 'Iris-versicolor', 'Iris-versicolor', 'Iris-versicolor', 'Iris-versicolor', 'Iris-versicolor', 'Iris-versicolor', 'Iris-versicolor', 'Iris-versicolor', 'Iris-versicolor', 'Iris-versicolor', 'Iris-versicolor', 'Iris-versicolor', 'Iris-versicolor', 'Iris-versicolor', 'Iris-versicolor', 'Iris-versicolor', 'Iris-versicolor', 'Iris-versicolor', 'Iris-versicolor', 'Iris-versicolor', 'Iris-versicolor', 'Iris-versicolor', 'Iris-versicolor', 'Iris-versicolor', 'Iris-versicolor', 'Iris-versicolor', 'Iris-versicolor', 'Iris-versicolor', 'Iris-versicolor', 'Iris-versicolor', 'Iris-versicolor', 'Iris-versicolor', 'Iris-versicolor', 'Iris-versicolor', 'Iris-versicolor', 'Iris-versicolor', 'Iris-versicolor', 'Iris-versicolor', 'Iris-versicolor', 'Iris-versicolor', 'Iris-versicolor', 'Iris-versicolor', 'Iris-versicolor', 'Iris-versicolor', 'Iris-versicolor', 'Iris-versicolor', 'Iris-versicolor', 'Iris-versicolor', 'Iris-versicolor', 'Iris-virginica', 'Iris-virginica', 'Iris-virginica', 'Iris-virginica', 'Iris-virginica', 'Iris-virginica', 'Iris-virginica', 'Iris-virginica', 'Iris-virginica', 'Iris-virginica', 'Iris-virginica', 'Iris-virginica', 'Iris-virginica', 'Iris-virginica', 'Iris-virginica', 'Iris-virginica', 'Iris-virginica', 'Iris-virginica', 'Iris-virginica', 'Iris-virginica', 'Iris-virginica', 'Iris-virginica', 'Iris-virginica', 'Iris-virginica', 'Iris-virginica', 'Iris-virginica', 'Iris-virginica', 'Iris-virginica', 'Iris-virginica', 'Iris-virginica', 'Iris-virginica', 'Iris-virginica', 'Iris-virginica', 'Iris-virginica', 'Iris-virginica', 'Iris-virginica', 'Iris-virginica', 'Iris-virginica', 'Iris-virginica', 'Iris-virginica', 'Iris-virginica', 'Iris-virginica', 'Iris-virginica', 'Iris-virginica', 'Iris-virginica', 'Iris-virginica', 'Iris-virginica', 'Iris-virginica', 'Iris-virginica', 'Iris-virginica']
    return data,labels
# test()
# test(classify=classifyKNN_standard)


def readTrainData(filepath):
    import os
    pathDir =  os.listdir(filepath)
    dataset = []
    labels = []
    for allDir in pathDir:
        child = os.path.join('%s%s' % (filepath, allDir))
        file = open(child)
        lines = file.readlines()
        fileVect = []
        labels.append(allDir.split('_')[0])
        for line in lines:
            arrLine = line.strip()
            for i in range(32):
                fileVect.append(int(arrLine[i]))
        dataset.append(fileVect)
    return dataset,labels


def handwriteTest(classify=classifyKNN):
    train_path = 'D:\\ML\\project\\data\\digits\\trainingDigits\\'
    test_path = 'D:\\ML\\project\\data\\digits\\testDigits\\'

    trainData,trainLabels = readTrainData(train_path)
    testData,testLabels = readTrainData(test_path)
    group = np.array(trainData)
    resultLabels = []
    for vect in testData:
        inputVect = np.array(vect)
        digit = classify(inputVect,group,trainLabels,3)
        resultLabels.append(digit)

    total = 0;correct =0;wrong = 0
    total = len(resultLabels)
    for i in range(total):
        if testLabels[i] == resultLabels[i]:
            correct +=1
        else:
            wrong += 1
    print 'total:%d correct:%d wrong:%d' % (total,correct,wrong)

handwriteTest()
