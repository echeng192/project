# -*- coding: utf-8 -*-
"""
Created on Fri Jun 07 21:20:05 2017
@author: Paul
KNN  classify function
"""
from math import log
import operator
def createDataset():
    dataset = [[1, 1, 1,'Y'], [1, 1, 0,'Y'], [1, 0, 1,'n'], [0, 1, 1,'n'], [0, 0, 0,'m'], [0, 0, 1,'m']]
    labels = ['nosurfacing', 'flippers','Test']
    return dataset, labels
'''
calculate entropy for input dataset
'''
def calEntropy(dataset):
    numEn = len(dataset)
    labelsClass = {}
    for vect in dataset:
        label = vect[-1]
        labelsClass[label] = labelsClass.get(label, 0) +1
    entrop = 0.0
    for key in labelsClass:
        prob = float(labelsClass[key])/numEn
        entrop -= prob * log(prob, 2)
    return entrop
'''
split dataset by input feature
'''
def splitDataset(dataset,axis,value):
    reDataset = []
    for feat in dataset:
        if feat[axis] == value:
            reduceFeat = feat[:axis]
            reduceFeat.extend(feat[axis+1:])
            reDataset.append(reduceFeat)
    return reDataset

'''
choose the best feature
'''
def chooseBestFeature(dataset):
    numFeature = len(dataset[0]) -1
    baseEntropy = calEntropy(dataset)
    bestInfoGain = 0.0; bestFeature = -1
    for i in range(numFeature):
        featList = [example[i] for example in dataset]
        uniqueVals = set(featList)
        newEntropy = 0.0
        for value in uniqueVals:
            subDataset = splitDataset(dataset, i, value)
            prob = len(subDataset)/float(len(dataset))
            newEntropy += prob * calEntropy(subDataset)
        infoGain = baseEntropy - newEntropy
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature

def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys() : 
            classCount[vote] = 0
    classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]


def createTree(dataset, labels):
    classList = [example[-1] for example in dataset]
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    if len(dataset[0]) == 1:
        return majorityCnt(classList)
    bestFeat = chooseBestFeature(dataset)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}}
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataset]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataset(dataset,bestFeat,value),subLabels)
    return myTree

def classify(inputTree,labels,inputVect):
    firstLab = inputTree.keys()[0]
    secordDict = inputTree[firstLab]
    featIndex = labels.index(firstLab)
    for key in secordDict.keys():
        if inputVect[featIndex] == key:
            if type(secordDict[key]).__name__=='dict':
                classLabel = classify(secordDict[key],labels,inputVect)
            else:
                classLabel = secordDict[key]
    return classLabel

def test():
    mydata,labels = createDataset()
    tree = createTree(mydata,labels[:])
    print tree
    saveTree(tree, 'testsavetree.data')
    testVect = [0, 0, 1]
    re = classify(tree,labels,testVect)
    print re
def test2():
    tree = grabTree('testsavetree.data')
    labels = ['nosurfacing', 'flippers','Test']
    testVect = [0, 0, 1]
    re = classify(tree,labels[:],testVect)
    print re

def saveTree(tree, filename):
    import pickle
    fw = open(filename,'w')
    pickle.dump(tree, fw)
    fw.close()

def grabTree(filename):
    import pickle
    fr = open(filename)
    return pickle.load(fr)

test()
test2()