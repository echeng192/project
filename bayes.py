@@ -1,175 +0,0 @@
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 05 21:20:05 2017
@author: Paul
bayes classify function
"""
import math
import numpy as np
from sklearn import svm
def createDataset():
    documentList=[['love','you','like','has','help','please'],
    ['maybe','not','take','him','to','dog','park','stupid'],
    ['my','dalmation','is','so','cute','love','him'],
    ['stop','posting','stupid','worthless','garbage','sheet'],
    ['mr','licks','ate','my','steak','how'],
    ['quit','buying','worthless','dog','food','stupid','fuck']]
    classList=['good','bad','good','bad','good','bad']
    return documentList,classList

'''
create the vocab list
'''
def createVocabList(dataset):
    vocabList = set([])
    for doc in dataset:
        vocabList =vocabList | set(doc)          
    return list(vocabList)

'''
change a docment to a vector
'''
def changeDoc2Vect(vocabList,inputDoc):
    n = len(vocabList)
    returnList =[0] * n
    for word in inputDoc:
        if word in vocabList:
            returnList[vocabList.index(word)] += 1
    return returnList

'''
train the beys classify
'''
def trainBN(trainData,trainCategory):
    numTrainDoc = len(trainData)
    numWords = len(trainData[0])
    categorySet = set(trainCategory)
    ctDen = {}; ctNum ={}; ctPA ={}; ctVect ={}
    for ct in categorySet:
        ctNum[ct] = [1.0] * numWords
        ctDen[ct] = 2.0
        ctPA[ct] = sum([1 for t in trainCategory if t==ct])/float(numTrainDoc)
    for i in range(numTrainDoc):
        for category in categorySet:
            if trainCategory[i] == category:
                ctDen[category] += sum(trainData[i])
                for j in range(numWords):
                    ctNum[category][j] +=trainData[i][j]  
    for ct in categorySet:
        ctVect[ct] = [math.log(t/ctDen[ct]) for t in ctNum[ct]]
    return ctVect,ctPA

'''
classify a doction by train data
'''
def classifyBN(doc, ctVect,ctPA):
    size = len(doc)
    probCategory = {}
    for ct in ctPA:
        probCategory[ct] = ctPA[ct]
    for i in range(size):
        for ct in probCategory:
            probCategory[ct] += doc[i] * ctVect[ct][i]
    
    listProb= [ [probCategory[ct],ct] for ct in probCategory]
    listProb.sort()
    listProb.reverse()
    return listProb[0][1]

def test():
    dataSet,classSet = createDataset()
    vocabList = createVocabList(dataSet)
    trainData =[]
    for doc in dataSet:
        trainData.append(changeDoc2Vect(vocabList,doc))
    ctVect,ctPA = trainBN(trainData, classSet)
    # testDoc='stupid fuck worthless dog'.split(' ')
    testDoc = ['love','you','like','has','help','please']
    testEnter = changeDoc2Vect(vocabList,testDoc)
    result = classifyBN(testEnter,ctVect,ctPA)
    return result
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

def saveProb(ctVect, ctPA,filename):
    import pickle
    prob =[ctVect, ctPA]
    fw = open(filename,'w')
    pickle.dump(prob, fw)
    fw.close()
def readProb(filename):
    import pickle
    fr = open(filename)
    prob = pickle.load(fr)
    return prob[0],prob[1]

def testTrain():
    train_path = 'D:\\ML\\project\\data\\digits\\trainingDigits\\'
    trainData,trainLabels = readTrainData(train_path)
    ctVect,ctPA = trainBN(trainData, trainLabels)
    saveProb(ctVect,ctPA,'testprobsave')
    print 'ok'
def handwriteTest():
    train_path = 'D:\\ML\\project\\data\\digits\\trainingDigits\\'
    test_path = 'D:\\ML\\project\\data\\digits\\testDigits\\'

    # trainData,trainLabels = readTrainData(train_path)
    testData,testLabels = readTrainData(test_path)
    # group = np.array(trainData)
    resultLabels = []
    ctVect,ctPA = readProb('testprobsave')
    for vect in testData:
        # inputVect = np.array(vect)
        digit = classifyBN(vect,ctVect,ctPA)
        resultLabels.append(digit)

    total = 0;correct =0;wrong = 0
    total = len(resultLabels)
    for i in range(total):
        if testLabels[i] == resultLabels[i]:
            correct +=1
        else:
            wrong += 1
    print 'total:%d correct:%d wrong:%d pe:%d' % (total,correct,wrong,correct/float(total))

def testSVM():
    train_path = 'D:\\ML\\project\\data\\digits\\trainingDigits\\'
    test_path = 'D:\\ML\\project\\data\\digits\\testDigits\\'
    clf = svm.SVC()
    trainData,trainLabels = readTrainData(train_path)
    testData,testLabels = readTrainData(test_path)
    clf.fit(trainData,trainLabels)
    resultLabels = []
    for vect in testData:
        result = clf.predict(vect)
        resultLabels.append(result[0])

    total = 0;correct =0;wrong = 0
    total = len(resultLabels)
    for i in range(total):
        if testLabels[i] == resultLabels[i]:
            correct +=1
        else:
            wrong += 1
    print 'total:%d correct:%d wrong:%d pe:%d' % (total,correct,wrong,correct/float(total))

    
# handwriteTest()
testSVM()

