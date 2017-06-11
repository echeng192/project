# -*- coding: utf-8 -*-
"""
Created on Fri Jun 02 21:09:05 2017

@author: Paul
"""
import numpy as np
critics= {'Lisa Rose': {'Lady in the Water': 2.5, 'Snakes on a Plane': 3.5,
    'Just My Luck': 3.0, 'Superman Returns': 3.5, 'You, Me and Dupree': 2.5,
    'The Night Listener': 3.0},
    'Gene Seymour': {'Lady in the Water': 3.0, 'Snakes on a Plane': 3.5,
    'Just My Luck': 1.5, 'Superman Returns': 5.0, 'The Night Listener': 3.0,
    'You, Me and Dupree': 3.5},
    'Michael Phillips': {'Lady in the Water': 2.5, 'Snakes on a Plane': 3.0,
    'Superman Returns': 3.5, 'The Night Listener': 4.0},
    'Claudia Puig': {'Snakes on a Plane': 3.5, 'Just My Luck': 3.0,
    'The Night Listener': 4.5, 'Superman Returns': 4.0,
    'You, Me and Dupree': 2.5},
    'Mick LaSalle': {'Lady in the Water': 3.0, 'Snakes on a Plane': 4.0,
    'Just My Luck': 2.0, 'Superman Returns': 3.0, 'The Night Listener': 3.0,
    'You, Me and Dupree': 2.0},
    'Jack Matthews': {'Lady in the Water': 3.0, 'Snakes on a Plane': 4.0,
    'The Night Listener': 3.0, 'Superman Returns': 5.0, 'You, Me and Dupree': 3.5},
    'Toby': {'Snakes on a Plane':4.5,'You, Me and Dupree':3.0,'Superman Returns':4.0}
}

'''
Euclidean Distance Score
'''
def sim_distance(prefs,person1,person2):
    sl={}
    for item in prefs[person1]:
        if item in prefs[person2]:
            sl[item]=1
    if len(sl)==0:
        return 0
    sum_of = sum([(prefs[person1][item]-prefs[person2][item])**2 for item in prefs[person1] if item in prefs[person2]]) 
    return 1/(1+sum_of**0.5)

'''
Pearson Correlation Score
'''
def sim_pearson(prefs,person1,person2):
    sl={}
    for item in prefs[person1]:
        if item in prefs[person2]:
            sl[item]=1
    n = len(sl)
    if n==0:
        return 0
    sum1 = sum([prefs[person1][t] for t in sl])
    sum2 = sum([prefs[person2][t] for t in sl]) 
    sum_pow1=sum([prefs[person1][t]**2 for t in sl])
    sum_pow2=sum([prefs[person2][t]**2 for t in sl])  
    psum = sum([prefs[person1][t]*prefs[person2][t] for t in sl])
    den = ((sum_pow1 -sum1**2/n)**0.5)*((sum_pow2 -sum2**2/n)**0.5)
    if den ==0:return 0
    pearson = float((psum - sum1*sum2/n)/den)
    return pearson
'''
cos similarity
'''
def sim_cos(prefs,person1,person2):
    sl ={}
    for item in prefs[person1]:
        if item in prefs[person2]:
            sl[item]=1
    n = len(sl)
    if n==0:
        return 0
    
    sump = sum([prefs[person1][t] * prefs[person2][t] for t in sl])
    sumx = sum([prefs[person1][t]**2 for t in sl])**0.5
    sumy = sum([prefs[person2][t]**2 for t in sl])**0.5
    return sump/(sumx * sumy)           
'''
get the top n similarity person for input person
'''        
def compareTop(prefs,person,n=3,similarity=sim_distance):
    scores = [(similarity(prefs,person,other),other) for other in prefs if other!=person]
    scores.sort()
    scores.reverse()
    return scores[0:n]

# recommend items for input person by similarity person                  
def recommendItemByPerson(prefs,person,similarity=sim_distance):
    totals ={}
    simSums ={}
    for other in prefs:
        if other == person:continue
        sim = similarity(prefs,person,other)
        if sim ==0:continue
        for item in prefs[other]:
            maxs = max([prefs[other][t] for t in prefs[other]])
            mins = min([prefs[other][t] for t in prefs[other]])
            if item not in prefs[person] or prefs[person][item]==0:
                totals.setdefault(item,0)
                totals[item]+=prefs[other][item] * sim
                simSums.setdefault(item,0)
                simSums[item]+=sim
    rankings =[(total/simSums[item],item) for item,total in totals.items()]                    
    
    rankings.sort()
    rankings.reverse()
    return rankings 
'''
calcluate cos value for two input production
'''
def sim_item_cos(prefs,item1,item2):
    avgs={}
    for key,ratings in prefs.items():
        avgs[key] = sum(ratings.values())/float(len(ratings.values()))
    sum1 =0
    dem1=0
    dem2=0
    for (person,ratings) in prefs.items():
        if item1 in ratings and item2 in ratings:
            avg = avgs[person]
            sum1 += (ratings[item1] - avg)*(ratings[item2]-avg)
            dem1 +=((ratings[item1] - avg)**2)
            dem2 +=((ratings[item2] - avg)**2)
    if dem1 !=0 and dem2 !=0:     
        return sum1/((dem1 **0.5) * (dem2**0.5))
    else:
        return 0
    
def getAllItemSim(prefs):
    sl ={}
    for person,ratings in prefs.items():
        for key in ratings:
            if key not in sl:
                sl[key] =0
    for key,item in sl.items():
        detail ={}
        for key2 in sl:          
            if key != key2:
                detail[key2] = sim_item_cos(prefs,key,key2)
        sl[key]=detail
    return sl             
'''
use cos similarity to recommend items for sepical person
'''
def recommendItemByCos(prefs,person):
    if person not in prefs:
        print 'cannot find the person'
        return []
    sl ={}
    simTable = getAllItemSim(prefs)
    mn=min(prefs[person].values())
    mx=max(prefs[person].values())
    for key in simTable:
        if key not in prefs[person]:
            sl[key] =0
    for item in sl:
        s=0
        sn=0
        for key in prefs[person]:
            nr =0
            if mx != mn:
                nr =(2*(prefs[person][key]-mn) -(mx-mn))/(mx-mn)
            s += abs(simTable[item][key])
            sn +=simTable[item][key]*nr      
        r =sn/s
        score = 0.5 * (r+1)*(mx-mn)+mn
        sl[item] =score
        #sl[item] =sn/s
    rating = [(item,key) for key,item in sl.items()]
    rating.sort()
    rating.reverse()
    return rating
    
'''
slope one
'''
def getAllItemSlopeDev(prefs):
    sl ={}
    for person in prefs:
        for item in prefs[person]:
            if item not in sl:
                sl[item] =1
            else:
                sl[item]+=1    
    for item in sl:
        detail ={}
        for item2 in sl:
            if item ==item2:continue
            card = 0
            sumu =0
            for person in prefs:
                if item in prefs[person] and item2 in prefs[person]:
                    card +=1
                    sumu +=(prefs[person][item] - prefs[person][item2])
            if card==0:continue
            dev = sumu/card
            detail[item2] = [dev,card]
        sl[item] = detail
    return sl

'''
use slope one to recommend items for sepical person
'''
def recommendItemBySlope(prefs,person):
    if person not in prefs:
        print 'cannot find the person'
        return []
    sl ={}
    devTable = getAllItemSlopeDev(prefs)
    for key in devTable:
        if key not in prefs[person]:
            sl[key] =0        
    for item in sl:
        sum1 = sum([(u + devTable[item][key][0])* devTable[item][key][1] for key,u in prefs[person].items()])
        sum2 = sum([devTable[item][key][1] for key in prefs[person]])
        sl[item] = float(sum1/sum2)
    rating = [(item,key) for key,item in sl.items()]
    rating.sort()
    rating.reverse()
    return rating
#print getAllItemSlopeDev(critics)
print recommendItemByCos(critics,'Toby')
print recommendItemBySlope(critics,'Toby')
print recommendItemByPerson(critics,'Toby',similarity=sim_distance)
print recommendItemByPerson(critics,'Toby',similarity=sim_pearson)
print recommendItemByPerson(critics,'Toby',similarity=sim_cos)