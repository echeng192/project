# -*- coding: utf-8 -*-
"""
Created on Fri Jun 04 10:20:05 2017
@author: Paul
"""

music ={
    "Dr Dog":{"piano":2.5,"vocals":4,"beat":3.5,"blues":3,"guitar":5,"backup":3,"rap":1},
    "Phoenix":{"piano":3,"vocals":3,"beat":3,"blues":2,"guitar":2,"backup":4,"rap":2},
    "Out at Sea":{"piano":4,"vocals":2.5,"beat":2.5,"blues":3,"guitar":3,"backup":3,"rap":1.5},
    "Cat me":{"piano":3,"vocals":4.5,"beat":1.5,"blues":2,"guitar":2.5,"backup":3.5,"rap":3.5},
    "Tank you":{"piano":1,"vocals":2,"beat":4.5,"blues":4,"guitar":4,"backup":1.5,"rap":2},
    "Fuck":{"piano":2.5,"vocals":2,"beat":3.5,"blues":3.5,"guitar":3.5,"backup":5,"rap":4},
    "Paul":{"piano":2.5,"vocals":4,"beat":3.5,"blues":3,"guitar":5,"backup":3,"rap":1},
    "Hot":{"piano":2.5,"vocals":4,"beat":3.5,"blues":3.2,"guitar":5,"backup":3,"rap":1}
}
'''
calcluate distance for sepicial two items
'''
def cal_distance(prefs,item1,item2):
    sumdev = sum([(item1[attr] - item2[attr])**2 for attr in item1 if attr in item2])
    return sumdev**0.5

'''
use standard score to calcluate distance for sepicial two items
'''
def cal_dis_standard(prefs,item1,item2):
    st_item1 = standard_item(prefs,item1)
    st_item2 = standard_item(prefs,item2)
    sumdev = sum([(st_item1[attr] - st_item2[attr])**2 for attr in st_item1 if attr in st_item2])
    return sumdev**0.5

'''
use absolute standard score to calcluate distance for sepicial two items
'''
def cal_dis_standard_abs(prefs,item1,item2):
    st_item1 = abs_standard_item(prefs,item1)
    st_item2 = abs_standard_item(prefs,item2)
    sumdev = sum([(st_item1[attr] - st_item2[attr])**2 for attr in st_item1 if attr in st_item2])
    return sumdev**0.5
'''
change to standard score
'''
def standard_item(prefs,item):
    si ={}
    for attr in item:
        count =sum([1 for tp in prefs if attr in prefs[tp]])
        if count ==0:continue
        sum1= sum([prefs[tp][attr] for tp in prefs if attr in prefs[tp]])       
        avg = float(sum1/count)
        sum2 = sum([(prefs[tp][attr] -avg)**2 for tp in prefs if attr in prefs[tp]])
        sd = (sum2/count)**0.5
        si[attr] = (item[attr] - avg)/sd   
    return si
'''
change to absolute standard score
'''
def abs_standard_item(prefs,item):
    si ={}
    for attr in item:
        count =sum([1 for tp in prefs if attr in prefs[tp]])
        if count ==0:continue
        lst =[prefs[tp][attr] for tp in prefs if attr in prefs[tp]]
        lst.sort()
        avg =0
        if count%2==0:
            avg = (lst[count/2-1]+lst[count/2])/2.0
        else:
            avg =lst[count/2]
        sum2 = sum([abs(prefs[tp][attr] -avg) for tp in prefs if attr in prefs[tp]])
        asd = float(sum2/count)
        si[attr] = (item[attr] - avg)/asd   
    return si

def recommendTopItem(prefs,item,n=3,cal_dis=cal_dis_standard): 
    si = []
    for tp in prefs:
        if tp == item:continue
        score = cal_dis(prefs,prefs[item],prefs[tp])
        si.append((score,tp))
    si.sort()
    return si[:n]

print recommendTopItem(music,'Dr Dog',10,cal_dis=cal_dis_standard)
print recommendTopItem(music,'Dr Dog',10,cal_dis=cal_dis_standard_abs)
print recommendTopItem(music,'Dr Dog',10,cal_dis=cal_distance)