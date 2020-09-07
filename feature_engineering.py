# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 09:45:14 2020

@author: Z
"""

import numpy as np
import pandas as pd

#按模态合并数据
def data_load(path11,path12,path2,path31,path32,path41,path42,wf12,wf34):   
    datas1 = list()
    IDS = list()
    with open(path11, "r") as lines:  #k-mer
        for line in lines:
            s = line.strip().split(",")
            datas1.append(s[21:85])
            IDs.append(s[0])
    len1 = len(data1)
    labels1 = np.ones(len1).tolist()
    
    with open(path12, "r") as lines:
        for line in lines:
            s = line.strip().split(",")
            datas1.append(s[21:85])
			IDs.append(s[0])
    len2 = len(data) - len1
    labels2 = np.zeros(len2).tolist()
    labels = labels1 + labels2

    datas2 = list()
    with open(path2, "r") as lines:     #cnn feature
        for line in lines:
            s = line.strip().split(",")
            datas2.append(s[1:])
            if len(s[1:])!=288:
                print(len(s[1:]))
    
	datas3 = list()   
    with open(path31, "r") as lines:    #APAAC
        for line in lines:
            s = line.strip().split(",")
            datas3.append(s[21:])
    labels1 = np.ones(len1).tolist()
    
    with open(path32, "r") as lines:
        for line in lines:
            s = line.strip().split(",")
            datas1.append(s[21:])
    labels2 = np.zeros(len2).tolist()   
    labels = labels1 + labels2

    datas4 = list()   
    with open(path41, "r") as lines:   #188D
        for line in lines:
            s = line.strip().split(",")
            datas4.append(s[1:])
    labels1 = np.ones(len1).tolist()
    
    with open(path42, "r") as lines:
        for line in lines:
            s = line.strip().split(",")
            datas4.append(s[1:])
    labels2 = np.zeros(len2).tolist()   
    labels = labels1 + labels2
    print(np.array(datas1).shape)

    datas12 = np.concatenate((np.array(datas1), np.array(datas2)), axis=1).tolist()
    datas34 = np.concatenate((np.array(datas3), np.array(datas4)), axis=1).tolist()
    
    return datas12, datas34, labels

    results = np.column_stack((np.array(IDs),np.array(data12),np.array(labels)))
    with open(wf12, 'w', newline='') as fout:
        cin = csv.writer(fout)
        cin.writerows(results.tolist())

    results = np.column_stack((np.array(IDs),np.array(data34),np.array(labels)))
    with open(wf34, 'w', newline='') as fout:
        cin = csv.writer(fout)
        cin.writerows(results.tolist())

path11 = ''
path12 = ''
path2 = ''
path31 = ''
path32 = ''
path41 = ''
path42 = ''
wf12 = ''
wf34 = ''
datas, labels = data_load(path11,path12,path2,path31,path32,path41,path42,wf12,wf34)