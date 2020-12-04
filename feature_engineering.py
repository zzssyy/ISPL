# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import sys

#按模态合并数据
def data_load(path11,path12,path2,path31,path32,path41,path42,wf12,wf34):   
    datas1 = list()
    IDs = list()
    with open(path11, "r") as lines:  # k-mer
        for line in lines:
            s = line.strip().split(",")  
            datas1.append(s[21:85])
            IDs.append(s[0])  
    len1 = len(datas1)
    labels1 = np.ones(len1).tolist()

    with open(path12, "r") as lines:
        for line in lines:
            s = line.strip().split(",")
            datas1.append(s[21:85])
            IDs.append(s[0])

    len2 = len(datas1) - len1
    labels2 = np.zeros(len2).tolist()
    labels = labels1 + labels2

    datas2 = list()
    with open(path2, "r") as lines:  #cnn feature
        for line in lines:
            s = line.strip().split(",")
            datas2.append(s[1:])      

    datas3 = list()
    with open(path31, "r") as lines:  # APAAC
        for line in lines:
            s = line.strip().split(",")
            datas3.append(s[21:])
    labels1 = np.ones(len1).tolist()

    with open(path32, "r") as lines:
        for line in lines:
            s = line.strip().split(",")
            datas3.append(s[21:])
    labels2 = np.zeros(len2).tolist()
    labels = labels1 + labels2

    datas4 = list()
    with open(path41, "r") as lines:  # 188D
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

    datas12 = np.concatenate((np.array(datas1), np.array(datas2)), axis=1).tolist()
    datas34 = np.concatenate((np.array(datas3), np.array(datas4)), axis=1).tolist()

    results = np.column_stack((np.array(IDs), np.array(datas12), np.array(labels)))
    print(results.shape)
    with open(wf12, 'w', newline='') as fout:
        cin = csv.writer(fout)
        cin.writerows(results.tolist())

    results = np.column_stack((np.array(IDs), np.array(datas34), np.array(labels)))
    print(results.shape)
    with open(wf34, 'w', newline='') as fout:
        cin = csv.writer(fout)
        cin.writerows(results.tolist())

	
	return datas12, datas34, labels

path11 = sys.argv[1]
path12 = sys.argv[2]
path2 = sys.argv[3]
path31 = sys.argv[4]
path32 = sys.argv[5]
path41 = sys.argv[6]
path42 = sys.argv[7]
wf12 = sys.argv[8]
wf34 = sys.argv[9]
datas12, datas34, labels = data_load(path11,path12,path2,path31,path32,path41,path42,wf12,wf34)