import csv
import numpy as np

def data_load(path1, path2):
    datas1 = list()
    with open(path1, "r") as lines:
        for line in lines:
            s = line.strip().split(",")
            datas1.append(s[1:])
    datas2 = list()
    with open(path2, "r") as lines:
        for line in lines:
            s = line.strip().split(",")
            datas2.append(s[1:])
    return datas1, datas2

def reduce(datas1, datas2):
    new_data1 = list()
    new_data2 = list()
    datas1 = np.array(datas1).T
    datas2 = np.array(datas2).T
    s1 = [0,144,128,152,248,100,133,84,344,224,119,99,120,229,347,351,255,196,93,251,145,338,112,173,260,326,63,230,321,269,17,159,292,2,242,310,319,303,257,294,308,21,32,61,295,164,214,77,71,207,281,161,198,225,5,134,212,185,48,282,59,223,155]
    s2 = [0,134,184,104,89,154,79,124,11,169,6,1,35,31,139,42,29,20,27,94,8,26,32,54,5,123,73,163,109,59,37,68,3,148,12,13,118,174,133,119,4,99,38,2,93,58,53,153,23,103,41,88,144,57,168,178,33,149,129,147,55,25,140,155,74,159,43,7,84,145,132,61,48,56,80,15,175,78,131,160,60,114,117,122,116,44,72,77,100,161,22,10,39,85,177,179,105,130,167,75,176,63,146,136,162,76]

    for i in range(0, len(datas1)):
        if i in s1:
            new_data1.append(datas1[i])
        else:
            continue
    for i in range(0, len(datas2)):
        if i in s2:
            new_data2.append(datas2[i])
        else:
            continue
    return np.array(new_data1).T, np.array(new_data2).T

def save_data(datas1, datas2, path3, path4):
    with open(path3, 'w', newline='') as fout:
        cin = csv.writer(fout)
        cin.writerows(datas1.tolist())

    with open(path4, 'w', newline='') as fout:
        cin = csv.writer(fout)
        cin.writerows(datas2.tolist())

# def run(path1, path2, path3, path4):
#     datas1, datas2 = data_load(path1, path2)
#     datas1, datas2 = reduce(datas1, datas2)
#     save_data(datas1, datas2, path3, path4)

def run(datas1, datas2):
    # save_data(datas1, datas2, "C:\\Users\Wo1verien\Desktop\\bioinformatics-master\cnn-mer.csv", "C:\\Users\Wo1verien\Desktop\\bioinformatics-master\\188-ac.csv")
    # exit(0)
    datas1, datas2 = reduce(datas1, datas2)
    return datas1, datas2