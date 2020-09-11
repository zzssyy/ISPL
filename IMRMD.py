#!/usr/bin/env python3
# -*- coding=utf-8 -*-

### 消除警告
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import math
import mic_new
from sklearn.ensemble import RandomForestClassifier
import logging
import sklearn.metrics
import pandas as pd
import datetime
import argparse
from format import pandas2arff
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from math import  ceil
import os
from scipy.io import arff
from sklearn.datasets import load_svmlight_file
from sklearn.datasets import dump_svmlight_file
import time

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", type=str, help="input file",required=True)
    parser.add_argument("-o", type=str, help="output the metrics file",default=None)
    parser.add_argument("-c", type=str, help="output the dimensionality reduction file")

    args = parser.parse_args()

    return args
	
class Dim_Rd(object):

    def __init__(self,file_csv,logger):
        self.file_csv=file_csv
        self.logger = logger
    def read_data(self):  #default csv

        def read_csv():
            self.df = pd.read_csv(self.file_csv,engine='python').dropna(axis=1)
            datas = np.array(self.df)
            self.datas = datas
            self.X = datas[:, 1:]
            self.y = datas[:, 0]

        file_type = self.file_csv.split('.')[-1]
        if file_type == 'csv':
            read_csv()

    def range_steplen(self, start=1, end=1, length=1):
        self.start = start
        self.end = end
        self.length = length

    def Randomforest(self,X,y):
        clf = RandomForestClassifier(random_state=1, n_estimators=100)

        ypred = sklearn.model_selection.cross_val_predict(clf, X, y, n_jobs=-1, cv=5)
        f1=sklearn.metrics.f1_score(y,ypred,average='weighted')
        precision = sklearn.metrics.precision_score(self.y, ypred, average='weighted')
        recall = sklearn.metrics.recall_score(self.y, ypred, average='weighted')
        roc = sklearn.metrics.roc_auc_score(self.y, ypred, average='weighted')
        acc = sklearn.metrics.accuracy_score(self.y,ypred)
       
        return acc,f1,precision,recall,roc,ypred

    def Result(self,seqmax,clf,features,csvfile):
        ypred = sklearn.model_selection.cross_val_predict(clf,self.X[:,seqmax],self.y, n_jobs=-1,cv=5)
        # print(ypred)
        confusion_matrix = sklearn.metrics.confusion_matrix(self.y,ypred,)

        TP = confusion_matrix[1, 1]
        TN = confusion_matrix[0, 0]
        FP = confusion_matrix[0, 1]
        FN = confusion_matrix[1, 0]
        logger.info('***confusion matrix***')

        s1 = '{:<15}'.format('')
        s2 = '{:<15}'.format('pos_class')
        s3 = '{:<15}'.format('neg_class')
        logger.info(s1+s2+s3)

        s1 = '{:<15}'.format('pos_class')
        s2 = 'TP:{:<15}'.format(TP)
        s3 = 'FN:{:<15}'.format(FN)
        logger.info(s1 + s2 + s3)

        s1 = '{:<15}'.format('neg_class')
        s2 = 'FP:{:<15}'.format(FP)
        s3 = 'TN:{:<15}'.format(TN)
        logger.info(s1+s2+s3)
        f1 = sklearn.metrics.f1_score(self.y,ypred,average='weighted')
        logger.info(('f1 ={:0.4f} '.format(f1)))
        acc = sklearn.metrics.accuracy_score(self.y,ypred,)
        logger.info('accuarcy = {:0.4f} '.format(acc))
        precision = sklearn.metrics.precision_score(self.y,ypred,average='weighted')
        logger.info('precision ={:0.4f} '.format(precision))
        recall = sklearn.metrics.recall_score(self.y,ypred,average ='weighted')
        logger.info(('recall ={:0.4f}'.format(recall)))
        auc = sklearn.metrics.roc_auc_score(self.y,ypred,average='weighted')
        logger.info('auc = {:0.4f}'.format(auc))

        columns_index=[0]
        columns_index.extend([i+1 for i in seqmax])
        data = np.concatenate((self.y.reshape(len(self.y),1), self.X[:, seqmax]),axis=1)
        features_list=(self.df.columns.values)


        ###实现-m参数
        if args.m == -1:
            pass
        else:
            columns_index = columns_index[0:args.m + 1]
            data = data[:, 0:args.m + 1]
        df = pd.DataFrame(data, columns=features_list[columns_index])
        df.iloc[0,:].astype(int)
        df.to_csv(csvfile, index=None)

    def Dim_reduction(self,features,features_sorted,outfile,csvfile):
        logger.info("Start dimension reduction ...")
        features_number=[]
        for value in features_sorted:
            features_number.append(features[value[0]]-1)
        stepSum=0
        max=0
        seqmax=[]
        predmax = []
        scorecsv=outfile
        print(features_number)
        with open(scorecsv,'w') as f:
            f.write('length,accuracy,f1,precision,recall,roc\n')
            for i in range(int(ceil((self.end-self.start)/self.length))+1):
                if (stepSum + self.start )<self.end:
                    n=stepSum + self.start
                else:
                    n=self.end

                stepSum+=self.length

                ix = features_number[self.start - 1:n]
                print(len(self.X[0]))
                print(ix)
                acc,f1,precision,recall,auc,ypred= self.Randomforest(self.X[:, ix], self.y)

                if args.t == "f1":
                    benchmark = f1
                elif args.t == "acc":
                    benchmark = acc
                elif args.t == "precision":
                    benchmark = precision
                elif args.t == "recall":
                    benchmark = recall
                elif args.t == "auc":
                    benchmark = auc

                if benchmark > max:
                    max = benchmark
                    seqmax = ix

                logger.info('length={} f1={:0.4f} accuarcy={:0.4f} precision={:0.4f} recall={:0.4f} auc={:0.4f} '.format(len(ix), f1,acc, precision,recall,auc))
                f.write('{},{:0.4f},{:0.4f},{:0.4f},{:0.4f},{:0.4f}\n'.format(len(ix), acc,f1,precision,recall,auc))


        logger.info('----------')
        logger.info('the max {} = {:0.4f}'.format(args.t,max))


        index_add1 = [x + 1 for x in seqmax]
        logger.info('{},length = {}'.format(self.df.columns.values[index_add1],len(seqmax)))
        logger.info('-----------')
        clf = RandomForestClassifier(random_state=1, n_estimators=100)
        self.Result(seqmax,clf,features,csvfile)
        logger.info('-----------')

    def run(self,inputfile):

        args = parse_args()
        file = inputfile

        outputfile =args_o
        csvfile = args.c
        mrmr_featurLen = args.m
        features_sorted,features=mrmd_run(file,self.logger)
        self.read_data()
        if int(args.e) == -1:
            args.e = len(pd.read_csv(file,engine='python').columns) - 1
        self.range_steplen(args.s, args.e, args.l)
        outputfile = os.getcwd()+os.sep+'Results'+os.sep+outputfile
        csvfile = os.getcwd()+os.sep+'Results'+os.sep+csvfile
        self.Dim_reduction(features,features_sorted,outputfile,csvfile)

def read_csv(filecsv):
    dataset=pd.read_csv(filecsv,engine='python').dropna(axis=1)
    features_name = dataset.columns.values.tolist()
    dataset=np.array(dataset)

    X=dataset[:,1:]
    y=dataset[:,0]
    return(X,y,features_name)

def calcE(X,coli,colj,flag):

    sum = 0.0
    if flag == 0:
        sum = np.sum((X[:,coli]-X[:,colj])**2)
        return math.sqrt(sum)
    elif flag == 1:
        s = np.linalg.norm(X[:,coli])*np.linalg.norm(X[:,colj])
        if s == 0:
            sum = 0.0
        else:
            sum = np.dot(X[:,coli],X[:,colj])/s
        return sum
    else:
        t = np.dot(X[:,coli],X[:,coli])+np.dot(X[:,colj],X[:,colj])-(np.linalg.norm(X[:,coli])*np.linalg.norm(X[:,colj]))
        if t == 0:
            sum ==0.0
        else:
            sum = np.dot(X[:,coli],X[:,colj])/t
        return sum
    
def Euclidean(X,n):

    Euclideandata=np.zeros([n,n])

    for i in range(n):
        for j in range(n):
            Euclideandata[i,j]=calcE(X,i,j,0)
            Euclideandata[j,i]=Euclideandata[i,j]
    Euclidean_distance=[]

    for i in range(n):
        sum = np.sum(Euclideandata[i,:])
        Euclidean_distance.append(sum/n)

    return Euclidean_distance

def varience(data,avg1,col1,avg2,col2):

    return np.average((data[:,col1]-avg1)*(data[:,col2]-avg2))

def Person(X,y,n):
    feaNum=n
    label_num=1
    PersonData=np.zeros([n])
    for i in range(feaNum):
        for j in range(feaNum,feaNum+label_num):
            average1 = np.average(X[:,i])
            average2 = np.average(y)
            yn=(X.shape)[0]
            y=y.reshape((yn,1))
            dataset = np.concatenate((X,y),axis=1)
            numerator = varience(dataset, average1, i, average2, j);
            denominator = math.sqrt(
                varience(dataset, average1, i, average1, i) * varience(dataset, average2, j, average2, j));
            if (abs(denominator) < (1E-10)):
                PersonData[i]=0
            else:
                PersonData[i]=abs(numerator/denominator)
    
    return list(PersonData)

def mrmd_run(filecsv,logger):
    logger.info('new mrmd start...')
    X,y,features_name=read_csv(filecsv)
    n=len(features_name)-1
    e=Euclidean(X,n)
    max_e = max(e)
    e = [x/max_e for x in e]
    logger.info('the Euclidean matrix is {}'.format(e))
    p = Person(X,y,n)
    max_p = max(p)
    p = [x/max_p for x in p]
    logger.info('the Person matrix is {}'.format(p))	
    datas = mic_new.readData(filecsv)
    mic=mic_new.multi_processing_mic(datas)
    max_mic = max(mic)
    mic = [x/max_mic for x in mic]
    logger.info('the mic matrix is {}'.format(mic))
    mrmrValue=[]
    for i,j,z in zip(p,e,mic):
        if abs(i-z) <= 0.1:
            mrmrValue.append(4*i/5+j/5)
        else:
            mrmrValue.append(4*i/15+j/5+8*z/15)
    mrmr_max=max(mrmrValue)
    mrmrValue = [x / mrmr_max for x in mrmrValue]
    mrmrValue = [(i,j) for i,j in zip(features_name[1:],mrmrValue)]   # features 和 mrmrvalue绑定
    mrmd=sorted(mrmrValue,key=lambda x:x[1],reverse=True)  #按mrmrValue 由大到小排序
    feature={}
    i=1
    for x in features_name:
        if x == 'class' or x == '0':
	        continue
        else:
            feature[x]=i
            i+=1
    print(feature)
    logger.info('new mrmd end.')
    return mrmd,feature

if __name__ == '__main__':
    print(datetime.datetime.now())
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    log_path = os.getcwd() + os.sep+'Logs'+os.sep
    rq = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
    log_name = log_path + rq + '.log'
    logfile = log_name
    fh = logging.FileHandler(logfile, mode='w')
    fh.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)


    formatter = logging.Formatter('[%(asctime)s]: %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.info("---imrmd start----")

    args = parse_args()
    file = args.i
    file_type = file.split('.')[-1]
    if file_type == 'csv':
        pass
    else:
        assert "format error"

    if int(args.e) == -1:
        args.e = len(pd.read_csv(file,engine='python').columns) - 1
        
    global args_o
    if args.o == None:

        args.o = ''.join(args.i.split('.')[:-1])+'.metrics.csv'
    args_o = args.o

    d=Dim_Rd(file,logger)
    d.run(inputfile=file)
    outputfile = os.getcwd() + os.sep + 'Results' + os.sep + args_o
    csvfile = os.getcwd() + os.sep + 'Results' + os.sep + args.c
    logger.info("The output by the terminal's log has been saved in the {}.".format(logfile))
    logger.info('metrics have been saved in the {}.'.format(outputfile))


    outputfile_file_type = args.c.split('.')[-1]
    if outputfile_file_type == 'csv':
        logger.info('Reduced dimensional dataset has been saved in the {}.'.format(csvfile))

    else:
        logger.info('Reduced dimensional dataset has been saved in the {}.'.format(csvfile))
    logger.info("---imrmd end---")
    print(datetime.datetime.now())


