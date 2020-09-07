# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 09:38:01 2020

@author: Z
"""

import numpy as np
import multiprocessing
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

def load_file(path1,path2,outpath):   
    datas = list() 
    
    with open(path1, "r") as lines:
        for line in lines:
            s = line.strip()
            if s[0] != '>':
                datas.append(s)
    labels1 = np.ones(len(datas)).tolist()
    
    with open(path2, "r") as lines:
        for line in lines:
            s = line.strip()
            if s[0] != '>':
                datas.append(s)

    labels2 = np.zeros(int(len(datas)/2)).tolist()   
    labels = labels1 + labels2
    
    texts = [list(map(str,s)) for s in datas]
    results = str()
    
    for i in datas:
        s = list(map(str,i))
        m = str()
        for j in s:
            m = m + j + ' '
        result = m + '\n'
        results = results + result
 
    f = open(outpath, 'w')
    f.writelines(results)
    f.close()
    return texts, labels

def word2vector(inp,outp1,outp2):  
    model = Word2Vec(LineSentence(inp), size=128, window=5, min_count=5, workers=multiprocessing.cpu_count())
    model.save(outp1)
    model.wv.save_word2vec_format(outp2, binary=False)

def test_model(outp1):
    aa_model = Word2Vec.load(outp1)

    testwords = ['A','F','C','D','E']
    for i in range(5):
        res = aa_model.most_similar(testwords[i])
        print(testwords[i])
        print(res)

# if __name__ == "__main__":
    # load_file()
    # inp = 'G:\赵思远\miRNA-encoded peptides\新实验\特征工程\CNN\\aas.txt'
    # outp1 = 'G:\赵思远\miRNA-encoded peptides\新实验\特征工程\CNN\\aas.model'
    # outp2 = 'G:\赵思远\miRNA-encoded peptides\新实验\特征工程\CNN\\aas.vector'
    # word2vector(inp,outp1,outp2)
    # test_model(outp1)