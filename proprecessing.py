# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 00:02:14 2020

@author: Z
"""

from sklearn.model_selection import train_test_split
from keras.preprocessing import sequence
from gensim.models import Word2Vec 
import numpy as np

def tokenizer(texts, word_index):
    MAX_SEQUENCE_LENGTH = 101
    data = []
    for sentence in texts:
        new_txt = []
        for word in sentence:
            try:
                new_txt.append(word_index[word])  #把句子中的词语转化为index
            except:
                new_txt.append(0)
            
        data.append(new_txt)

    texts = sequence.pad_sequences(data, maxlen = MAX_SEQUENCE_LENGTH)  #使用keras的内置函数padding对齐句子,好处是输出numpy数组
    return texts

def split_data(texts, labels):
    x_train, x_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2)
    return x_train, x_test, y_train, y_test

def get_word2vec():
    myPath = 'G:\赵思远\miRNA-encoded peptides\新实验\特征工程\CNN\\aas.model' #本地词向量的地址
    Word2VecModel = Word2Vec.load(myPath) #读取词向量
    vocab_list = [word for word, Vocab in Word2VecModel.wv.vocab.items()]#存储所有的词语
    word_index = {" ": 0}#初始化 `[word : token]` ，后期tokenize语料库就是用该词典。
    word_vector = {} #初始化`[word : vector]`字典
    embeddings_matrix = np.zeros((len(vocab_list) + 1, Word2VecModel.vector_size))
    for i in range(len(vocab_list)):
        word = vocab_list[i]  #每个词语
        word_index[word] = i + 1 #词语：序号
        word_vector[word] = Word2VecModel.wv[word] #词语：词向量
        embeddings_matrix[i + 1] = Word2VecModel.wv[word]  #词向量矩阵
    
    
    return word_index, embeddings_matrix