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
                new_txt.append(word_index[word])
            except:
                new_txt.append(0)
            
        data.append(new_txt)

    texts = sequence.pad_sequences(data, maxlen=MAX_SEQUENCE_LENGTH)
    return texts

def split_data(texts, labels):
    x_train, x_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2)
    return x_train, x_test, y_train, y_test

def get_word2vec(path):
    myPath = path
    Word2VecModel = Word2Vec.load(myPath)
    vocab_list = [word for word, Vocab in Word2VecModel.wv.vocab.items()]
    word_index = {" ": 0}
    word_vector = {}
    embeddings_matrix = np.zeros((len(vocab_list) + 1, Word2VecModel.vector_size))
    for i in range(len(vocab_list)):
        word = vocab_list[i]
        word_index[word] = i + 1
        word_vector[word] = Word2VecModel.wv[word]
        embeddings_matrix[i + 1] = Word2VecModel.wv[word]
    
    
    return word_index, embeddings_matrix