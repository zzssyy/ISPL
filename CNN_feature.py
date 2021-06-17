# -*- coding: utf-8 -*-

from tensorflow.python.keras.models import load_model
from tensorflow.keras import backend as K
import numpy as np
from proprecessing import tokenizer, get_word2vec

#load model
def model_load():
    model = load_model('weights.best.hdf5')
    return model

#cnn feature extraction
def get_feature(model, X):
    mid_layer = K.function([model.layers[0].input],
                       [model.layers[15].output])
    mid_layer_output = mid_layer([X])[0]
    print(mid_layer_output.shape)
    return mid_layer_output

#load model
def data_load(path):
    datas = list()
    IDs = list()
    with open(path, "r") as lines:
        for line in lines:
            s = line.strip()
            if s[0] != '>':
                datas.append(s)
            else:
                IDs.append(s[1:])
    # len1 = len(datas)
    # labels1 = np.ones(m).tolist()
    # labels2 = np.zeros(len(datas)-len1).tolist()
    #
    # labels = labels1 + labels2
    texts = [list(map(str, s)) for s in datas]
    # return texts, labels, IDs
    return texts, IDs


def run(path):
    text, label, IDs = data_load(path)
    word_index, embedding_matrix = get_word2vec(path='aas.model')
    texts = tokenizer(text, word_index)
    model = model_load()
    feature = get_feature(model, texts)
    print("starting to save CNN features...")
    results = np.column_stack((np.array(IDs), np.array(feature)))
    return results