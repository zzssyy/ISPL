# -*- coding: utf-8 -*-

from tensorflow.keras.layers import Dense, Embedding, Dropout, Flatten, Conv1D, Input, concatenate
from wordvectrain import load_file
from proprecessing import tokenizer, split_data, get_word2vec
from tensorflow.keras.utils import to_categorical
from sklearn import metrics
from tensorflow.keras import Model
from tensorflow.keras.optimizers import SGD
import numpy as np
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
import os
import tensorflow as tf
from keras.engine import Layer, InputSpec
import sys

class KMaxPooling(Layer):
    def __init__(self, k=1, **kwargs):
        super().__init__(**kwargs)
        self.k = k

    def call(self, inputs):
        
        shifted_input = tf.transpose(inputs, [0, 2, 1])
        top_k = tf.nn.top_k(shifted_input, k=self.k, sorted=True, name=None)[0]
        return tf.transpose(top_k, [0, 2, 1])

#构建CNN模型
def CNN_model(x_train_padded_seqs,y_train,x_test_padded_seqs,y_test,embedding_matrix,top_k,num_filters,BASE_DATA_PATH):
    main_input = Input(shape=(101,), dtype='float64')
    EMBEDDING_DIM = 128
    MAX_SEQUENCE_LENGTH = 101
    embedder = Embedding(input_dim = len(embedding_matrix), 
                          output_dim = EMBEDDING_DIM, 
                          weights=[embedding_matrix], 
                          input_length=MAX_SEQUENCE_LENGTH, 
                          trainable=False 
                          )
    embed = embedder(main_input)
    print(embed.shape)
    
    cnn1 = Conv1D(32, 2, padding='same', strides=1, activation='relu')(embed)
    kmp = KMaxPooling()
    cnn1 = kmp.call(cnn1)
    cnn2 = Conv1D(32, 3, padding='same', strides=1, activation='relu')(embed)
    cnn2 = kmp.call(cnn2)
    cnn3 = Conv1D(32, 4, padding='same', strides=1, activation='relu')(embed)
    cnn3 = kmp.call(cnn3)
    cnn = concatenate([cnn1, cnn2, cnn3], axis=-1)
    flat = Flatten()(cnn)
    drop = Dropout(0.2)(flat)
    print(drop.shape)
    main_output = Dense(2, activation='sigmoid')(drop)
    model = Model(inputs=main_input, outputs=main_output)
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    BASE_DATA_PATH = BASE_DATA_PATH
    mylog_dir = os.path.join(BASE_DATA_PATH, "train_log")    
    one_hot_labels = to_categorical(y_train, num_classes=2) 
    
    index = np.arange(len(one_hot_labels))
    np.random.shuffle(index)
    x_train_padded_seqs = x_train_padded_seqs[index]
    one_hot_labels = one_hot_labels[index]
    
    tensorboard = TensorBoard(log_dir=mylog_dir)
    checkpoint = ModelCheckpoint(filepath='weights.best.hdf5',monitor='val_accuracy',mode='auto' ,save_best_only='True')
    callback_lists=[tensorboard,checkpoint]
    
    model.fit(x_train_padded_seqs, one_hot_labels, validation_split=0.1, batch_size=64, epochs=1000, callbacks=callback_lists)
    result = model.predict(x_test_padded_seqs)
    result_labels = np.argmax(result, axis=1)
    y_predict = list(map(float, result_labels))
    print('accuracy', metrics.accuracy_score(y_test, y_predict))
    print('recall', metrics.recall_score(y_test, y_predict))
    print('precision', metrics.precision_score(y_test, y_predict))
    print('f1-score:', metrics.f1_score(y_test, y_predict))

if __name__ == "__main__":
    path1 = sys.argv[1]
    path2 = sys.argv[2]
    outpath = sys.argv[3]   
    texts, labels = load_file(path1,path2,outpath)
    word_index, embedding_matrix = get_word2vec()
    texts = tokenizer(texts, word_index)
    x_train, x_test, y_train, y_test = split_data(texts, labels)
	BASE_DATA_PATH = sys.argv[4]
    CNN_model(x_train, y_train, x_test, y_test, embedding_matrix, top_k=3, num_filters=64,BASE_DATA_PATH)
    model_load()