import numpy as np
import pandas as pd
import json
import csv
import sys

import jieba
from gensim.models.word2vec import Word2Vec

from keras.models import Sequential
from keras.layers import Embedding, Dense, LSTM, Dropout
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from keras.models import load_model

np.random.seed(0)

def get_model():
    model = Word2Vec.load('word2vec.model')

    # vocab_dict = dict([(k, model.wv[k]) for k, v in model.wv.vocab.items()])
    # np.save('word2vec_dict.npy', vocab_dict)
    word2vec_weights = model.wv.syn0
    word2vec_weights = np.concatenate((np.zeros((1,250)), word2vec_weights), axis = 0)
    np.save('word2vec_weights.npy', word2vec_weights)
    idx2word_list = ['_'] + model.wv.index2word
    idx2word = dict([(x, idx2word_list[x]) for x in range(len(idx2word_list))])
    word2idx = dict([(v, k) for k, v in idx2word.items()])
    with open('word2idx.json', 'w+') as f:
        json.dump(word2idx, f)
    with open('idx2word.json', 'w+') as f:
        json.dump(idx2word, f)

def load_data(train_x_path, train_y_path, test_x_path):
    data = pd.read_csv(train_x_path, header = 0)
    data = np.array(data.values)
    train_x = data[:, 1]
    # train_x.shape = (120000,)

    data = pd.read_csv(train_y_path, header = 0)
    data = np.array(data.values)
    train_y = data[:, 1]
    # train_y.shape = (120000,)

    data = pd.read_csv(test_x_path, header = 0)
    data = np.array(data.values)
    test_x = data[:, 1]
    # test_x.shape = (20000,)

    return train_x, train_y, test_x

def get_seg(x):
    thre = 200
    seg_list = []
    
    for i in x:
        seg = jieba.lcut(i)
        if len(seg) > thre:
            seg = seg[:thre]
        seg_list.append(seg)

    return seg_list

def load_train_data(train_x, train_y):
    x = get_seg(train_x)
    max_len = 0

    word2idx = json.load(open('word2idx.json'))

    # run through each sentence
    for i in range(len(x)):
        # run through each word
        for j in range(len(x[i])):
            try:
                x[i][j] = word2idx[x[i][j]]
            except:
                x[i][j] = 0
        x[i] = np.array(x[i], dtype = int)
        if len(x[i]) > max_len:
            max_len = len(x[i])
    
    x_train = np.zeros((len(x), max_len), dtype = int)
    for i in range(x_train.shape[0]):
        x_train[i][:x[i].shape[0]] = x[i]
    
    y_train = np_utils.to_categorical(train_y, 2)

    return x_train, y_train

def load_test_data(test_x):
    x = get_seg(test_x)
    max_len = 0

    word2idx = json.load(open('word2idx.json'))

    # run through each sentence
    for i in range(len(x)):
        # run through each word
        for j in range(len(x[i])):
            try:
                x[i][j] = word2idx[x[i][j]]
            except:
                x[i][j] = 0
        x[i] = np.array(x[i], dtype = int)
        if len(x[i]) > max_len:
            max_len = len(x[i])
    
    x_test = np.zeros((len(x), max_len), dtype = int)
    for i in range(x_test.shape[0]):
        x_test[i][:x[i].shape[0]] = x[i]
    
    return x_test
    
def split_valid(x_all, y_all, percentage):
    data_size = len(x_all)
    valid_data_size = round(data_size * percentage)
    x_all, y_all = shuffle(x_all, y_all)
    
    x_valid = x_all[0:valid_data_size]
    y_valid = y_all[0:valid_data_size]
    x_train = x_all[valid_data_size:]
    y_train = y_all[valid_data_size:]
    return x_train, y_train, x_valid, y_valid

def shuffle(X, Y):
    rand = np.arange(len(X))
    np.random.shuffle(rand)
    return X[rand], Y[rand]

def rnn_model():
    word2vec_weights = np.load('word2vec_weights.npy')
    
    model = Sequential()
    model.add(Embedding(word2vec_weights.shape[0], 
                        word2vec_weights.shape[1], 
                        weights=[word2vec_weights], 
                        trainable=False))
    model.add(Dropout(0.25))
    model.add(LSTM(256, dropout=0.25, recurrent_dropout=0.25, 
                   activation='sigmoid', inner_activation='hard_sigmoid',
                   implementation=2))
    model.add(Dense(2, activation='softmax'))
    
    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    
    # print(model.summary())
    
    return model

def train(x_all, y_all):
    x_train, y_train, x_valid, y_valid = split_valid(x_all, y_all, 0.1)
    
    model = rnn_model()
    
    history = model.fit(x_train, y_train, batch_size = 128, epochs = 40, 
                        validation_data = (x_valid, y_valid))
    
    result = model.evaluate(x_train, y_train, batch_size = 1000)
    print('\nTrain acc =', result[1])
    
    result = model.evaluate(x_valid, y_valid, batch_size = 1000)
    print('\nValid acc =', result[1])
    
    model.save('my_model.h5')
    
if __name__ == '__main__':
    jieba.load_userdict(sys.argv[4])
    get_model()

    train_x, train_y, test_x = load_data(sys.argv[1], sys.argv[2], sys.argv[3])
    
    x_train, y_train = load_train_data(train_x, train_y)
    train(x_train, y_train)
    