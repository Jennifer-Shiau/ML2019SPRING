import numpy as np
import pandas as pd
import csv
import sys
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.layers import Conv2D, MaxPooling2D, Flatten, BatchNormalization, Activation
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model

def load_data(train_path):
    train = pd.read_csv(train_path, header = 0)
    train = np.array(train.values)

    Y_train = train[:, 0]
    Y_train = np_utils.to_categorical(Y_train, 7)
    # Y_train.shape = (28709, 7)

    feature = train[:, 1]
    n = len(feature)
    X_train = np.zeros((n, 48*48))
    for i in range(n):
        x = [int(f) for f in feature[i].split()]
        x = np.array(x)
        X_train[i] = x
    # X_train.shape = (28709, 2304)

    return X_train, Y_train

def train(X_train, Y_train):
    n = len(X_train)
    X_train /= 255
    X_train = X_train.reshape(n, 48, 48, 1)
    
    datagen = ImageDataGenerator(rotation_range = 10, horizontal_flip = True)
    datagen.fit(X_train)
    
    model = Sequential()

    model.add(Conv2D(64, (3, 3), padding = 'same', input_shape = (48, 48, 1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    
    model.add(Conv2D(64, (3, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(2, 2))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(128, (3, 3), padding = 'same'))
    model.add(Activation('relu'))
    
    model.add(Conv2D(128, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(2, 2))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(128, (3, 3), padding = 'same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    
    model.add(Conv2D(128, (3, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(2, 2))
    model.add(Dropout(0.25))

    model.add(Flatten())
    
    model.add(Dense(units = 512))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.35))

    model.add(Dense(units = 512))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.35))

    model.add(Dense(units = 7, activation = 'softmax'))
    
    # print(model.summary())

    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    history = model.fit_generator(datagen.flow(X_train, Y_train, batch_size=100), steps_per_epoch = len(X_train)/20, epochs = 20)
    
    model.save('my_model.h5')
    
    result = model.evaluate(X_train, Y_train, batch_size = 1000)
    print('\nTrain acc =', result[1])
    

if __name__ == '__main__':
    train_path = sys.argv[1]
    X_train, Y_train = load_data(train_path)

    train(X_train, Y_train)
    