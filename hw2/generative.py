import numpy as np
import pandas as pd
import csv
import sys
from numpy.linalg import inv

def load_data(X_train_path, Y_train_path, X_test_path):
    X_train = pd.read_csv(X_train_path, header = 0)
    X_train = np.array(X_train.values)
    Y_train = pd.read_csv(Y_train_path, header = 0)
    Y_train = np.array(Y_train.values)
    X_test = pd.read_csv(X_test_path, header = 0)
    X_test = np.array(X_test.values)

    # X_train.shape = (32561, 106)
    # Y_train.shape = (32561, 1)
    # X_test.shape = (16281, 106)
    return X_train, Y_train, X_test

def normalize(X_train, X_test):
    X_all = np.concatenate((X_train, X_test), axis = 0)
    mu = np.mean(X_all, axis = 0)
    sigma = np.std(X_all, axis = 0)
    mu = np.tile(mu, (X_all.shape[0], 1))
    sigma = np.tile(sigma, (X_all.shape[0], 1))
    X_all = (X_all - mu) / sigma
    X_train = X_all[0:len(X_train)]
    X_test = X_all[len(X_train):]

    return X_train, X_test

def shuffle(X, Y):
    np.random.seed(123564)
    rand = np.arange(len(X))
    np.random.shuffle(rand)
    return X[rand], Y[rand]

def split_valid(X_all, Y_all, percentage):
    data_size = len(X_all)
    valid_size = round(data_size * percentage)
    X_all, Y_all = shuffle(X_all, Y_all)

    X_valid = X_all[0:valid_size]
    Y_valid = Y_all[0:valid_size]
    X_train = X_all[valid_size:]
    Y_train = Y_all[valid_size:]
    return X_train, Y_train, X_valid, Y_valid

def sigmoid(z):
    result = 1.0 / (1.0 + np.exp(-z))
    result = np.clip(result, 1e-8, 1-(1e-8))
    return result

def valid_result(w, b, X_valid, Y_valid):
    z = np.dot(X_valid, w.T) + b
    f = sigmoid(z)
    y = np.rint(f)

    result = (np.squeeze(Y_valid) == y)
    acc = np.sum(result) / len(result)
    return acc

def train(X_all, Y_all):
    # percentage = 0.1
    # X_train, Y_train, X_valid, Y_valid = split_valid(X_all, Y_all, percentage)
    X_train, Y_train = X_all, Y_all
    
    X1 = []     # label = 1
    X2 = []     # label = 0

    for i in range(len(Y_train)):
        if Y_train[i] == 1:
            X1.append(X_train[i])
        else:
            X2.append(X_train[i])
    
    X1 = np.array(X1)
    X2 = np.array(X2)
    
    n1 = len(X1)
    n2 = len(X2)

    mu1 = np.mean(X1, axis = 0)
    sigma1 = np.zeros((len(X_train[0]), len(X_train[0])))
    for i in range(n1):
        sigma1 += np.outer(X1[i] - mu1, X1[i] - mu1)
    sigma1 /= n1

    mu2 = np.mean(X2, axis = 0)
    sigma2 = np.zeros((len(X_train[0]), len(X_train[0])))
    for i in range(n2):
        sigma2 += np.outer(X2[i] - mu2, X2[i] - mu2)
    sigma2 /= n2

    sigma = (n1 * sigma1 + n2 * sigma2) / (n1 + n2)
    sigma_inv = inv(sigma)
    
    w = np.dot(mu1 - mu2, sigma_inv)
    b = - 0.5 * np.dot(np.dot(mu1, sigma_inv), mu1.T) + 0.5 * np.dot(np.dot(mu2, sigma_inv), mu2.T) + np.log(n1/n2)

    # acc = valid_result(w, b, X_valid, Y_valid)
    # print('valid acc = %f' % acc)

    return w, b

def predict(w, b, X_test, output_path):
    # w = np.load('./generative/w.npy')
    # b = np.load('./generative/b.npy')
    
    z = np.dot(X_test, w.T) + b
    f = sigmoid(z)
    y = np.rint(f).astype(int)
    
    file = open(output_path, 'w+')
    out_file = csv.writer(file, delimiter = ',', lineterminator = '\n')
    out_file.writerow(['id', 'label'])
    for i in range(len(y)):
        out_file.writerow([i+1, y[i]])
    file.close()

if __name__ == '__main__':
    X_train_path = sys.argv[3]
    Y_train_path = sys.argv[4]
    X_test_path = sys.argv[5]
    output_path = sys.argv[6]

    X_train, Y_train, X_test = load_data(X_train_path, Y_train_path, X_test_path)
    X_train, X_test = normalize(X_train, X_test)

    w, b = train(X_train, Y_train)
    # np.save('./generative/w.npy', w)
    # np.save('./generative/b.npy', b)

    predict(w, b, X_test, output_path)
