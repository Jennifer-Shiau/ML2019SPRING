import numpy as np
import pandas as pd
import csv
import sys

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
    percentage = 0.1
    X_train, Y_train, X_valid, Y_valid = split_valid(X_all, Y_all, percentage)
    # X_train, Y_train = X_all, Y_all

    w = np.zeros((len(X_train[0]),))
    b = np.zeros((1,))
    
    lr = 1
    w_prev = np.zeros(len(w))
    b_prev = 0

    data_size = len(X_train)
    batch_size = 64
    step_num = data_size // batch_size
    iteration = 1000
    test_iter = 100

    loss = 0.0
    for i in range(1, iteration+1):
        if i % test_iter == 0:
            # print('===== iteration %d =====' % i)
            # loss_ave = loss / (test_iter * data_size)
            # print('loss = %f' % loss_ave)
            loss = 0
            # acc = valid_result(w, b, X_valid, Y_valid)
            # print('valid acc = %f' % acc)
        
        X_train, Y_train = shuffle(X_train, Y_train)

        for j in range(step_num):
            X = X_train[batch_size*j:batch_size*(j+1)]
            Y = Y_train[batch_size*j:batch_size*(j+1)]

            z = np.dot(X, w.T) + b
            y = sigmoid(z)

            cross_entropy = -1 * (np.dot(np.squeeze(Y), np.log(y)) + np.dot(np.squeeze(1-Y), np.log(1-y)))
            loss += cross_entropy

            w_grad = np.dot(X.T, y - np.squeeze(Y)) / batch_size + 0.001 * np.sign(w) / batch_size
            # w_grad = np.dot(X.T, y - np.squeeze(Y)) / batch_size + 0.001 * w / batch_size
            b_grad = np.sum(y - np.squeeze(Y)) / batch_size

            w_prev += w_grad**2
            b_prev += b_grad**2
            w_ada = np.sqrt(w_prev)
            b_ada = np.sqrt(b_prev)
            w -= lr * w_grad / w_ada
            b -= lr * b_grad / b_ada

    return w, b

def predict(w, b, X_test, output_path):
    # w = np.load('./logistic/w.npy')
    # b = np.load('./logistic/b.npy')
    
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

    feature = [2, 3, 5, 35, 41, 1, 7, 8, 12, 16, 21, 25, 27, 31, 39, 45, 48, 63, 65, 68, 74, 78, 80, 83, 85, 88, 89, 98, 101, 105]
    # feature = [0, 1, 3, 4, 5, 6, 8, 10, 20, 21, 27, 34, 40, 43, 50, 53, 55, 67, 68, 70, 75, 80, 87, 94, 95, 98, 101, 104]
    train_f = X_train[:, feature]
    test_f = X_test[:, feature]

    de = [4, 9, 15, 17, 18, 19, 22, 26, 28, 30, 32, 34, 36, 37, 38, 40, 42, 46, 47, 49, 52, 54, 56, 58, 60, 61, 64, 70, 75, 77, 81, 82, 86, 87, 90, 92, 93, 94, 96, 97]
    # de = [2, 14, 15, 18, 22, 24, 30, 36, 38, 39, 41, 51, 56, 57, 58, 59, 62, 63, 64, 69, 76, 81, 83, 89, 90, 93, 102]
    train_d = np.delete(X_train, de, 1)
    test_d = np.delete(X_test, de, 1)

    X_train = np.concatenate((X_train, train_d**2, train_d**3, train_f**4, train_f**5, np.log(1 + X_train)), axis = 1)
    X_test = np.concatenate((X_test, test_d**2, test_d**3, test_f**4, test_f**5, np.log(1 + X_test)), axis = 1)
    
    X_train, X_test = normalize(X_train, X_test)

    w, b = train(X_train, Y_train)
    # np.save('./logistic/w.npy', w)
    # np.save('./logistic/b.npy', b)

    predict(w, b, X_test, output_path)
