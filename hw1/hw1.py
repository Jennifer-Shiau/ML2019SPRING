import numpy as np
import csv
import sys

def load_data(file_name):
    file = open(file_name, 'r', encoding='big5')
    train_data = csv.reader(file, delimiter = ',')
    data = [[] for i in range(18)]      # 記錄18種觀測數據
    n_row = 0
    
    for row in train_data:
        if n_row != 0:
            for i in range(3, 27, 1):   # 第 3 ~ 26 欄是24小時的資料
                if row[i] != 'NR':
                    data[(n_row-1)%18].append(float(row[i]))
                else:
                    data[(n_row-1)%18].append(float(0))
        n_row += 1

    file.close()
    
    x = []
    y = []
    
    for i in range(12):
        for j in range(471):
            x.append([])
            for k in range(18):
                for s in range(9):
                    x[471*i+j].append(data[k][480*i+j+s])
            y.append(data[9][480*i+j+9])
            
    x = np.array(x)
    y = np.array(y)
    
    return x, y

def adagrad(x, y):
    x = np.concatenate((np.ones((x.shape[0], 1)), x), axis = 1)
    x_t = x.transpose()
    
    w = np.zeros(x.shape[1])
    iteration = 100000
    lr = 1
    lamda = 0.00
    pre_gra = np.zeros(x.shape[1])
    
    for i in range(1, iteration+1, 1):
        _y = np.dot(x, w)
        loss = _y - y + lamda * np.sum(w**2)
        cost = np.sqrt(np.sum(loss**2) / len(x))
        gra = 2 * np.dot(x_t, loss) + 2 * lamda * w
        pre_gra += gra**2
        ada = np.sqrt(pre_gra)
        w -= lr * gra / ada
        
        if i % 10000 == 0:
            print("iteration %d: cost = %f" % (i, cost))
            
    return w

def load_file(input_file):
    file = open(input_file, 'r', encoding='big5')
    test_data = csv.reader(file, delimiter = ',')
    x_test = []
    n_row = 0
    
    for row in test_data:
        if n_row % 18 == 0:
            x_test.append([])
        for i in range(2, 11, 1):
            if row[i] != 'NR':
                x_test[n_row//18].append(float(row[i]))
            else:
                x_test[n_row//18].append(float(0))
        n_row += 1
        
    x_test = np.array(x_test)
    
    return x_test

def predict(x_test):
    w = np.load('model.npy')
    x_test = np.concatenate((np.ones((x_test.shape[0], 1)), x_test), axis = 1)
    y_test = np.dot(x_test, w)
    
    for i in range(len(y_test)):
        if y_test[i] < 0:
            y_test[i] = 0
    
    return y_test

def output(y_test, output_file):
    file = open(output_file, 'w+')
    out_file = csv.writer(file, delimiter = ',', lineterminator = '\n')
    out_file.writerow(['id', 'value'])
    for i in range(len(y_test)):
        out_file.writerow(['id_'+str(i), y_test[i]])
    file.close()


if __name__ == '__main__':
#    x, y = load_data('./data/train.csv')
#    w = adagrad(x, y)
#    np.save('model.npy', w)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    x_test = load_file(input_file)
    y_test = predict(x_test)
    output(y_test, output_file)
    