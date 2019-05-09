import numpy as np
import pandas as pd
import csv
import json
import random
import sys
from keras.models import load_model
import jieba


def load_data(test_x_path):
    data = pd.read_csv(test_x_path, header = 0)
    data = np.array(data.values)
    test_x = data[:, 1]
    # test_x.shape = (20000,)

    return test_x

def get_seg(x):
    thre = 200
    seg_list = []
    
    for i in x:
        seg = jieba.lcut(i)
        if len(seg) > thre:
            seg = seg[:thre]
        seg_list.append(seg)
    
    return seg_list

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

def ensemble(output_path, x_test):
    model1 = load_model('model_2.h5')
    model2 = load_model('model_3.h5')
    model3 = load_model('model_5.hdf5')
    
    ans1 = model1.predict_classes(x_test)
    ans2 = model2.predict_classes(x_test)
    ans3 = model3.predict_classes(x_test)
    
    ans = vote(ans1, ans2, ans3)

    file = open(output_path, 'w+')
    out_file = csv.writer(file, delimiter = ',', lineterminator = '\n')
    out_file.writerow(['id', 'label'])
    for i in range(len(ans)):
        out_file.writerow([i, ans[i]])
    file.close()

def vote(ans1, ans2, ans3):
    random.seed(0)
    ans = np.zeros(ans1.shape)

    for i in range(len(ans1)):
        if ans1[i] == ans2[i] and ans1[i] == ans3[i]:
            ans[i] = ans1[i]
        elif ans1[i] == ans2[i]:
            ans[i] = ans1[i]
        elif ans2[i] == ans3[i]:
            ans[i] = ans2[i]
        elif ans1[i] == ans3[i]:
            ans[i] = ans1[i]
        else:
            r = random.randint(1, 3)
            if r == 1:
                ans[i] = ans1[i]
            elif r == 2:
                ans[i] = ans2[i]
            else:
                ans[i] = ans3[i]
    
    return ans.astype(int)


if __name__ == '__main__':
    jieba.load_userdict(sys.argv[2])

    test_x = load_data(sys.argv[1])
    x_test = load_test_data(test_x)
    
    ensemble(sys.argv[3], x_test)
