import numpy as np
import pandas as pd
import csv
import random
import sys
from keras.models import load_model

def load_data(test_path):
    test = pd.read_csv(test_path, header = 0)
    test = np.array(test.values)

    feature = test[:, 1]
    n = len(feature)
    X_test = np.zeros((n, 48*48))
    for i in range(n):
        x = [int(f) for f in feature[i].split()]
        x = np.array(x)
        X_test[i] = x
    # X_test.shape = (7178, 2304)

    return X_test

def ensemble(output_path, X_test):
    model1 = load_model('model_7.h5')
    model2 = load_model('model_8.h5')
    model3 = load_model('model_9.h5')

    n = len(X_test)
    X_test /= 255
    X_test = X_test.reshape(n, 48, 48, 1)

    ans1 = model1.predict_classes(X_test)
    ans1 = ans1.squeeze()
    ans2 = model2.predict_classes(X_test)
    ans2 = ans2.squeeze()
    ans3 = model3.predict_classes(X_test)
    ans3 = ans3.squeeze()
    
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
    X_test = load_data(sys.argv[1])
    ensemble(sys.argv[2], X_test)
