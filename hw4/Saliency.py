import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import load_model
from keras import backend as K
import sys

def load_data(train_path):
    train = pd.read_csv(train_path, header = 0)
    train = np.array(train.values)

    feature = train[:, 1]
    n = len(feature)
    X_train = np.zeros((n, 48*48))
    for i in range(n):
        x = [int(f) for f in feature[i].split()]
        x = np.array(x)
        X_train[i] = x
    # X_train.shape = (28709, 2304)

    return X_train

def plot_saliency(mode, path, X_train, model):
    n = len(X_train)
    X_train /= 255
    X_train = X_train.reshape(n, 48, 48, 1)
    
    input_img = model.input

    classes = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    idx_list = [307, 416, 5, 401, 390, 128, 112]

    for c, idx in enumerate(idx_list):
        target = model.output[:, c]
        gradient = K.gradients(target, input_img)[0]
        fn = K.function([input_img, K.learning_phase()], [gradient])

        grads = fn([X_train[idx].reshape(-1, 48, 48, 1), 0])[0].reshape(48, 48, -1)
        grads = np.abs(grads)
    
        # normalization
        grads = (grads - np.mean(grads)) / (np.std(grads) + 1e-4)
        grads *= 0.1

        # clip to [0, 1]
        grads += 0.5
        grads = np.clip(grads, 0, 1)

        saliency = grads.reshape(48, 48)
        
        if mode == 'all':
            # original image
            img = (X_train[idx]*255).reshape(48, 48)
            fig = plt.figure(figsize = (10, 3))
            ax = fig.add_subplot(1, 3, 1)
            ax.imshow(img, cmap = 'gray')
            plt.tight_layout(pad = 2)
        
            # saliency map
            ax = fig.add_subplot(1, 3, 2)
            ax2 = ax.imshow(saliency, cmap = 'jet')
            plt.colorbar(ax2)
            plt.tight_layout(pad = 2)

            # mask
            thre = 0.5
            mask = img
            mask[np.where(saliency <= thre)] = np.mean(mask)

            ax = fig.add_subplot(1, 3, 3)
            ax3 = ax.imshow(mask, cmap='gray')
            plt.colorbar(ax3)
            plt.tight_layout(pad = 2)
        
            fig.suptitle('Class: %s (#%d)' % (classes[c], idx))
            plt.savefig('%s1_%d.png' % (path, c))

        else:
            fig, ax = plt.subplots()
            ax = ax.imshow(saliency, cmap = 'jet')
            plt.colorbar(ax)
            plt.tight_layout(pad = 2)
            plt.savefig('%sfig1_%d.jpg' % (path, c))


if __name__ == '__main__':
    train_path = sys.argv[1]
    X_train = load_data(train_path)
    model = load_model('model_7.h5')

    # test_path = 'test/'
    # plot_saliency('all', test_path, X_train, model)

    output_path = sys.argv[2]
    plot_saliency('saliency', output_path, X_train, model)
    