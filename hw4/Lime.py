import numpy as np
import pandas as pd
from keras.models import load_model
from keras.utils import np_utils
from lime import lime_image
from skimage.color import gray2rgb, rgb2gray
from skimage.segmentation import slic
import matplotlib.pyplot as plt
import sys

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

def predict(input):
    input = rgb2gray(input)
    pred = model.predict(input.reshape(-1, 48, 48, 1))
    return pred

def segmentation(input):
    return slic(input)

def plot_lime(path, X_train, Y_train):
    n = len(X_train)
    X_train /= 255
    X_train = X_train.reshape(n, 48, 48, 1)
    Y_train_label = np.argmax(Y_train, axis = 1)

    idx_list = [307, 416, 5, 7, 390, 128, 112]

    x = X_train[idx_list]
    x_label = Y_train_label[idx_list]
    
    x_rgb = np.zeros((7, 48, 48, 3))
    for i in range(7):
        x_rgb[i] = gray2rgb(x[i].reshape(48, 48))

    # Initiate explainer instance
    explainer = lime_image.LimeImageExplainer()
    
    for i in range(7):
        # print('class %d ...' % i)
        # Get the explaination of an image
        explanation = explainer.explain_instance(
                                    image = x_rgb[i], 
                                    classifier_fn = predict, 
                                    segmentation_fn = segmentation
                                )

        # Get processed image
        image, mask = explanation.get_image_and_mask(
                                    label = x_label[i],
                                    positive_only = False,
                                    hide_rest = False,
                                    num_features = 5,
                                    min_weight = 0.0
                                )

        # save the image
        fig, ax = plt.subplots()
        ax = ax.imshow(image)
        fig.savefig('%sfig3_%d.jpg' % (path, i))

if __name__ == '__main__':
    train_path = sys.argv[1]
    X_train, Y_train = load_data(train_path)
    model = load_model('model_7.h5')

    np.random.seed(123)
    output_path = sys.argv[2]
    plot_lime(output_path, X_train, Y_train)
