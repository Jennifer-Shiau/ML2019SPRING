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

def deprocess_image(x):
    # normalization
    x = (x - np.mean(x)) / (np.std(x) + 1e-5)
    x *= 0.1
    
    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)
    
    # convert to RGB array
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def filter(model, random_img, test_img, layer_name, path, plot_x, plot_y):
    layer_dict = dict([(layer.name, layer) for layer in model.layers])
    layer_output = layer_dict[layer_name].output

    # visualize filter - random image
    img_ascs = list()
    for filter_index in range(layer_output.shape[3]):
        loss = K.mean(layer_output[:, :, :, filter_index])
        grads = K.gradients(loss, model.input)[0]
        grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
        iterate = K.function([model.input], [loss, grads])

        img_asc = np.array(random_img)
        
        step = 5
        epoch = 20
        for i in range(epoch):
            loss_value, grads_value = iterate([img_asc])
            img_asc += grads_value * step

        img_asc = img_asc[0]
        img_ascs.append(deprocess_image(img_asc).reshape((48, 48)))
        
    fig, ax = plt.subplots(plot_x, plot_y, figsize = (2*plot_y, 2*plot_x))
    fig.suptitle('Filters of layer %s' % (layer_name))
    fig.tight_layout(pad = 0.3, rect = [0, 0, 1, 0.95])
    for (x, y) in [(i, j) for i in range(plot_x) for j in range(plot_y)]:
        ax[x, y].imshow(img_ascs[x * plot_y + y], cmap = 'Oranges')
    fig.savefig('%sfig2_1.jpg' % path)

    # filter output, given test image
    input_img = model.input
    fn = K.function([input_img, K.learning_phase()], [layer_output])
    output_img = fn([test_img, 0])
    output_img = deprocess_image(np.array(output_img))
    # output_img.shape = (1, 1, 48, 48, 64)
    
    fig2, ax2 = plt.subplots(plot_x, plot_y, figsize = (2*plot_y, 2*plot_x))
    fig2.suptitle('Output of layer %s' % (layer_name))
    fig2.tight_layout(pad = 0.3, rect = [0, 0, 1, 0.95])
    for (x, y) in [(i, j) for i in range(plot_x) for j in range(plot_y)]:
        ax2[x, y].imshow(output_img[0, 0, :, :, x * plot_y + y], cmap = 'Oranges')
    fig2.savefig('%sfig2_2.jpg' % path)


def filter_visualization(path, model, X_train):
    random_img = np.random.random((1, 48, 48, 1)) * 20 + 128

    X_train /= 255
    test_img = X_train[307].reshape(1, 48, 48, 1)

    filter(model, random_img, test_img, 'activation_1', path, 4, 16)
    
if __name__ == '__main__':
    train_path = sys.argv[1]
    X_train = load_data(train_path)
    model = load_model('model_7.h5')

    output_path = sys.argv[2]
    filter_visualization(output_path, model, X_train)
