import numpy as np
from numpy.linalg import svd
from skimage.io import imread, imsave
import os
import sys


def load_data(img_dir):
    X = np.zeros((415, 600*600*3))
    for i in range(415):
        fn = os.path.join(img_dir, str(i)+'.jpg')
        im = imread(fn).reshape(600*600*3,)
        X[i] = im

    return X

def mean_face(X):
    mu = np.mean(X, axis=0)
    meanface = mu.reshape(600, 600, 3).astype(np.uint8)
    imsave('pca/mean.jpg', meanface)

    return mu

def get_SVD(X, mu):
    X_hat = X - mu
    U, s, V_t = svd(X_hat, full_matrices=False)
    V = V_t.T

    return U, s, V

def compute_ratio(s):
    for i in range(5):
        r = s[i] / np.sum(s)
        print('[%d]  %.1f%%' % (i+1, 100*r))

def plot_eigenface(n, V):
    for i in range(n):
        vec = V[:, i]
        im = to_img(vec).reshape(600, 600, 3)
        imsave('pca/eigenface_%d.jpg' % (i+1), im)

def to_img(M):
    M -= np.min(M)
    M /= np.max(M)
    M = (M * 255).astype(np.uint8)

    return M

def plot_reconstruct(X, mu, five_eigenfaces):
    index = [3, 14, 159, 265, 358]
    for i in index:
        reconstruct('%d.jpg' % i, 'pca/reconstruct_%d.jpg' % i, X, mu, five_eigenfaces)

def reconstruct(ori_img, rec_img, X, mu, five_eigenfaces):
    idx = int(ori_img[:-4])
    x = (X - mu)[idx]
    p = np.dot(five_eigenfaces.T, x)

    x_r = mu.copy()
    for i in range(5):
        x_r += p[i] * five_eigenfaces[:, i]
    
    x_r = to_img(x_r).reshape(600, 600, 3)
    imsave(rec_img, x_r)


if __name__ == '__main__':
    X = load_data(sys.argv[1])   # (415, 1080000)
    mu = mean_face(X)
    # U, s, V = get_SVD(X, mu)    # eigenvectors: columns of V
    # compute_ratio(s)
    # np.save('five_eigenfaces.npy', V[:, :5])
    
    # plot_eigenface(10, V)

    five_eigenfaces = np.load('five_eigenfaces.npy')    # (1080000, 5)
    
    # plot_reconstruct(X, mu, five_eigenfaces)
    reconstruct(sys.argv[2], sys.argv[3], X, mu, five_eigenfaces)

