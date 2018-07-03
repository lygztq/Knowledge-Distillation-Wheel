from six.moves import cPickle as pickle
import numpy as np
import os
from scipy.misc import imread
#import matplotlib.pyplot as plt
import platform

def load_pickle(f):
    version = platform.python_version_tuple()
    if version[0] == '2':
        return  pickle.load(f)
    elif version[0] == '3':
        return  pickle.load(f, encoding='latin1')
    raise ValueError("invalid python version: {}".format(version))

def load_CIFAR100_batch(filename, with_image_shape, one_hot):
  """ load single batch of cifar """
  with open(filename, 'rb') as f:
    datadict = load_pickle(f)
    # print(datadict.keys())
    X = datadict['data']
    Y = datadict['fine_labels']
    X = X.reshape(-1, 3, 32, 32).transpose(0,2,3,1).astype("float")
    if not with_image_shape:
        X = X.reshape(-1, 3*32*32)
    Y = np.array(Y)
    if one_hot:
        vectors = np.eye(100)
        Y = vectors[Y]
    return X, Y

def load_CIFAR100(path='./data/cifar100/cifar-100-python', with_image_shape=False, one_hot=True):
  """ load all of cifar """
  Xtr, Ytr = load_CIFAR100_batch(os.path.join(path, 'train'), with_image_shape, one_hot)
  Xte, Yte = load_CIFAR100_batch(os.path.join(path, 'test'), with_image_shape, one_hot)
  return Xtr, Ytr, Xte, Yte
