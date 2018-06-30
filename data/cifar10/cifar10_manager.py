from six.moves import cPickle as pickle
import numpy as np
import os
from scipy.misc import imread
import matplotlib.pyplot as plt
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
    X = datadict['data']
    Y = datadict['labels']
    X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("float")
    if not with_image_shape:
        X = X.reshape(10000, -1)
    Y = np.array(Y)
    if one_hot:
        vectors = np.eye(10)
        Y = vectors[Y]
    return X, Y

def load_CIFAR10(path='./data/cifar10/cifar-10-batches-py', with_image_shape=False, one_hot=True):
  """ load all of cifar """
  xs = []
  ys = []
  for b in range(1,6):
    f = os.path.join(path, 'data_batch_%d' % (b, ))
    X, Y = load_CIFAR100_batch(f, with_image_shape, one_hot)
    xs.append(X)
    ys.append(Y)    
  Xtr = np.concatenate(xs)
  Ytr = np.concatenate(ys)
  del X, Y
  Xte, Yte = load_CIFAR100_batch(os.path.join(path, 'test_batch'), with_image_shape, one_hot)
  return Xtr, Ytr, Xte, Yte



# test
# if __name__ == "__main__":
#     train_data, train_label, test_data, test_label = load_CIFAR10(path='./cifar-10-batches-py', with_image_shape=True, one_hot=False)
#     print(train_data.shape, train_label.shape)
#     print(test_data.shape, test_label.shape)
#     print(train_label[100])
#     plt.imshow(train_data[100])
#     plt.show()
