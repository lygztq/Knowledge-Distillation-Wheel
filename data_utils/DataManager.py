import numpy as np
import os
from data.cifar10.cifar10_manager import load_CIFAR10
from data.cifar100.cifar100_manager import load_CIFAR100
from data.MNIST.mnist_manager import load_MNIST

class DataManager(object):
    """
    The class that manager data.
    
    :param dataset_name:        The name of dataset, candidates are 'CIFAR-10', 'CIFAR-100'. TODO: 'MNIST', 'IMAGE_NET'
    :param with_image_shape:    Whether keep the shape of images.
    :param one_hot:             Whether use one hot representation
    :param val_ratio:           The ratio of validation set
    :param dev_ratio:           The ratio of development set
    """
    def __init__(self, dataset_name, path='./data/', val_ratio=0.1, dev_ratio=0.01, with_image_shape=True, one_hot=True):
        f = { # Decoding function of different datasets.
            'CIFAR-10': load_CIFAR10,
            'CIFAR-100': load_CIFAR100,
            'MNIST': load_MNIST
        }

        p = { # Detailed path of dataset.
            'CIFAR-10': 'cifar10/cifar-10-batches-py',
            'CIFAR-100': 'cifar100/cifar-100-python',
            'MNIST': 'MNIST/'
        }

        self.train_data, self.train_label, self.test_data, self.test_label = f[dataset_name](
            path=os.path.join(path, p[dataset_name]), with_image_shape=with_image_shape, one_hot=one_hot
        )
        self._transformDataType()

        self._num_train_data = self.train_data.shape[0]
        self.val_ratio = val_ratio
        self.dev_ratio = dev_ratio
        self._trainValidationSplit()
        #self._preprocess()
        print("From the DataManager: successfully loaded {} dataset with training data amount {}".format(dataset_name, self._num_train_data))
    
    def _transformDataType(self):
        self.train_data = self.train_data.astype(np.float32)
        self.train_label = self.train_label.astype(np.float32)
        self.test_data = self.test_data.astype(np.float32)
        self.test_label = self.test_label.astype(np.float32)

    def _trainValidationSplit(self):
        """ Split the training dataset to get the training set and validation set """
        idx = np.arange(self._num_train_data)
        np.random.shuffle(idx)
        self.num_val_set = int(self._num_train_data * self.val_ratio)
        self.num_train_set = self._num_train_data - self.num_val_set
        self._num_dev_set = int(self._num_train_data * self.dev_ratio)

        mask = idx[:self.num_val_set]
        self.val_data = self.train_data[mask]
        self.val_label = self.train_label[mask]

        mask = idx[:self._num_dev_set]
        self.dev_data = self.train_data[mask]
        self.dev_label = self.train_label[mask]

        mask = idx[self.num_val_set:]
        self.train_data = self.train_data[mask]
        self.train_label = self.train_label[mask]

    def _preprocess(self):
        self.data_means = np.mean(self.train_data, axis=0)
        self.data_std = np.std(self.train_data, axis=0)
        self.train_data = (self.train_data - self.data_means) / self.data_std
        self.val_data = (self.val_data - self.data_means) / self.data_std
        self.test_data = (self.test_data - self.data_means) / self.data_std


if __name__ == "__main__":
    d = DataManager('CIFAR-100')
    print(d.train_data.shape, d.train_label.shape)
    

        