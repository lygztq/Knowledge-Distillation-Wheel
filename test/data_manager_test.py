from data_utils.DataManager import DataManager


def data_manager_test():
    d = DataManager('MNIST', path='./data')
    print(d.train_data.shape, d.train_label.shape)