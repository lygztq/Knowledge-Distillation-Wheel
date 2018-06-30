from data_utils.DataManager import DataManager


def data_manager_test():
    d = DataManager('CIFAR-10', path='./data')
    print(d.train_data.shape, d.train_label.shape)