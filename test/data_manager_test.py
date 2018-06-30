from data_utils.DataManager import DataManager


if __name__ == "__main__":
    d = DataManager('CIFAR-100', path='../data')
    print(d.train_data.shape, d.train_label.shape)