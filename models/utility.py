import numpy as np
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from numpy.random import shuffle

def load_customer_data():

    file = open('../data/etl_member_churn.csv', 'rb')
    customer_data = np.loadtxt(file, delimiter=",")
    customer_data = train_test_split(customer_data, normolize=True)

    return customer_data


def train_test_split(customer_data, ratio=0.8, normolize=True):

    shuffle(customer_data)
    train_data_len = int(0.8 * len(customer_data))
    train_data_set = customer_data[:train_data_len, 1:-2]
    test_data_set = customer_data[train_data_len:, 1:-2]
    print(train_data_set.shape)

    train_label_set = customer_data[:train_data_len, -1]
    test_label_set = customer_data[train_data_len:, -1]
    print(train_label_set.shape)
    if normolize:
        norm_ = MinMaxScaler(feature_range=(0, 1))
        train_data_set = norm_.fit_transform(train_data_set)
        test_data_set = norm_.fit_transform(test_data_set)
    return train_data_set, test_data_set, train_label_set, test_label_set
