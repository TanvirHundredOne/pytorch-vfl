from params import *


def split_data(partitions, label, test_size=10):
    train_partitions = []
    test_partitions = []
    # train_label = []
    # test_label = []
    for cid in range(SERVER_CONFIG["client_num"]):
        train_partitions.append(partitions[cid][:-test_size])
        test_partitions.append(partitions[cid][-test_size:])

    train_label = label[:-test_size]
    test_label = label[-test_size:]
    return train_partitions, train_label, test_partitions, test_label
