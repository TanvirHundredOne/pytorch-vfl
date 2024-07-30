# import flwr as fl
import numpy as np

from server_strategy import ServerStrategy
from client_strategy import ClientStrategy
from pathlib import Path
from data_process import get_partitions_and_label, data_transform
from params import *
from utils import *

pretty_print("Starting simulation")
partitions, label = get_partitions_and_label()
partitions, label, test_partitions, test_label = split_data(partitions, label)
# partitions = data_transform(partitions)
# test_partitions = data_transform(test_partitions)
# --------------------------------------------------------------------------------------------------

pretty_print(f"data shape: {partitions[0].shape}")
pretty_print(f"Label shape: {len(label)}")
# create composite configs
create_composite_cofigs()
pretty_print(SERVER_CONFIG)
pretty_print(CLIENT_CONFIG)
# create server and clients
vfl_server = ServerStrategy(label)
vfl_clients = [
    ClientStrategy(cid, data_transform(partitions[cid]))
    for cid in range(0, SERVER_CONFIG["client_num"])
]

vfl_server.attach_clients(vfl_clients)
vfl_server.server_train()
vfl_server.server_test(
    [
        data_transform(test_partitions[cid])
        for cid in range(0, SERVER_CONFIG["client_num"])
    ],
    test_label,
)
