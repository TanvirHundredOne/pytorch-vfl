import numpy as np
import pandas as pd

## Configuration
NUM_ROUNDS = 3
NUM_CLIENTS = 3

SERVER_CONFIG = {
    "class_num": 2,
    "client_num": 3,
    "epoch_num": 5,
    "round_num": 300,
    # "split_list": [4, 4, 4], Composite
    "get_metrics": True,
    "batch_train": True,
    "batch_size": 100,
}
CLIENT_CONFIG = {"client_model_out_size": 4, "client_weight": "same"}


def create_composite_cofigs():
    # create embedding split_strategy
    embedding_split_strategy()
    create_server_model_input_size()
    if SERVER_CONFIG["batch_train"]:
        CLIENT_CONFIG["batch_train"] = True
    else:
        CLIENT_CONFIG["batch_train"] = False


def embedding_split_strategy():
    split_list = []
    if CLIENT_CONFIG["client_weight"] == "same":
        for _ in range(0, SERVER_CONFIG["client_num"]):
            split_list.append(CLIENT_CONFIG["client_model_out_size"])
    SERVER_CONFIG["split_list"] = split_list


def create_server_model_input_size():
    size = 0
    for split in SERVER_CONFIG["split_list"]:
        size += split
    SERVER_CONFIG["server_model_input_size"] = size


def pretty_print(msg):
    print("\n>>>>:::")
    print(f"{msg}")
    print(":::>>>\n")
