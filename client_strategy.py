# import flwr as fl
import torch
import torch.nn as nn
from params import *


class ClientModel(nn.Module):
    def __init__(self, client_input_size, client_output_size):
        super(ClientModel, self).__init__()
        self.fc = nn.Linear(client_input_size, client_output_size)

    def forward(self, x):
        return self.fc(x)


class ClientStrategy:  # fl.client.NumPyClient
    def __init__(self, cid, train_data, test_data):
        self.cid = cid
        self.train_data = torch.tensor(train_data).float()
        self.test_data = torch.tensor(test_data).float()

        self.model = ClientModel(
            client_input_size=self.train_data.shape[1],
            client_output_size=CLIENT_CONFIG["client_model_out_size"],
        )
        # print(f"\n\nself.model.parameters():::{self.model.parameters()}\n\n")
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        # self.embedding = self.model(self.train_data )

    def get_parameters(self):  # , param
        print(
            f"\nflower_client:>{self.cid} embedding:>{self.embedding.shape}\n{self.embedding.detach().numpy()[:2]}\nparams>>{self.model.parameters()}\n"
        )
        # pass

    def fit(self):  # , parameters, config
        self.embedding = self.model(self.train_data)
        return self.embedding.detach().numpy()

    def evaluate(self, parameters):  # , config
        self.model.zero_grad()
        self.embedding.backward(torch.from_numpy(parameters[int(self.cid)]))
        self.optimizer.step()
        # self.get_parameters(None)

    def get_test_embedding(self):
        self.embedding = self.model(self.test_data)
        return self.embedding.detach().numpy()
