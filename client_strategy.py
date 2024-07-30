# import flwr as fl
import torch
from sklearn.preprocessing import StandardScaler
import torch.nn as nn
from params import *


class ClientModel(nn.Module):
    def __init__(self, client_input_size, client_output_size):
        super(ClientModel, self).__init__()
        self.fc = nn.Linear(client_input_size, client_output_size)

    def forward(self, x):
        return self.fc(x)


class ClientStrategy:  # fl.client.NumPyClient
    def __init__(self, cid, train_data):
        self.cid = cid
        # print(data.shape)
        self.train_data = torch.tensor(
            train_data
        ).float()  # StandardScaler().fit_transform(data)
        self.model = ClientModel(
            client_input_size=self.train_data.shape[1],
            client_output_size=CLIENT_CONFIG["client_model_out_size"],
        )
        # print(f"\n\nself.model.parameters():::{self.model.parameters()}\n\n")
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        self.batch_train = CLIENT_CONFIG["batch_train"]
        # self.embedding = self.model(self.train_data)

    # def get_parameters(self):  # , param
    #     print(
    #         f"\nflower_client:>{self.cid} embedding:>{self.embedding.shape}\n{self.embedding.detach().numpy()[:2]}\nparams>>{self.model.parameters()}\n"
    #     )
    #     # pass
    def set_batch_indexes(self, batch_index):
        self.batch_index = batch_index

    def fit(self):  # , parameters, config
        # if self.batch_train:
        training_data = torch.tensor(
            np.array(self.train_data)[self.batch_index.astype(int)]
        )
        # else:
        #     training_data = self.train_data
        self.embedding = self.model(training_data)
        return self.embedding.detach().numpy()

    def backprop(self, parameters):  # , config
        self.model.zero_grad()
        self.embedding.backward(torch.from_numpy(parameters[int(self.cid)]))
        self.optimizer.step()
        # self.get_parameters(None)

    def fit_test(self, data):  # , parameters, config
        self.embedding = self.model(torch.tensor(data).float())
        return self.embedding.detach().numpy()
