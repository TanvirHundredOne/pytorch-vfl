# import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from params import *

# ,server_strategy='FedAvg'
# server==Model that trains on aggregate features from clients and returns embedding
# ActiveParty == that uses server and cleints to train
# PassiveParty == that provides feature and train with all avalilable data without ever seeing it


class ServerModel(nn.Module):
    def __init__(self, input_size):
        super(ServerModel, self).__init__()
        self.fc = nn.Linear(input_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc(x)
        return self.sigmoid(x)


class ServerStrategy:
    def __init__(self, train_labels, test_labels, server_config):
        self.label = torch.tensor(train_labels).float().unsqueeze(1)
        self.test_label = torch.tensor(test_labels).float().unsqueeze(1)

        self.model = ServerModel(server_config["server_model_input_size"])
        # self.server_config = server_config
        self.class_num = server_config["class_num"]
        self.client_num = server_config["client_num"]
        self.epoch_num = server_config["epoch_num"]
        self.num_rounds = server_config["round_num"]
        # self.lr = server_config['learning_rate']
        # server_config["split_list"] = [4, 4, 4]  # comment this while running
        self.split_list = server_config["split_list"]
        self.get_metrics = server_config["get_metrics"]  # boolean==>(vanilla) 1

        self.initial_parameters = [
            val.cpu().numpy() for _, val in self.model.state_dict().items()
        ]
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)
        self.criterion = nn.BCELoss()

        if server_config["batch_train"]:
            self.batch_size = server_config["batch_size"]
            # embedding and gradient data
            self.embedding_data = np.zeros(
                shape=(self.client_num, self.class_num, self.batch_size)
            )
            # For test eval
            self.test_embedding_data = np.zeros(
                shape=(self.client_num, self.class_num, len(self.Y_test))
            )

            self.batch_indexes = [0] * self.batch_size

            # w.r.t each client's embedding data
            self.embedding_grads = np.zeros(shape=(self.class_num, self.batch_size))

    def attach_clients(self, clients_list):
        self.clients = clients_list

    def server_train(self):
        pretty_print("VFL Training Starts")
        for i in range(0, self.num_rounds):
            pretty_print(f"Round>>{i}")
            self.get_client_embeddings()
            self.aggregate_embdedding()
            self.fit_server()
            self.compute_loss()
            self.backprop_server()
            self.send_embedding_gradients()
            if self.get_metrics:  # spcify this in seperate func
                self.get_model_metrics()
                pretty_print(f"Model Metrics::\n{self.metrics_aggregated}")

    def server_test(self):
        test_embedding_results = [
            torch.from_numpy(np.array(client.get_test_embedding()))
            for client in self.clients
        ]
        test_embeddings_aggregated = torch.cat(test_embedding_results, dim=1)
        test_server_embedding = test_embeddings_aggregated.detach().requires_grad_()
        preds = self.model(test_server_embedding).detach().numpy()
        correctness = 0
        for pred, label in zip(preds, self.test_label.detach().numpy()):
            if (pred > 0.5).astype(float)[0] == label[0]:
                correctness += 1
        pretty_print(f"Correct prediction {correctness} out of {preds.shape[0]} labels")

        # return output

    def get_client_embeddings(self):
        self.client_embedding_results = [
            torch.from_numpy(np.array(client.fit())) for client in self.clients
        ]

    def aggregate_embdedding(self):
        self.embeddings_aggregated = torch.cat(self.client_embedding_results, dim=1)
        self.server_embedding = self.embeddings_aggregated.detach().requires_grad_()
        # self.fit_server()
        # self.backprop_server()

    def fit_server(self):
        self.output = self.model(self.server_embedding)

    def compute_loss(self):
        self.loss = self.criterion(self.output, self.label)

    def backprop_server(self):
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()

    def send_embedding_gradients(self):
        grads = self.server_embedding.grad.split([4, 4, 4], dim=1)  # self.split_list
        np_grads = [grad.numpy() for grad in grads]
        for cid in range(0, self.client_num):
            self.clients[cid].evaluate(np_grads)

    def get_model_metrics(self):
        with torch.no_grad():
            correct = 0
            output = self.model(self.server_embedding)
            predicted = (output > 0.5).float()

            correct += (predicted == self.label).sum().item()

            accuracy = correct / len(self.label) * 100

        self.metrics_aggregated = {"accuracy": accuracy}


def testing(server):
    server.get_client_embeddings()
    server.aggregate_embdedding()
    server.fit_server()
    pretty_print(server.output)


# class ServerStrategy:  # fl.server.strategy.FedAvg
#     def __init__(
#         self,
#         labels,
#         *,
#         fraction_fit=1,
#         fraction_evaluate=1,
#         min_fit_clients=2,
#         min_evaluate_clients=2,
#         min_available_clients=2,
#         evaluate_fn=None,
#         on_fit_config_fn=None,
#         on_evaluate_config_fn=None,
#         accept_failures=True,
#         initial_parameters=None,
#         fit_metrics_aggregation_fn=None,
#         evaluate_metrics_aggregation_fn=None,
#     ) -> None:
#         self.labels = labels
#         self.fraction_fit = fraction_fit
#         self.fraction_evaluate = fraction_evaluate
#         self.min_fit_clients = min_fit_clients
#         self.min_evaluate_clients = min_evaluate_clients
#         self.min_available_clients = min_available_clients
#         self.evaluate_fn = evaluate_fn
#         self.on_fit_config_fn = on_fit_config_fn
#         self.on_evaluate_config_fn = on_evaluate_config_fn
#         self.accept_failures = accept_failures
#         self.initial_parameters = initial_parameters
#         self.fit_metrics_aggregation_fn = fit_metrics_aggregation_fn
#         self.evaluate_metrics_aggregation_fn = evaluate_metrics_aggregation_fn

#         self.model = ServerModel(12)
#         self.initial_parameters = [
#             val.cpu().numpy() for _, val in self.model.state_dict().items()
#         ]

#         self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)
#         self.criterion = nn.BCELoss()
#         self.label = torch.tensor(labels).float().unsqueeze(1)

#     def aggregate_fit(
#         self,
#         rnd,
#         params,  # ndarray of train_params from all clients>> for default example 3x (894,4)==(894,12)
#         failures,  # single boolean value; but need to pass boolean for all client
#     ):
#         # Do not aggregate if there are failures and failures are not accepted
#         if not self.accept_failures and failures:
#             return None, {}

#         # Convert results
#         # embedding_results = [
#         #     torch.from_numpy(parameters_to_ndarrays(fit_res.parameters)[0])
#         #     for _, fit_res in params
#         # ]
#         # for param in params:
#         #     print(f"in server before backwrd>>param val:{param[:2]}")
#         # print("\n")
#         embedding_results = [torch.from_numpy(np.array(param)) for param in params]
#         embeddings_aggregated = torch.cat(embedding_results, dim=1)
#         embedding_server = embeddings_aggregated.detach().requires_grad_()
#         output = self.model(embedding_server)
#         loss = self.criterion(output, self.label)
#         print(
#             f"in server before backwrd>>loss type:{type(loss)}, loss shape:{loss.shape} loss val:{loss}"
#         )
#         self.optimizer.zero_grad()
#         loss.backward()
#         self.optimizer.step()

#         grads = embedding_server.grad.split([4, 4, 4], dim=1)
#         np_grads = [grad.numpy() for grad in grads]
#         # for grad in grads:
#         #     print(f"in server after backwrd>>grad val:{grad[:2]}")
#         # print("\n")
#         # parameters_aggregated = ndarrays_to_parameters(np_grads)
#         parameters_aggregated = np_grads
#         with torch.no_grad():
#             correct = 0
#             output = self.model(embedding_server)
#             predicted = (output > 0.5).float()

#             correct += (predicted == self.label).sum().item()

#             accuracy = correct / len(self.label) * 100

#         metrics_aggregated = {"accuracy": accuracy}
#         return parameters_aggregated, metrics_aggregated

#     def aggregate_evaluate(
#         self,
#         rnd,
#         results,
#         failures,
#     ):
#         return None, {}
