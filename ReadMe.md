# Vertical Federated Learning Pytorch Neural Network Implementation
Basic vertical federated learning training and testing using FedAvg server strategy. I have used <a href='https://www.kaggle.com/competitions/titanic/data'>Titanic Dataset</a> for this.

<h2>Simple Overview </h2>
<h3>Trainig</h3>

- 1 Server only has training labels, no features and NN model as training model, it concatenates embedding from all clients and trains server_model, cuts the embedding gradients in same method and sends back the seperated embedding gradients to specific clients, computes accuracy on labels. 

- N Clients, each having its own set of trainnig features, forward_pass own client model, share embedding to server, recieves embedding gradients on which it runs backprop.

<h3>Testing</h3>

- Coming up...






# Acknowledgments

- Inspired by <a href='https://github.com/FLAIR-THU/VFLAIR'>VFLAIR</a> and <a href='https://github.com/adap/flower'>Flower Repo</a>. 

- <a href='http://arxiv.org/pdf/2211.12814'>Vertical Federate Learning Cocepts</a>.
