'''
The network for predicting the contact location of the robot
'''

import torch
import torch.nn as nn


class JTNNPredictor(nn.Module):
    def __init__(self, latent_size, output_size, n_hidden_layers=1, hidden_layer_size=128):
        super(JTNNPredictor, self).__init__()

        self.input_layer = nn.Linear(int(latent_size), hidden_layer_size)

        # hidden layer
        self.hidden_list = torch.nn.ModuleList()
        self.n_hidden_layers = n_hidden_layers
        for i in range(n_hidden_layers):
            self.hidden_list.append(nn.Linear(hidden_layer_size, hidden_layer_size))
        self.output_layer = nn.Linear(hidden_layer_size, output_size)
        self.activation = torch.nn.ReLU()

        self.loss_func = nn.MSELoss()


    def forward(self, tree_vec, true_loc):
        x = self.activation(self.input_layer(tree_vec))
        for i in range(self.n_hidden_layers):
            x = self.activation(self.hidden_list[i](x))

        x = self.output_layer(x)
        loss = self.loss_func(x, true_loc)

        return loss