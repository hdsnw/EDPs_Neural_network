import torch
import torch.optim as optim
from torch import nn
from collections import OrderedDict
import numpy as np


class Model(nn.Module):
    def __init__(self, layers: list, activation_function: str = 'tanh'):
        super(Model, self).__init__()

        self.features = nn.Sequential(
            nn.Linear(layers[0], layers[1]),
            nn.Tanh()
        )

        hidden_layers = OrderedDict()
        for i in range(1, len(layers)-2):
            hidden_layers['linear'+str(i)] = nn.Linear(layers[i], layers[i+1])
            hidden_layers[activation_function+str(i)] = nn.Tanh()

        self.hidden = nn.Sequential(hidden_layers)

        self.out = nn.Sequential(
            nn.Linear(layers[-2], layers[-1]),
            # nn.Tanh()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.features(x)
        output = self.hidden(output)
        output = self.out(output)

        return output


class Net():
    def __init__(self, layers, activation_function, loss, optimizer, optimizer_params: dict):
        self.layers = layers
        self.activation_function = activation_function
        self.loss = loss
        self.network = Model(layers)
        self.optimizer = optimizer(
            self.network.parameters(), **optimizer_params)

    def fit(self, x, epochs=1000, nbatches=1, output_loss=True):
        epochs_array = []
        loss_array = []
        index = torch.randperm(x.shape[0])

        batchs = torch.split(index, int(len(index)/nbatches))

        for epoch in range(1, epochs+1):
            for batch in batchs:
                self.optimizer.zero_grad()

                loss = self.loss(self.network, x[batch])
                sum(loss).backward()
                self.optimizer.step()

            if epoch % 1000 == 0:
                if output_loss:
                    print(f'Epoch {epoch}, Loss {float(sum(loss))}')
                epochs_array.append(epoch)
                loss_array.append([i.detach().numpy() for i in loss])

        return epochs_array, loss_array

    @torch.no_grad()  # Desabilita o calculo do gradiente
    def predict(self, x: torch.Tensor):
        return self.network(x)
