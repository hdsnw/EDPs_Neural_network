import numpy as np
from collections import OrderedDict
import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader


class Model(nn.Module):
    def __init__(self, layers: list, activation_functions: list):
        super(Model, self).__init__()

        self.features = nn.Sequential(
            nn.Linear(layers[0], layers[1]),
            activation_functions[0]()
        )

        hidden_layers = OrderedDict()
        for i in range(1, len(layers)-2):
            hidden_layers['linear'+str(i)] = nn.Linear(layers[i], layers[i+1])
            hidden_layers[f'activaction function {i}'] = activation_functions[i]()

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
    def __init__(self,
                 layers: list,
                 activation_functions: list,
                 loss,
                 optimizer,
                 optimizer_params: dict,
                 device: str):
        self.layers = layers
        self.activation_function = activation_functions
        self.loss = loss.to(device)
        self.network = Model(layers, activation_functions).to(device)
        self.optimizer = optimizer(self.network.parameters(),
                                   **optimizer_params)
        self.__device = device

    @property
    def net(self):
        """Retorna a rede

        Returns
        -------
        _type_
            Torch network
        """
        return self.network

    def fit(self, x, epochs=1000, nbatches=1, output_loss=True):
        dataloader_train = DataLoader(x, batch_size=x.size()[0]//nbatches)
        epochs_array = torch.empty(0)
        loss_array = torch.empty(0)

        for epoch in range(1, epochs+1):
            for batch in dataloader_train:
                self.optimizer.zero_grad()

                loss = self.loss(self.network, batch.to(self.__device))
                sum(loss).backward()
                self.optimizer.step()

            if epoch % 1000 == 0:
                if output_loss:
                    print(f'Epoch {epoch}, Loss {float(sum(loss))}')
                epochs_array = torch.cat((epochs_array,
                                          torch.tensor([epoch])), axis=0)
                loss_array = torch.cat((loss_array,
                                        torch.tensor([sum(loss)])), axis=0)

        return epochs_array, loss_array

    @torch.no_grad()  # Desabilita o calculo do gradiente
    def predict(self, x: torch.Tensor):
        return self.network(x.to(self.__device)).to('cpu')
