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


class DGMCELL(nn.Module):
    def __init__(self, n_neurons, activation_function):
        super(DGMCELL, self).__init__()

        self.u_g = nn.Parameter(torch.Tensor(2, n_neurons))
        self.u_z = nn.Parameter(torch.Tensor(2, n_neurons))
        self.u_r = nn.Parameter(torch.Tensor(2, n_neurons))
        self.u_h = nn.Parameter(torch.Tensor(2, n_neurons))

        self.w_z = nn.Parameter(torch.Tensor(n_neurons, n_neurons))
        self.w_g = nn.Parameter(torch.Tensor(n_neurons, n_neurons))
        self.w_r = nn.Parameter(torch.Tensor(n_neurons, n_neurons))
        self.w_h = nn.Parameter(torch.Tensor(n_neurons, n_neurons))

        self.b_z = nn.Parameter(torch.Tensor(1, n_neurons))
        self.b_g = nn.Parameter(torch.Tensor(1, n_neurons))
        self.b_r = nn.Parameter(torch.Tensor(1, n_neurons))
        self.b_h = nn.Parameter(torch.Tensor(1, n_neurons))

        nn.init.xavier_uniform_(self.u_z)
        nn.init.xavier_uniform_(self.u_g)
        nn.init.xavier_uniform_(self.u_r)
        nn.init.xavier_uniform_(self.u_h)

        nn.init.xavier_uniform_(self.w_z)
        nn.init.xavier_uniform_(self.w_g)
        nn.init.xavier_uniform_(self.w_r)
        nn.init.xavier_uniform_(self.w_h)

        nn.init.zeros_(self.b_z)
        nn.init.zeros_(self.b_g)
        nn.init.zeros_(self.b_r)
        nn.init.zeros_(self.b_h)

        self.activation_function = activation_function()

    def forward(self, s, x):
        z = self.activation_function(torch.matmul(x, self.u_z) + torch.matmul(s, self.w_z,) + self.b_z)
        g = self.activation_function(torch.matmul(x, self.u_g) + torch.matmul(s, self.w_g,) + self.b_g)
        r = self.activation_function(torch.matmul(x, self.u_r) + torch.matmul(s, self.w_r,) + self.b_r)
        h = self.activation_function(torch.matmul(x, self.u_h) + torch.matmul(s*r, self.w_h) + self.b_h)

        s_novo = (1-g)*h + z*s
        return s_novo


class DGM(nn.Module):
    def __init__(self, n_input, n_neurons, n_layers, activation_function):
        super(DGM, self).__init__()

        self.features = nn.Sequential(
            nn.Linear(n_input, n_neurons),
            activation_function()
        )

        self.n_layers = n_layers
        self.dgm_layer_list = nn.ModuleList()

        for _ in range(self.n_layers):
            self.dgm_layer_list.append(DGMCELL(n_neurons, activation_function))

        self.out = nn.Sequential(
            nn.Linear(n_neurons, 1),
            # nn.Tanh()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = self.features(x)
        for i in range(self.n_layers):
            s = self.dgm_layer_list[i](s, x)

        output = self.out(s)

        return output


class Net_DGM():
    def __init__(self, n_input, n_neurons, n_layers, activation_functions, loss, loss_params: dict, optimizer, optimizer_params: dict):
        self.activation_function = activation_functions
        self.loss = loss
        self.loss_params = loss_params
        self.network = DGM(n_input, n_neurons, n_layers, activation_functions)
        # self.network = DGMNet(layer_width=n_neurons, n_layers=5, input_dim=n_input)
        self.optimizer = optimizer(
            self.network.parameters(), **optimizer_params)

    @property
    def net(self):
        return self.network

    def fit(self, x, epochs=1000, nbatches=1, output_loss=True):
        epochs_array = []
        loss_array = []
        index = torch.randperm(x.shape[0])

        batchs = torch.split(index, int(len(index)/nbatches))

        for epoch in range(1, epochs+1):
            for batch in batchs:
                self.optimizer.zero_grad()

                loss = self.loss(self.network, x[batch], **self.loss_params)
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
