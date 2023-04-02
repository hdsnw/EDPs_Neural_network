import math
import pandas as pd
import numpy as np
import torch
import torch.optim as optim
import random
from torch.autograd import Variable
from matplotlib import pyplot as plt
import seaborn as sns
from torch import nn
from collections import OrderedDict

import torchmetrics

from Loss import loss_laplace
import nn_architectures.feed_forward_net as feed_forward_net

torch.set_default_dtype(torch.float64)

### Amostra para entrada:
### L: tamanho da amostra
L=100
a = 1
b = 2
x = np.random.uniform(0,a, size=L) ##
t = np.random.uniform(0,b, size=L)


plt.plot(x, t, 'or')
#plt.plot(dom_pred, im_pred, 'or')
plt.xlabel('x')
plt.ylabel('t')
plt.title('Conjunto de treinamento')
plt.show()

net = feed_forward_net.Net(layers=[2, 30, 30, 1],
              loss=loss_laplace.loss_laplace,
              optimizer=optim.Adamax,
              optimizer_params={'lr': 0.001})

net.fit(x=x, y=t, epochs=1000, nbatches=1, output_loss=True)