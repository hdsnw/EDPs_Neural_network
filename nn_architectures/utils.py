from matplotlib import pyplot as plt
import torch
import torchmetrics


def metricas(net, sol_eq, a, b):
    """Calcula a o erro médio quadrado e o erro médio
    absoluto de uma rede neural.

    Parameters
    ----------
    net : _type_
        Rede neural
    sol_eq : _type_
        Função da equação analítica
    a : int
        _description_
    b : int
        _description_

    Returns
    -------
    Tuple
        Erro médio quadrado e erro médio absoluto
    """
    xx = torch.linspace(0, a, 100)
    yy = torch.linspace(0, b, 100)

    xx, yy = torch.meshgrid(xx, yy, indexing='xy')

    xx = xx.reshape((xx.shape[0]**2, 1))
    yy = yy.reshape((yy.shape[0]**2, 1))

    sol = sol_eq(xx, yy)
    prev = net.predict(torch.cat([xx, yy], axis=1))

    return torchmetrics.functional.mean_squared_error(sol, prev),\
           torchmetrics.functional.mean_absolute_error(sol, prev)


def plot_solucao_erro_l2(net, sol_eq, a, b):
    xx = torch.linspace(0, a, 100)
    yy = torch.linspace(0, b, 100)

    xx, yy = torch.meshgrid(xx, yy, indexing='xy')

    xx = xx.reshape((xx.shape[0]**2, 1))
    yy = yy.reshape((yy.shape[0]**2, 1))

    sol = sol_eq(xx, yy)
    prev = net.predict(torch.cat([xx, yy], axis=1))

    prev = prev.reshape((100, 100))
    sol = sol.reshape((100, 100))
    xx = xx.reshape((100, 100))
    yy = yy.reshape((100, 100))

    #plot
    fig, ax = plt.subplots(1, 2, figsize=(20,7))

    img1 = ax[0].pcolor(xx.numpy(), yy.numpy(), sol.numpy())
    fig.colorbar(img1, ax=ax[0])
    ax[0].set_title('Solução Analítica')
    ax[0].set_xlabel('x')
    ax[0].set_ylabel('y')

    img2 = ax[1].pcolor(xx.numpy(), yy.numpy(), prev.numpy())
    fig.colorbar(img2, ax=ax[1])

    ax[1].set_title('Solução numérica obtida pela rede neural1')
    ax[1].set_xlabel('x')
    ax[1].set_ylabel('y')

    # Plot erro L2
    fig, ax = plt.subplots(1, 1, figsize=(7,5))

    img1 = ax.pcolor(xx.numpy(), yy.numpy(), abs(prev.numpy() - sol.numpy()))
    fig.colorbar(img1, ax=ax)
    ax.set_title('Erro L2')
    ax.set_xlabel('x')
    ax.set_ylabel('y')

def dy(model, x, y, h=0.001):
    # d = torch.autograd.grad(model(x, t), (x, t), grad_outputs=torch.ones_like(
    #    model(x, t)), create_graph=True, retain_graph=True)
    # return d[1]
    inp = torch.cat([x.unsqueeze(0), y.unsqueeze(0)]).T
    y = y+h
    inp_h = torch.cat([x.unsqueeze(0), y.unsqueeze(0)]).T
    return (model(inp_h) - model(inp))/h


def dyy(model, x, y, h=0.001):
    y_plus_h = torch.cat([x.unsqueeze(0), (y + h).unsqueeze(0)]).T
    y_minus_h = torch.cat([x.unsqueeze(0), (y - h).unsqueeze(0)]).T
    inp = torch.cat([x.unsqueeze(0), y.unsqueeze(0)]).T

    return (model(y_plus_h) - 2*model(inp) + model(y_minus_h))/h**2


def dx(model, x, y, h=0.001):
    inp = torch.cat([x.unsqueeze(0), y.unsqueeze(0)]).T
    x = x+h
    inp_h = torch.cat([x.unsqueeze(0), y.unsqueeze(0)]).T

    return (model(inp_h) - model(inp))/h


def dxx(model, x, y, h=0.001):
    x_plus_h = x + h
    x_1 = torch.cat([x_plus_h.unsqueeze(0), y.unsqueeze(0)]).T
    x_minus_h = x - h
    x_3 = torch.cat([x_minus_h.unsqueeze(0), y.unsqueeze(0)]).T
    x_2 = torch.cat([x.unsqueeze(0), y.unsqueeze(0)]).T

    return (model(x_1) - 2*model(x_2) + model(x_3))/h**2
