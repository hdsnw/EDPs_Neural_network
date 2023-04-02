import tempfile
import numpy as np
import torch


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


def f(x):
    return x**2


def f2(x):
    return (x-2)**2


def g(y):
    return y**2


def g2(y):
    return (y-1)**2


def loss_laplace(model, inp: torch.tensor):
    x = inp[:, 0].unsqueeze(0).T
    y = inp[:, 1].unsqueeze(0).T
    # Perda domínio
    loss_dom = (dxx(model, x, y) + dyy(model, x, y) - 4)**2
    L1 = loss_dom.sum()*0.001**2

    # Perda contorno y
    zero = torch.zeros(y.size()[0], 1, requires_grad=True)
    x_zero = torch.cat([x, zero], axis=1)
    loss_y0 = (model(x_zero) - x.detach().clone().apply_(f))**2
    L2 = loss_y0.sum()*0.001**2

    one = torch.ones(y.size()[0], 1, requires_grad=True)
    x_two = x_zero = torch.cat([x, one*2], axis=1)
    loss_yb = (model(x_two) - x.detach().clone().apply_(f2))**2
    L3 = loss_yb.sum()*0.001**2

    # Perda contorno x
    zero = torch.zeros(x.size()[0], 1, requires_grad=True)
    y_zero = torch.cat([zero, y], axis=1)
    loss_x0 = (model(y_zero) - y.detach().clone().apply_(g))**2
    L4 = loss_x0.sum()*0.001**2

    one = torch.ones(x.size()[0], 1, requires_grad=True)
    y_one = torch.cat([one, y], axis=1)
    loss_xa = (model(y_one) - y.detach().clone().apply_(g2))**2
    L5 = loss_xa.sum()*0.001**2

    return L1, L2, L3, L4, L5


def u(x, t):
    """Solução analítica da equação de laplace

    Parameters
    ----------
    x : torch.Tensor
        _description_
    t : torch.Tensor
        _description_

    Returns
    -------
    torch.Tensor
        _description_
    """
    sol = (x-t)**2
    return sol
