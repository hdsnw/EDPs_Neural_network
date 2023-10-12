import numpy as np
import torch
from torch import nn


class Derivatives():
    def dx(self, func, x, y, h=0.001):
        inp = torch.cat([x, y], axis=1)
        inp_h = torch.cat([x + h, y], axis=1)
        return (func(inp_h) - func(inp))/h

    def dy(self, func, x, y, h=0.001):
        inp = torch.cat([x, y], axis=1)
        inp_h = torch.cat([x, y + h], axis=1)
        return (func(inp_h) - func(inp))/h

    def dxx(self, func, x, y, h=0.0001):
        x_plus_h = torch.cat([x + h, y], axis=1)
        x_minus_h = torch.cat([x - h, y], axis=1)
        x_y = torch.cat([x, y], axis=1)
        return (func(x_plus_h) - 2*func(x_y) + func(x_minus_h))/h**2


class LossLaplace(nn.Module):
    def __init__(self, taxa_aceleracao=1):
        super(LossLaplace, self).__init__()
        self.__taxa_aceleracao = taxa_aceleracao

    def __k(self, x):
        # if 0 <= y and y <= 1:
        #   return y
        # elif 1 <= y and y <= 2:
        #   return 2-y
        return torch.where(x <= 1.0, x, 2 - x)

    def __dxx(self, func, x, y, h=0.0001):
        x_plus_h = torch.cat([x + h, y], axis=1)
        x_minus_h = torch.cat([x - h, y], axis=1)
        x_y = torch.cat([x, y], axis=1)

        return (func(x_plus_h) - 2*func(x_y) + func(x_minus_h))/h**2

    def __dyy(self, func, x, y, h=0.0001):
        y_plus_h = torch.cat([x, (y + h)], axis=1)
        y_minus_h = torch.cat([x, (y - h)], axis=1)
        x_y = torch.cat([x, y], axis=1)

        return (func(y_plus_h) - 2*func(x_y) + func(y_minus_h))/h**2

    def forward(self, net, inputs: torch.tensor):
        # Realiza as operações no mesmo dispositivo da rede (cpu ou cuda)
        # with torch.device(inputs.device):
        x = inputs[:, 0].unsqueeze(0).T
        y = inputs[:, 1].unsqueeze(0).T

        # Perda domínio
        loss_dom = (self.__dxx(net, x, y) + self.__dyy(net, x, y))**2
        # loss_dom = (v_xx_yy.sum(axis=1))**2
        L1 = loss_dom.sum()*self.__taxa_aceleracao

        # Perda contorno y
        zero = torch.zeros(y.size()[0], 1, requires_grad=True).to(inputs.device)
        x_zero = torch.cat([x, zero], axis=1)
        loss_y0 = (net(x_zero))**2
        L2 = loss_y0.sum()*self.__taxa_aceleracao

        one = torch.ones(y.size()[0], 1, requires_grad=True).to(inputs.device)
        x_two = x_zero = torch.cat([x, one*2], axis=1)
        loss_yb = (net(x_two))**2
        L3 = loss_yb.sum()*self.__taxa_aceleracao

        # Perda contorno x
        zero = torch.zeros(x.size()[0], 1, requires_grad=True).to(inputs.device)
        y_zero = torch.cat([zero, y], axis=1)
        loss_x0 = (net(y_zero))**2
        L4 = loss_x0.sum()*self.__taxa_aceleracao

        one = torch.ones(x.size()[0], 1, requires_grad=True).to(inputs.device)
        y_three = torch.cat([one*3, y], axis=1)
        loss_xa = (net(y_three) - self.__k(y))**2
        L5 = loss_xa.sum()*self.__taxa_aceleracao
        return L1, L2, L3, L4, L5


class LossCalor(nn.Module):
    def __init__(self, taxa_aceleracao=1):
        super(LossCalor, self).__init__()
        self.__taxa_aceleracao = taxa_aceleracao

    def __f(self, x):
#        if 0 <= x and x < 20:
#            return x
#        elif 20 <= x and x <= 40:
#            return 40-x
        return torch.where(x < 20, x, 40 - x)

    def __dy(self, func, x, y, h=0.001):
        inp = torch.cat([x, y], axis=1)
        inp_h = torch.cat([x, y + h], axis=1)
        return (func(inp_h) - func(inp))/h

    def __dxx(self, func, x, y, h=0.0001):
        x_plus_h = torch.cat([x + h, y], axis=1)
        x_minus_h = torch.cat([x - h, y], axis=1)
        x_y = torch.cat([x, y], axis=1)
        return (func(x_plus_h) - 2*func(x_y) + func(x_minus_h))/h**2

    def __dyy(self, func, x, y, h=0.0001):
        y_plus_h = torch.cat([x, (y + h)], axis=1)
        y_minus_h = torch.cat([x, (y - h)], axis=1)
        x_y = torch.cat([x, y], axis=1)
        return (func(y_plus_h) - 2*func(x_y) + func(y_minus_h))/h**2

    def forward(self, net, inputs: torch.tensor):
        # Realiza as operações no mesmo dispositivo da rede (cpu ou cuda)
        # with torch.device(inputs.device):
        x = inputs[:, 0].unsqueeze(0).T
        y = inputs[:, 1].unsqueeze(0).T
        # Perda domínio
        loss_dom = (self.__dy(net, x, y) - self.__dxx(net, x, y))**2
        L1 = loss_dom.sum()*self.__taxa_aceleracao

        # Perda contorno y
        zero = torch.zeros(y.size()[0], 1, requires_grad=True).to(inputs.device)
        x_zero = torch.cat([x, zero], axis=1)
        loss_y0 = (net(x_zero) - self.__f(x))**2
        L2 = loss_y0.sum()*self.__taxa_aceleracao

        # Perda contorno x
        zero = torch.zeros(x.size()[0], 1, requires_grad=True).to(inputs.device)
        y_zero = torch.cat([zero, y], axis=1)
        loss_x0 = (net(y_zero))**2
        L3 = loss_x0.sum()*self.__taxa_aceleracao

        one = torch.ones(x.size()[0], 1, requires_grad=True).to(inputs.device)
        y_one = torch.cat([one*40, y], axis=1)
        loss_xa = (net(y_one))**2
        L4 = loss_xa.sum()*self.__taxa_aceleracao

        return L1, L2, L3, L4


class LossOnda(nn.Module):
    def __init__(self, taxa_aceleracao=1):
        super(LossOnda, self).__init__()
        self.__taxa_aceleracao = taxa_aceleracao

    def __dxx(self, func, x, y, h=0.0001):
        x_plus_h = torch.cat([x + h, y], axis=1)
        x_minus_h = torch.cat([x - h, y], axis=1)
        x_y = torch.cat([x, y], axis=1)
        return (func(x_plus_h) - 2*func(x_y) + func(x_minus_h))/h**2

    def __dyy(self, func, x, y, h=0.0001):
        y_plus_h = torch.cat([x, (y + h)], axis=1)
        y_minus_h = torch.cat([x, (y - h)], axis=1)
        x_y = torch.cat([x, y], axis=1)
        return (func(y_plus_h) - 2*func(x_y) + func(y_minus_h))/h**2

    def forward(self, net, inputs: torch.tensor):
        # Realiza as operações no mesmo dispositivo da rede (cpu ou cuda)
        # with torch.device(inputs.device):
        x = inputs[:, 0].unsqueeze(0).T
        y = inputs[:, 1].unsqueeze(0).T

        # Perda domínio
        loss_dom = (self.__dxx(net, x, y) - self.__dyy(net, x, y))**2
        L1 = loss_dom.sum()*self.__taxa_aceleracao

        # Perda contorno y
        zero = torch.zeros(y.size()[0], 1, requires_grad=True).to(inputs.device)
        x_zero = torch.cat([x, zero], axis=1)
        loss_y0 = (net(x_zero) - 2*torch.sin(x*torch.pi))**2
        L2 = loss_y0.sum()*self.__taxa_aceleracao

        one = torch.ones(y.size()[0], 1, requires_grad=True).to(inputs.device)
        x_one = x_zero = torch.cat([x, one], axis=1)
        loss_yb = (net(x_one) + torch.sin(2*x*torch.pi))**2
        L3 = loss_yb.sum()*self.__taxa_aceleracao

        # Perda contorno x
        zero = torch.zeros(x.size()[0], 1, requires_grad=True).to(inputs.device)
        y_zero = torch.cat([zero, y], axis=1)
        loss_x0 = (net(y_zero))**2
        L4 = loss_x0.sum()*self.__taxa_aceleracao

        one = torch.ones(x.size()[0], 1, requires_grad=True).to(inputs.device)
        y_one = torch.cat([one, y], axis=1)
        loss_xa = (net(y_one))**2
        L5 = loss_xa.sum()*self.__taxa_aceleracao

        return L1, L2, L3, L4, L5


class LossOnda2(nn.Module):
    def __init__(self, taxa_aceleracao=1):
        super(LossOnda2, self).__init__()
        self.__taxa_aceleracao = taxa_aceleracao

    def __dy(self, func, x, y, h=0.001):
        inp = torch.cat([x, y], axis=1)
        inp_h = torch.cat([x, y + h], axis=1)
        return (func(inp_h) - func(inp))/h

    def __dxx(self, func, x, y, h=0.0001):
        x_plus_h = torch.cat([x + h, y], axis=1)
        x_minus_h = torch.cat([x - h, y], axis=1)
        x_y = torch.cat([x, y], axis=1)
        return (func(x_plus_h) - 2*func(x_y) + func(x_minus_h))/h**2

    def __dyy(self, func, x, y, h=0.0001):
        y_plus_h = torch.cat([x, (y + h)], axis=1)
        y_minus_h = torch.cat([x, (y - h)], axis=1)
        x_y = torch.cat([x, y], axis=1)
        return (func(y_plus_h) - 2*func(x_y) + func(y_minus_h))/h**2

    def forward(self, net, inputs: torch.tensor):
        # Realiza as operações no mesmo dispositivo da rede (cpu ou cuda)
        # with torch.device(inputs.device):
        x = inputs[:, 0].unsqueeze(0).T
        y = inputs[:, 1].unsqueeze(0).T

        # Perda domínio
        loss_dom = (self.__dyy(net, x, y) - self.__dxx(net, x, y))**2
        L1 = loss_dom.sum()*self.__taxa_aceleracao

        # Perda contorno y
        zero = torch.zeros(y.size()[0], 1, requires_grad=True).to(inputs.device)
        x_zero = torch.cat([x, zero], axis=1)
        loss_y0 = (torch.sin(x*torch.pi) - net(x_zero))**2
        L2 = loss_y0.sum()*self.__taxa_aceleracao

        loss_yb = (self.__dy(net, x, zero))**2
        L3 = loss_yb.sum()*self.__taxa_aceleracao

        # Perda contorno x
        zero = torch.zeros(x.size()[0], 1, requires_grad=True).to(inputs.device)
        y_zero = torch.cat([zero, y], axis=1)
        loss_x0 = (net(y_zero))**2
        L4 = loss_x0.sum()*self.__taxa_aceleracao

        one = torch.ones(x.size()[0], 1, requires_grad=True).to(inputs.device)
        y_one = torch.cat([one, y], axis=1)
        loss_xa = (net(y_one))**2
        L5 = loss_xa.sum()*self.__taxa_aceleracao

        return L1, L2, L3, L4, L5


def solution_laplace(x, t, n=10):
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
    sol = 0
    for i in range(0, n):
        sol += ((-1)**i / ((2*i + 1)**2 * (np.sinh(3*(2*i+1)*np.pi/2))))*np.sin((2*i+1)*np.pi*t/2)*np.sinh((2*i+1)*np.pi*x/2)
    sol = sol*(8/np.pi**2)
    return sol


def solucao_equacao_calor(x, t, n=50):
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
    sol = 0
    for i in range(0, n):
        f1 = (-1)**i/((2*i + 1)**2)
        f2 = np.sin((2*i+1)*np.pi*x/(40))
        f3 = np.exp(-((2*i+1)**2)*(np.pi)**2*t/1600)
        sol += f1*f2*f3
    sol = sol*(160/np.pi**2)
    return sol


def solution_onda_2(x, t, n=20):
    return np.cos(np.pi*t)*np.sin(np.pi*x)
