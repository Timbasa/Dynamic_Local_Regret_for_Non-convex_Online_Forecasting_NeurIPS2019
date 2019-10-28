import torch
import math
import copy
import numpy as np
from torch.optim.optimizer import Optimizer, required


class DTSSGD(Optimizer):
    r"""Implements stochastic gradient descent (optionally with momentum).

    Nesterov momentum is based on the formula from
    `On the importance of initialization and momentum in deep learning`__.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)

    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()

    __ http://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf

    .. note::
        The implementation of SGD with Momentum/Nesterov subtly differs from
        Sutskever et. al. and implementations in some other frameworks.

        Considering the specific case of Momentum, the update can be written as

        .. math::
                  v = \rho * v + g \\
                  p = p - lr * v

        where p, g, v and :math:`\rho` denote the parameters, gradient,
        velocity, and momentum respectively.

        This is in contrast to Sutskever et. al. and
        other frameworks which employ an update of the form

        .. math::
             v = \rho * v + lr * g \\
             p = p - v

        The Nesterov version is analogously modified.

        grad_list is used to store the gradient values with window_size
    """

    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, window_size=3, a=1):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if window_size < 1.0:
            raise ValueError("Invalid windows_size value: {}".format(window_size))
        if a < 0.0:
            raise ValueError("Invalid a value:{}".format(a))
        self.grad_list = []
        self.alpha = [math.pow(a, i) for i in range(window_size)]
        self.a = a

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov,
                        window_size=window_size, a=a)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(DTSSGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(DTSSGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, t=1, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            window_size = group['window_size']

            new_grad = []
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = copy.deepcopy(p.grad.data)
                new_grad.append(d_p)

            if len(self.grad_list) == window_size:
                self.grad_list.pop(0)
            self.grad_list.append(new_grad)

            # # using vector to calculate the result grad
            # a1 = np.mat(self.grad_list)
            # a2 = np.mat(self.alpha[:len(self.grad_list)])
            # result_grad = a2 * a1
            # result_grad = result_grad / len(self.grad_list)
            # result_grad = np.array(result_grad)
            # count = 0
            # for p in group['params']:
            #     p.data.add_(-group['lr'] / math.sqrt(t), result_grad[-1][count])
            #     count += 1

            count = 0
            for p in group['params']:
                sum_grad = 0
                denominator = 0
                for i in range(len(self.grad_list)):
                    sum_grad += (math.pow(self.a, len(self.grad_list) - 1 - i) * self.grad_list[i][count])
                    denominator += math.pow(self.a, len(self.grad_list) - 1 - i)
                result_grad = sum_grad / denominator
                p.data.add_(-group['lr'] / math.sqrt(t), result_grad)
                count += 1

        return loss

