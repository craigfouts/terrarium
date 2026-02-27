'''
Author(s): Craig Fouts
Correspondence: c.fouts25@imperial.ac.uk
License: Apache 2.0 license
'''

import torch
from torch import nn
from tqdm import tqdm
from ..utils.nets import OPTIMS
from ..utils.sugar import attrmethod, buildmethod

__all__ = [
    'GLV'
]

class GLV(nn.Module):
    @attrmethod
    def __init__(self, optim='adam', *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.log_ = []
    
    def _build(self, Y, x=None, learning_rate=1e-3):
        n = Y.shape[1] - 1

        if x is None:
            x = torch.zeros(n)

        self.r_ = nn.Parameter(torch.zeros_like(x), requires_grad=True)
        self.A_ = nn.Parameter(torch.zeros(n, n), requires_grad=True)
        self._optim = OPTIMS[self.optim](self.parameters(), learning_rate)

        return self

    def step(self, loss):
        loss.backward()
        self._optim.step()
        self._optim.zero_grad()
        self.log_.append(loss.item())

        return self

    def model(self, x, r=None, A=None):
        if r is None:
            r = self.r_

        if A is None:
            A = self.A_

        dx = x*(r + A@x)

        return dx

    def forward(self, x, y=None, dt=.1, r=None, A=None):
        x_ = x + self.model(x, r, A)*dt

        if y is not None:
            loss = (d := x_ - y).norm(1) + d.norm(2)

            return x_, loss
        return x_

    @buildmethod
    def fit(self, Y, x=None, dt=.1, n_epochs=2500, n_steps=350, learning_rate=1e-3, verbosity=1):
        for j in tqdm(range(n_epochs)) if verbosity > 0 else range(n_epochs):
            x_, t, loss = x.clone(), -1, 0

            for i in range(n_steps):
                if i in Y[1:, 0]:
                    x_, l = self(x_, Y[t := t + 1, 1:])
                    loss += l
                else:
                    x_ = self(x_)

            self.step(loss)

        return self
