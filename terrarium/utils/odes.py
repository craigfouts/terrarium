'''
Author(s): Craig Fouts
Correspondence: c.fouts25@imperial.ac.uk
License: Apache 2.0 license
'''

import numpy as np
import torch
from functools import singledispatch
from tqdm import tqdm

__all__ = [
    'glv',
    'rk4_int'
]

def _k4(x, model, dt=.1, *args, **kwargs):
    k1 = model(x, *args, **kwargs)
    k2 = model(x + k1*dt/2, *args, **kwargs)
    k3 = model(x + k2*dt/2, *args, **kwargs)
    k4 = model(x + k3*dt, *args, **kwargs)

    return k1, k2, k3, k4

@singledispatch
def rk4_int(x, model, dt=.1, n_steps=1, verbosity=0, *args, **kwargs):
    (X := np.zeros((len(x), n_steps)))[:, 0] = x

    for i in tqdm(range(1, n_steps)) if verbosity > 0 else range(1, n_steps):
        k1, k2, k3, k4 = _k4(x_ := X[:, i - 1], model, dt, *args, **kwargs)
        X[:, i] = x_ + (k1 + 2*k2 + 2*k3 + k4)*dt/6.

    return X[:, 0] if n_steps == 1 else X

@rk4_int.register(torch.Tensor)
def _(x, model, dt=.1, n_steps=1, verbosity=0, *args, **kwargs):
    (X := torch.zeros(len(x), n_steps))[:, 0] = x

    for i in tqdm(range(1, n_steps)) if verbosity > 0 else range(1, n_steps):
        k1, k2, k3, k4 = _k4(x_ := X[:, i - 1], model, dt, *args, **kwargs)
        X[:, i] = x_ + (k1 + 2*k2 + 2*k3 + k4)*dt/6.

    return X[:, 0] if n_steps == 1 else X

def glv(x, r, A):
    dx = x*(r + A@x)

    return dx
