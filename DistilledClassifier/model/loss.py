import torch.nn as nn
from losses import *

def supervised_loss(method):
    if method == 'CE':
        loss = nn.CrossEntropyLoss()
    else:
        raise NotImplementedError
    return loss

def inheritance_loss(method):
    if method == 'COS':
        loss = COS()
    else:
        raise NotImplementedError
    return loss
