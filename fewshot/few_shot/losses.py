import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class NoThetaLoss(nn.Module):

    """Does nothing given some input (returns 0)"""

    def __init__(self):
        nn.Module.__init__(self)

    def forward(self, transform, params):
        return 0

''' Get rank loss so that you can't have just any matrix '''
class IdentityTransformLoss(nn.Module):

    """Get nuclear norm"""

    def __init__(self):
        """ """
        nn.Module.__init__(self)

    def forward(self, transform):
        y = Variable(torch.Tensor([[1, 0, 0], [0, 1, 0]]))[None].double()
        y = y.to(transform.device)
        loss = 0
        loss = (transform - y)**2
        loss = loss.sum(1).mean()
        return loss


def kl_divergence(info):
    means = info['mean']
    logstd = info['logstd']
    loss = 0
    for m, ls in zip(means, logstd):
        s = torch.exp(ls)
        loss += (m**2 + s**2 - 2*ls - 1).sum()
    return loss

