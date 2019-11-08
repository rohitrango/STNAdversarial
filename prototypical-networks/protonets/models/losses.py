import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from protonets.models import register_model
from .utils import euclidean_dist

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

    def forward(self, transform, params):
        y = Variable(torch.Tensor([[1, 0, 0], [0, 1, 0]]))[None]
        y = y.to(transform[0].device)
        loss = 0
        for tr in transform:
            loss_ = (tr - y)**2
            loss = loss + loss_.sum()
        loss = loss / len(transform)
        loss = params * loss
        return loss


def kl_divergence(info):
    means = info['mean']
    logstd = info['logstd']
    loss = 0
    for m, ls in zip(means, logstd):
        s = torch.exp(ls)
        loss += (m**2 + s**2 - 2*ls - 1).sum()
    return loss


@register_model('nothetaloss')
def nothetaloss(**kwargs):
    return NoThetaLoss()

@register_model('identitydistance')
def identitydistance(**kwargs):
    return IdentityTransformLoss()

