import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, input):
        return input.view(input.size(0), -1)


class STNv0(nn.Module):
    def __init__(self, xdim, args):
        super(STNv0, self).__init__()
        self.xdim = xdim
        self.args = args
        hdim = self.args.stn_hid_dim
        self.fcx = int(xdim[1] / 4) if xdim[1] == 28 else int(xdim[1]/8)
        print(self.fcx)
        # get the module
        self.identity_transform = torch.tensor([[1, 0, 0, 0, 1, 0]], dtype=torch.double)
        self.identity_transform = Variable(self.identity_transform)
        if args is None:
            dropout = 0.5
        else:
            dropout = args.dropout
        self.dropout = dropout
        print('Using dropout of {} for STN'.format(dropout))
        module = []
        module.append(conv_block(xdim[0], hdim))
        module.append(conv_block(hdim, hdim))
        if self.xdim[1] > 28:
            module.append(conv_block(hdim, hdim))
        module.append(Flatten())
        # This is 7x7
        module.append(nn.Linear(hdim * self.fcx * self.fcx, 32))
        module.append(nn.ReLU())
        module.append(nn.Linear(32, 6))
        module.append(nn.Tanh())
        self.module = nn.Sequential(*module)
        self._init_weights()

    def _init_weights(self):
        # initialize weights here
        index = -2
        self.module[index].weight.data.zero_()
        self.module[index].bias.data.copy_(self.identity_transform.squeeze())

    def forward(self, sample, support=False):
        # do the actual forward passes
        # dropout probability for dropping the final theta and putting
        # default value of [1....10]
        self.identity_transform = self.identity_transform.to(sample.device)
        if self.training and not support:
            dropout = self.dropout
        else:
            dropout = 1

        sample = Variable(sample)
        inp_flatten = sample
        # do the forward pass
        theta = self.module(inp_flatten)
        theta = theta + 0

        # Scale it to have any values
        B = sample.shape[0]
        U = torch.rand(B)
        idx = (U <= dropout).nonzero().squeeze()
        theta[idx, :] = self.identity_transform
        # change the shape
        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, inp_flatten.size())
        results = F.grid_sample(inp_flatten, grid)

        transform = theta
        return results, transform, {}


class STNv1(nn.Module):
    def __init__(self, xdim, args):
        super(STNv1, self).__init__()
        hdim = args.stn_hid_dim
        self.xdim = xdim
        self.args = args
        self.fcx = int(xdim[1] / 4) if xdim[1] == 28 else int(xdim[1]/8)
        print(self.fcx)
        # get the module
        self.identity_transform = torch.tensor([[1, 0, 0, 0, 1, 0]], dtype=torch.double)
        self.identity_transform = Variable(self.identity_transform)
        if args is None:
            dropout = 0.5
        else:
            dropout = args.dropout
        self.dropout = dropout
        print('Using dropout of {} for STN'.format(dropout))
        module = []
        module.append(conv_block(xdim[0], hdim))
        module.append(conv_block(hdim, hdim))
        if self.xdim[1] > 28:
            module.append(conv_block(hdim, hdim))
        module.append(Flatten())
        # This is 7x7
        module.append(nn.Linear(hdim * self.fcx * self.fcx, 16))
        module.append(nn.ReLU())
        self.module = nn.Sequential(*module)
        # Add modules for scale, rotation, and translation
        self.scaler = nn.Linear(16, 1)
        self.theta = nn.Linear(16, 1)
        self.translation = nn.Linear(16, 2)

    def forward(self, sample, support=False):
        # do the actual forward passes
        # dropout probability for dropping the final theta and putting
        # default value of [1....10]
        self.identity_transform = self.identity_transform.to(sample.device)
        # Training and this is not the support set
        if self.training and not support:
            dropout = self.dropout
        else:
            dropout = 1

        sample = Variable(sample)
        inp_flatten = sample
        # do the forward pass
        params = self.module(inp_flatten)

        # Theta, translation and scale
        eps = self.args.scalediff
        theta = (torch.tanh(self.theta(params)) * self.args.theta).squeeze()
        txty  = torch.tanh(self.translation(params)) * self.args.t
        tx = txty[:, 0]
        ty = txty[:, 1]
        scale = (torch.tanh(self.scaler(params)) * eps + 1).squeeze()

        # Horizontal flip
        flip = torch.rand_like(scale) <= self.args.fliphoriz
        flip = flip.to(sample.device).double()
        flip = 1 - 2*flip

        # Get theta
        B = scale.shape[0]
        Theta = torch.zeros((B, 6)).double()
        Theta = Theta.to(sample.device)
        Theta[:, 0] = scale*flip*torch.cos(theta)
        Theta[:, 1] = -scale*torch.sin(theta)
        Theta[:, 2] = tx
        Theta[:, 3] = scale*flip*torch.sin(theta)
        Theta[:, 4] = scale*torch.cos(theta)
        Theta[:, 5] = ty
        # Back to theta
        theta = Theta

        # Scale it to have any values
        U = torch.rand(B)
        idx = (U <= dropout).nonzero().squeeze()
        theta[idx, :] = self.identity_transform
        # change the shape
        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, inp_flatten.size())
        results = F.grid_sample(inp_flatten, grid)

        transform = theta
        return results, transform, {}

# STN-VAE
class STNVAE(nn.Module):

    """STNVAE - Basically an extension where we get 2 outputs
    which acts as mean and variance
    """

    def __init__(self, xdim, hdim=64, dropout=0.5):
        """TODO: to be defined. """
        super(STNVAE, self).__init__()
        self.xdim = xdim
        self.fcx = int(xdim[1] / 4) if xdim[1] == 28 else int(xdim[1]/8)
        print(self.fcx)
        # get the module
        self.identity_transform = torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float)
        self.identity_transform = Variable(self.identity_transform)
        self.dropout = dropout
        print('Using VAE STN')
        module = []
        fc = []
        module.append(conv_block(xdim[0], hdim))
        module.append(conv_block(hdim, hdim))
        if xdim[1] > 28:
            module.append(conv_block(hdim, hdim))
        module.append(Flatten())
        # This is 7x7
        module.append(nn.Linear(hdim * self.fcx * self.fcx, 64))
        # Get mean variance here
        fc.append(nn.Linear(32, 32))
        fc.append(nn.ReLU())
        fc.append(nn.Linear(32, 6))
        fc.append(nn.Tanh())

        self.module = nn.Sequential(*module)
        self.fc = nn.Sequential(*fc)
        self._init_weights()

    def _init_weights(self):
        # initialize weights here
        index = -2
        self.fc[index].weight.data.zero_()
        self.fc[index].bias.data.copy_(self.identity_transform)

    def forward(self, sample, ):
        # do the actual forward passes
        # dropout probability for dropping the final theta and putting
        # default value of [1....10]
        results = dict()
        transform = []
        self.identity_transform = self.identity_transform.to(sample['xs'].device)
        info = dict(mean=[], logstd=[])
        for k in ['xs', 'xq']:
            sample[k] = Variable(sample[k])
            inp = sample[k]
            n_classes, n_shot = inp.shape[:2]
            inp_flatten = inp.view(n_classes*n_shot, *inp.shape[2:])
            # do the forward pass
            out = self.module(inp_flatten)
            outm = out[:, :32]
            outlogstd = out[:, 32:]
            outstd = torch.exp(outlogstd)
            out = outm + torch.randn_like(outstd).to(outstd.device)*outstd
            info['mean'].append(outm)
            info['logstd'].append(outlogstd)
            # Get theta
            theta = self.fc(out)
            # Scale it to have any values
            B = theta.shape[0]
            U = torch.rand(B) < self.dropout
            theta = theta + 0
            theta[U] = self.identity_transform
            # change the shape
            theta = theta.view(-1, 2, 3)
            grid = F.affine_grid(theta, inp_flatten.size())
            x = F.grid_sample(inp_flatten, grid)
            # put into results
            results[k] = x.view(*inp.shape)
            transform.append(theta)
        # Copy all the other non-tensor keys
        for k in sample.keys():
            if k in ['xs', 'xq']:
                continue
            results[k] = sample[k]
        return results, transform, info


