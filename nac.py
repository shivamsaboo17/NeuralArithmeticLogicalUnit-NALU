import math
import torch
import torch.nn as nn
import torch.nn.init as weight_init
import torch.nn.functional as F

from torch.nn.parameter import Parameter


class NeuralAccumulatorCell(nn.Module):
    
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.W_hat = torch.Tensor(out_dim, in_dim)
        self.M_hat = torch.Tensor(out_dim, in_dim)
        self.W = Parameter(F.tanh(self.W_hat) * F.sigmoid(self.M_hat))
        self.register_parameter('bias', None)

        self.W_hat = weight_init.kaiming_uniform(self.W_hat, a=math.sqrt(5))
        self.M_hat = weight_init.kaiming_uniform(self.M_hat, a=math.sqrt(5))


    def forward(self, input):
        return F.linear(input, self.W, self.bias)

    def extra_repr(self):
        return 'IN={}, OUT={}'.format(self.in_dim, self.out_dim)


class NAC(nn.Module):
    
    def __init__(self, *dims):
        super().__init__()

        layers = []
        for i in range(len(dims) - 1):
            layers += [NeuralAccumulatorCell(dims[i + 1], dims[i])]
        self.model = nn.Sequential(*layers)
    
    def forward(self, input):
        out = self.model(input)
        return out
