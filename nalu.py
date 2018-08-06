import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as weight_init
import torch.nn.functional as F

from torch.nn.parameter import Parameter
from nac import NeuralAccumulatorCell


class NeuralArithmeticLogicUnit(nn.Module):
    
    def __init__(self, in_dims, out_dims):
        super().__init__()
        self.in_dims = in_dims
        self.out_dims = out_dims
        self.eps = 1e-10

        self.G = Parameter(torch.Tensor(out_dims, in_dims))
        self.W = Parameter(torch.Tensor(out_dims, in_dims))
        self.register_parameter('bias', None)
        self.nac = NeuralAccumulatorCell(in_dims, out_dims)

        self.G = weight_init.kaiming_uniform(self.G, a=math.sqrt(5))
        self.W = weight_init.kaiming_uniform(self.W, a=math.sqrt(5))

    def forward(self, input):
        nac_op = self.nac(input)
        g = F.sigmoid(F.linear(input, self.G, self.bias))
        add_sub = g * nac_op
        log_ip = torch.log(torch.abs(input) + self.eps)
        m = torch.exp(F.linear(log_ip, self.W, self.bias))
        mul_div = (1 - g) * m
        y = add_sub + mul_div
        return y


class NALU(nn.Module):
    
    def __init__(self, *dims):
        super().__init__()
        layers = []

        for i in range(len(dims) - 1):
            layers += [NeuralArithmeticLogicUnit(dims[i], dims[i + 1])]
        
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

