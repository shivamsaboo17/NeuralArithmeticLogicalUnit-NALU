from nac import NAC
from nalu import NALU
import torch.nn as nn
import torch
from torch.autograd import Variable
import torch.nn.functional as F


class AdaptiveModule(nn.Module):
    
    def __init__(self, ni, no):
        super().__init__()

        self.conv = nn.Conv2d(ni, no, 1)
        self.adaptive_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        return self.adaptive_pool(self.conv(x))


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class BuildHead(nn.Module):

    def __init__(self, conv_model, op_channels, hidden_dims, task='classification', num_outputs=2, head_type='NALU'):
        super().__init__()

        if task == 'regression' and num_outputs != 1:
            raise ValueError(f'Expected num_output 1, got {num_outputs}') 

        self.conv_model = conv_model
        self.task = task
        self.num_classes = num_outputs
        self.head_type = head_type
        self.hidden_dims = hidden_dims

        self.adaptive_module = AdaptiveModule(op_channels, hidden_dims[0])
        self.flatten = Flatten()
        self.head = self.get_head(self.hidden_dims, self.num_classes, self.head_type)

    def forward(self, x):
        out = self.head(self.flatten(self.adaptive_module(self.conv_model(x))))
        out = F.log_softmax(out, dim=1) if self.task == 'classification' else out
        return out


    def get_head(self, hidden_dims, num_classes, head_type):
        dims = hidden_dims + [num_classes]
        head = None
        if head_type == 'NALU':
            head = NALU(*dims)
        elif head_type == 'NAC':
            head = NAC(*dims)
        else:
            raise ValueError(f'{head_type} not supported')
        return head


