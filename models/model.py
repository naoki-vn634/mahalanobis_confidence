import torch
import torchvision
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import re
import sys
import torch.utils.checkpoint as cp
from collections import OrderedDict
from torch import Tensor
from torch.jit.annotations import List
import torchvision


class CustomDensenet(nn.Module):
    def __init__(self, hidden_size=256, num_classes=2, convert=False):
        super(CustomDensenet, self).__init__()
        dense = models.densenet161(pretrained=True)
        self.convert = convert
        self.densenet = nn.Sequential(*list(dense.children())[:-1])
        self.avg = nn.AvgPool2d(7)
        self.fc1 = nn.Linear(2208, hidden_size)
        self.relu = nn.ReLU()
        self.dr1 = nn.Dropout()
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.device = self.init_device()

    def init_device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def converter(self, x):
        for i, logits in enumerate(x):
            neg = torch.log(torch.exp(logits[0]) + torch.exp(logits[2])).to(self.device)
            pos_conv = torch.unsqueeze(
                torch.tensor([neg, logits[1]]),
                dim=0,
            )
            pos_c = torch.cat([pos_c, pos_conv], dim=0) if i != 0 else pos_conv

        return pos_c

    def forward(self, x):
        h = self.densenet(x)
        h = self.avg(h)
        h_middle = h.view(len(h), -1)
        h = self.dr1(self.relu(self.fc1(h_middle)))
        y = self.fc2(h)
        if self.convert:
            y = self.converter(y).to(self.device)

        return y, h_middle
        # return y, h

    def feature_list(self, x):
        out_list = []
        out_layers = []
        out = self.densenet[0].relu0(self.densenet[0].norm0(self.densenet[0].conv0(x)))
        out_list.append(out)
        out_layers.append(out.size()[-1])
        out = self.densenet[0].pool0(out)
        out_list.append(out)
        out_layers.append(out.size()[-1])
        out = self.densenet[0].transition1(self.densenet[0].denseblock1(out))
        out_list.append(out)
        out_layers.append(out.size()[-1])
        out = self.densenet[0].transition2(self.densenet[0].denseblock2(out))
        out_list.append(out)
        out_layers.append(out.size()[-1])
        out = self.densenet[0].transition3(self.densenet[0].denseblock3(out))
        out_list.append(out)
        out_layers.append(out.size()[-1])
        out = self.densenet[0].denseblock4(out)
        out_list.append(out)
        out_layers.append(out.size()[-1])
        out = self.densenet[0].norm5(out)
        out_list.append(out)
        out_layers.append(out.size()[-1])
        h = self.avg(out)
        h_middle = h.view(len(h), -1)
        h = self.dr1(self.relu(self.fc1(h_middle)))
        y = self.fc2(h)

        return out_list, out_layers, y
