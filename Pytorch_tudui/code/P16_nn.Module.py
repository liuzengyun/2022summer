# coding=utf-8
# Time:2022/8/6 上午10:36
# Author:liuzengyun
# Email:i@cvzoo.cn
import torch
from torch import nn


class Tudui(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, input):
        output = input + 1
        return output


tudui = Tudui()

x = torch.tensor(1.0)
output = tudui(x)
print(output)