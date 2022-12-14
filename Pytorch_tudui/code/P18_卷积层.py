# coding=utf-8
# Time:2022/8/6 下午4:33
# Author:liuzengyun
# Email:i@cvzoo.cn
import torch
from torch import nn
import torchvision
from torch.nn import Conv2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10(root='./dataset', train=False, download=True, transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=64)

writer = SummaryWriter('tb_logs/p18logs')


class Tudui(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0)

    def forward(self, input):
        output = self.conv1(input)
        return output


tudui = Tudui()

step = 0
for data in dataloader:
    imgs, targets = data
    output = tudui(imgs)
    print(imgs.shape)
    print(output.shape)

    writer.add_images('input', imgs, step)

    output = torch.reshape(output, ([-1, 3, 30, 30]))
    writer.add_images('output', output, step)

    step = step + 1


writer.close()



