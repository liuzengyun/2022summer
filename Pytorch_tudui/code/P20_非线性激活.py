# coding=utf-8
# Time:2022/8/11 下午10:28
# Author:liuzengyun
# Email:i@cvzoo.cn

from torch import nn
import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

input = torch.tensor([[1, -0.5], [-1, 3]])
input = torch.reshape(input, (-1,1,2,2))


dataset = torchvision.datasets.CIFAR10(root='./dataset', train=False, download=True, transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=64)

class Tudui(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()


    def forward(self, input):
        output = self.sig(input)
        return output



tudui = Tudui()

# output = tudui(input)
# print(output)

writer = SummaryWriter('tb_logs/p20logs')

step = 0
for data in dataloader:
    imgs, targets = data
    writer.add_images('input', imgs, step)
    step = step + 1
    output = tudui(imgs)
    writer.add_images('output', output, step)



writer.close()




