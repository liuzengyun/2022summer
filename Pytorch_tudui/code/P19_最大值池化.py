# coding=utf-8
# Time:2022/8/11 下午6:16
# Author:liuzengyun
# Email:i@cvzoo.cn
import torch
from torch import nn
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10(root='./dataset', train=False, download=True, transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=64)

# input = torch.tensor([[1,2,0,3,1],
#                      [0,1,2,3,1],
#                      [1,2,1,0,0],
#                      [5,2,3,1,1],
#                      [2,1,0,1,1]],
#                      dtype=torch.float32)
# input = torch.reshape(input, (-1, 1, 5, 5))

class Tudui(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, ceil_mode=True)


    def forward(self, input):
        output = self.maxpool1(input)
        return output


tudui = Tudui()
# output = tudui(input)
# print(output)

writer = SummaryWriter('tb_logs/p19logs')

step = 0
for data in dataloader:
    imgs, targets = data
    writer.add_images('input', imgs, step)
    step = step + 1
    output = tudui(imgs)
    writer.add_images('output', output, step)



writer.close()






