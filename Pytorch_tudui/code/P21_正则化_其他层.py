# coding=utf-8
# Time:2022/8/11 下午10:57
# Author:liuzengyun
# Email:i@cvzoo.cn


import torchvision
import torch
from torch.utils.data import DataLoader

data_set = torchvision.datasets.CIFAR10("./dataset", train=False, transform=torchvision.transforms.ToTensor(),
                                        download=True)
dataloader = DataLoader(data_set,batch_size=64)

class Tudui(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.linear1 = torch.nn.Linear(196608, 10)

    def forward(self, input):
        output = self.linear1(input)
        return output



tudui = Tudui()
for data in dataloader:
    imgs, tars = data
    print(imgs.shape)
    output = torch.flatten(imgs)# 展平，展成一行数据
    print(output.shape)
    output = tudui(output)
    print(output.shape)

# 报错是因为最后一个batch的数据展开不够196608




