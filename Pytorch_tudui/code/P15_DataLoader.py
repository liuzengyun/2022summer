#coding=utf-8
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

data_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

test_data = torchvision.datasets.CIFAR10(root='./dataset', train=False, download=True, transform=data_transform)
test_loader = DataLoader(dataset=test_data, batch_size=64, shuffle=True, num_workers=0, drop_last=False)

# 测试集中的第一张图和标签
img, target = test_data[0]
print(img.shape)
print(target)

writer = SummaryWriter('tb_logs/p15logs')
step = 0
for data in test_loader:
    imgs, tars = data
    # print(imgs.shape)
    # print(tars)
    writer.add_images('test_data', imgs, step)
    step = step + 1





