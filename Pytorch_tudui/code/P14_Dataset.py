import torchvision
from torch.utils.tensorboard import SummaryWriter

data_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

train_set = torchvision.datasets.CIFAR10(root='./dataset', train=True, download=True, transform=data_transform)
test_set = torchvision.datasets.CIFAR10(root='./dataset', train=False, download=True, transform=data_transform)

# print(train_set[0])
# print(train_set.classes)
#
# img, target = train_set[0]
# img.show()
writer = SummaryWriter('tb_logs/p14logs')
for i in range(10):
    img, target = test_set[i]
    writer.add_image('test_image', img, i)


writer.close()