#coding=utf-8
'''
tensor数据类型:
    通过Transforms.Tensor看两个问题
    1、Transforms怎么使用
    2、Tensor数据类型的优势
'''
from torchvision import transforms
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

img_path = "data/data/train/ants_image/0013035.jpg"
img = Image.open(img_path)

writer = SummaryWriter('tb_logs/logs')


# 1、Transforms怎么使用:创建工具，使用工具
tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img)
print(type(tensor_img))
writer.add_image("pic", tensor_img)
writer.close()




