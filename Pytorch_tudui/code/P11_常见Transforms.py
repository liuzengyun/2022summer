from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

writer = SummaryWriter('tb_logs/p11logs')


img_path = "data/data/train/ants_image/0013035.jpg"
img = Image.open(img_path)
print(img)

# ToTensor
trans_totensor = transforms.ToTensor()
img_tensor = trans_totensor(img)
writer.add_image('ToTensor', img_tensor)

# Normalize
# output[channel] = (input[channel] - mean[channel]) / std[channel]
print(img_tensor[0][0][0])
trans_norm = transforms.Normalize([0.5, 0.5, 0.5],[0.5, 0.5, 0.5])
img_norm = trans_norm(img_tensor)
print(img_norm[0][0][0])
writer.add_image('Norm', img_norm)

# Resize
print(img.size)
trans_resize = transforms.Resize((512, 512))
img_resize = trans_resize(img)
print(img_resize.size)
writer.add_image('resize', trans_totensor(img_resize))

# Compose  resize2
trans_resize_2 = transforms.Resize(256)
trans_compose = transforms.Compose([trans_resize, trans_resize_2, trans_totensor])
img_compose = trans_compose(img)
writer.add_image('compose', img_compose)


writer.close()


