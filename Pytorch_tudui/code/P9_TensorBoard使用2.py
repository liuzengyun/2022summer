from torch.utils.tensorboard import SummaryWriter
#  打开tensorboard：tensorboard --logdir=tb_logs
#                  tensorboard --logdir=tb_logs --port=9009
from PIL import Image
import numpy as np





writer = SummaryWriter("tb_logs")

img_path = "data/data/train/ants_image/0013035.jpg"
img_PIL = Image.open(img_path)
img_array = np.array(img_PIL)

print(type(img_array))
print(img_array.shape)
writer.add_image("imgtest", img_array, dataformats='HWC')

for i in range(100):
    writer.add_scalar("y=x", i*i, i)

writer.close()


