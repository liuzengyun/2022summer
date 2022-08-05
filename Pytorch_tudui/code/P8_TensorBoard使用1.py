from torch.utils.tensorboard import SummaryWriter
#  打开tensorboard：tensorboard --logdir=tb_logs
#                  tensorboard --logdir=tb_logs --port=9009

writer = SummaryWriter("tb_logs")

# writer.add_image()

for i in range(100):
    writer.add_scalar("y=x", i*i, i)

writer.close()


