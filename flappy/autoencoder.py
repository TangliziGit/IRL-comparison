import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision import transforms, datasets, models
# import visdom
import time
import numpy as np

from encoder_net import AutoEncoder

np.random.seed(123)
torch.manual_seed(123)


# viz = visdom.Visdom()

BATCH_SIZE = 64
LR = 0.001
EPOCHS = 100

# USE_GPU = True
# if USE_GPU:
#     gpu_status = torch.cuda.is_available()
# else:
#     gpu_status = False
#
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# 图像预处理，408x408x1
trainTransform = transforms.Compose([
    transforms.Resize((408, 408)),
    transforms.Grayscale(),
    transforms.ToTensor(),])

train_dataset = datasets.ImageFolder('./train_images2/', transform=trainTransform)
test_dataset = datasets.ImageFolder('./test_images/', transform=trainTransform)

train_loader = DataLoader(train_dataset, BATCH_SIZE, True)
test_loader = DataLoader(test_dataset, 400, False)

# dataiter = iter(train_loader)
# inputs, labels = dataiter.next()
# 可视化visualize
# viz.images(inputs[:16], nrow=8, padding=3)
# time.sleep(0.5)
# image = viz.images(inputs[:16], nrow=8, padding=3)

net = AutoEncoder().cuda()
#权值初始化

optimizer = torch.optim.Adam(net.parameters(), lr=LR)
loss_f = nn.MSELoss()

# scatter=viz.scatter(X=np.random.rand(2, 2), Y=(np.random.rand(2) + 1.5).astype(int), opts=dict(showlegend=True))

for epoch in range(EPOCHS):
    net.train()
    for step, (images, _) in enumerate(train_loader, 1):
        net.zero_grad()
        data = images.cuda()
        code, decoded = net(data)
        loss = loss_f(decoded, data)
        loss.backward()
        optimizer.step()

        print('Train Epoch: {} [{}/{} ({:.0f}%)]  \tLoss: {:.6f}'.format(
    	    epoch, step * len(data), len(train_loader.dataset),
    	    100. * step / len(train_loader), loss.item()))

    if epoch % 5 == 0:
        net.eval()
        for step, (images, _) in enumerate(test_loader, 1):
            data = images.cuda()
            code, decoded = net(data)
            loss = loss_f(decoded, data)

        print('Test [{}/{} ({:.0f}%)]  \tLoss: {:.6f}'.format(
    	    step * len(data), len(test_loader.dataset),
    	    100. * step / len(test_loader), loss.item()))

torch.save(net.state_dict(), 'encoder2.pkl')
