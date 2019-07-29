import math
import torch
from torch import nn
from torchvision import transforms
import numpy as np

HIDDEN_SIZE = 1000

class AutoEncoder(nn.Module):#(288, 512, 3)
    def __init__(self):
        super(AutoEncoder, self).__init__()

        self.trans=transforms.Compose([
            transforms.Resize((408, 408)),
            transforms.Grayscale(),
            transforms.ToTensor(),
        ])
        self.en_conv = nn.Sequential(#408x408x1
            nn.Conv2d(1, 8, 4, 2),#203x203x8
            nn.BatchNorm2d(8),
            nn.Tanh(),
            nn.Conv2d(8, 16, 3, 2),#101x101x32
            nn.BatchNorm2d(16),
            nn.Tanh(),
            nn.Conv2d(16, 16, 3, 2),#50x50x16
            nn.BatchNorm2d(16),
            nn.Tanh(),
            nn.Conv2d(16, 8, 4, 2),#24x24x8 
            nn.BatchNorm2d(8),
            nn.Tanh(),
        )
        self.en_fc = nn.Linear(24 * 24 * 8, HIDDEN_SIZE)
        self.de_fc = nn.Linear(HIDDEN_SIZE, 24 * 24 * 8)
        self.de_conv = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 4, 2),
            nn.BatchNorm2d(16),
            nn.Tanh(),
            nn.ConvTranspose2d(16, 16, 3, 2),
            nn.BatchNorm2d(16),
            nn.Tanh(),
            nn.ConvTranspose2d(16, 8, 3, 2),
            nn.BatchNorm2d(8),
            nn.Tanh(),
            nn.ConvTranspose2d(8, 1, 4, 2),
            nn.Sigmoid(),
        )

    def forward(self, x):
        en = self.en_conv(x)
        # print(en.shape)
        code = self.en_fc(en.view(en.size(0), -1))
        de = self.de_fc(code)
        decoded = self.de_conv(de.view(de.size(0), 8, 24, 24))
        # print(decoded.shape)
        return code, decoded

    def encode(self, x):
        en = self.en_conv(x)
        return self.en_fc(en.view(en.size(0), -1)).data.numpy()

    def getKL(self, A, B):
        kl=0;ans=0
        A, B=self.trans(A), self.trans(B)
        A, B=self.encode(A.unsqueeze(0)), self.encode(B.unsqueeze(0))
        A, B=np.ravel(A), np.ravel(B)
        # C=np.ones(A.shape) # .copy()
        # for i, (a, b) in enumerate(zip(A, B)):
        #     if a!=0 and b!=0:
        #         C[i]=abs(a/b)
        #     elif a!=0 and b==0:
        #         C[i]=abs((2*a)/(a+b))
        #     else:
        #         C[i]=1
        # kl=np.sum(np.log(C)*A)
        for a, b in zip(A, B): 
            # print(a, b)
            if a!=0 and b!=0:
                ans=abs(a*math.log(abs(a/b)))
            elif a!=0 and b==0:
                ans=abs(a*math.log(abs((a)/(a/2+b/2))))
            elif a==0 and b==0:
                ans=0
            else:
                ans=0
            kl+=ans
        return kl
