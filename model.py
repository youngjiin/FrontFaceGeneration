import torch.nn as nn
import torch

class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(1024, 1024, 4, 1, 0, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(True),
            nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, style): #attr ( 1, 512)
        style = style.view(-1, 1024, 1, 1) #(10, 512, 1, 1)
        #x = torch.cat([x, style], 1) #(10 , 612, 1,1)

        return self.main(style)

class StyleEncoder(nn.Module):
    def __init__(self):
        super(StyleEncoder, self).__init__()
        self.first = nn.Conv2d(3, 64, 4, 2, 1, bias=False)
        self.main = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.attention = nn.Conv2d(256, 1, 1, 1, 0)

        self.last = nn.Sequential(
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1024, 4, 2, 1, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(1024, 512, 4, 1, 0, bias=False)
        )

    def forward(self, x):
        #update the wieghts
        x = self.first(x)
        x = self.main(x) #[10, 1024, 4,4]
        attr = self.attention(x)
        attr = torch.sigmoid(attr)
        s = attr * x
        s = self.last(s) #what dimension ?
        s = s.squeeze(3).squeeze(2)
        return s

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.feature_input = nn.Linear(1024, 128 * 128)
        self.first = nn.Conv2d(3 + 1, 64, 4, 2, 1, bias=False)
        self.main = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1024, 4, 2, 1, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.last = nn.Conv2d(1024, 1, 4, 1, 0, bias=False)

    def forward(self, x, attr):
        attr = self.feature_input(attr).view(-1, 1, 128, 128)
        x = torch.cat([x, attr], 1)
        x = self.first(x)
        x = self.main(x)
        return self.last(x).view(-1, 1)