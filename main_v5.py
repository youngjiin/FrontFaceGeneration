#residual version
import argparse

parser = argparse.ArgumentParser("cDCGAN")

parser.add_argument('--dataset_dir', type=str, default='/home/ubuntu/dataset')
parser.add_argument('--result_dir', type=str, default='./cDCGAN_result')
parser.add_argument('--condition_file', type=str, default='./list_attr_cDCGAN.txt')

parser.add_argument('--batch_size', type=int, default=10)
parser.add_argument('--nepoch', type=int, default=100)
parser.add_argument('--nz', type=int, default=100) # number of noise dimension
parser.add_argument('--nc', type=int, default=3) # number of result channel
parser.add_argument('--nfeature', type=int, default=512) # num of embedding
parser.add_argument('--lr', type=float, default=0.0002)
betas = (0.0, 0.99) # adam optimizer beta1, beta2

config, _ = parser.parse_known_args()

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
import torchvision.transforms as transforms
from torch.autograd import Variable
from math import sqrt
from facenet_pytorch import MTCNN, InceptionResnetV1# If required, create a face detection pipeline using MTCNN:

class ResidualBlock(nn.Module):
    """Residual Block with instance normalization."""
    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True))

    def forward(self, x):
        return x + self.main(x)

class Generator(nn.Module):
    """Generator network."""
    def __init__(self, conv_dim=1024, c_dim=config.nfeature, repeat_num=6):
        super(Generator, self).__init__()
        '''
        #shape test
        self.c1 = nn.ConvTranspose2d(config.nz+c_dim, conv_dim, kernel_size=4, stride=1, padding=0, bias=False) #612 , 1024
        self.c2 = nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1, bias=False)
        self.c3 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False)
        self.c4 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False)
        self.c5 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False)
        self.c6 = nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1, bias=False)
        '''

        layers = []
        layers.append(nn.ConvTranspose2d(config.nz+c_dim, conv_dim, kernel_size=4, stride=1, padding=0, bias=False))
        layers.append(nn.InstanceNorm2d(conv_dim, affine=True, track_running_stats=True))
        layers.append(nn.ReLU(inplace=True))

        # Down-sampling layers.
        curr_dim = conv_dim
        for i in range(2):
            layers.append(nn.ConvTranspose2d(curr_dim, curr_dim// 2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim// 2, affine=True, track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim // 2

        # Bottleneck layers.
        for i in range(repeat_num):
            layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))

        # Up-sampling layers.
        for i in range(2):
            layers.append(nn.ConvTranspose2d(curr_dim, curr_dim//2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim//2, affine=True, track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim // 2

        layers.append(nn.ConvTranspose2d(curr_dim, 3, kernel_size=4, stride=2, padding=1, bias=False))
        layers.append(nn.Tanh())
        self.main = nn.Sequential(*layers)

    def forward(self, x, c):
        # Replicate spatially and concatenate domain information.
        # Note that this type of label conditioning does not work at all if we use reflection padding in Conv2d.
        # This is because instance normalization ignores the shifting (or bias) effect.
        c = c.view(-1, config.nfeature, 1, 1)
        x = torch.cat([x, c], dim=1)
        #test shape
        '''
        x = self.c1(x) # [10,612,1,1]
        x = self.c2(x) # -> 10 1024 4 4
        x = self.c3(x) # -> 10 512 8 8
        x = self.c4(x) # -> 10 256 16 16
        x = self.c5(x) # -> 10 128 32 32
        x = self.c6(x) # -> 10 3 128 128
        '''
        return self.main(x)

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
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1024, 4, 2, 1, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.last = nn.Conv2d(1024, 512, 4, 1, 0, bias=False)

    def forward(self, x):
        #update the wieghts
        x = self.first(x)
        x = self.main(x) #[10, 1024, 4,4]
        s = self.last(x) #what dimension ?
        return s.squeeze(3).squeeze(2)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.feature_input = nn.Linear(config.nfeature, 128 * 128)
        self.first = nn.Conv2d(config.nc + 1, 64, 4, 2, 1, bias=False)
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

def get_infinite_batches(data_loader):
    while True:
        for i, (images, _) in enumerate(data_loader):
            yield images

class Trainer:
    def __init__(self):
        self.generator = Generator()
        self.discriminator = Discriminator()
        self.styleencoder = StyleEncoder()
        self.mtcnn = MTCNN(image_size=128, margin=20)
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval()
        self.loss = nn.MSELoss()
        self.optimizer_g = optim.Adam(self.generator.parameters(), lr=config.lr, betas=betas)
        self.optimizer_d = optim.Adam(self.discriminator.parameters(), lr=config.lr, betas=betas)
        self.optimizer_s = optim.Adam(self.styleencoder.parameters(), lr=config.lr, betas=betas)
        self.generator.cuda()
        self.discriminator.cuda()
        self.styleencoder.cuda()
        self.loss.cuda()

    def get_cropped_image(self, image):
        images = []
        for i in range(image.size(0)):
            images.append(self.mtcnn(torchvision.transforms.ToPILImage()(image[i])))
        images = torch.cat(images).view(10, 3, 128,128)
        return images

    def get_embedding_from_image(self, cropped_image):
        embeddings=[]
        for i in range(cropped_image.size(0)):
            embeddings.append(self.resnet(cropped_image[i].unsqueeze(0)))
        embeddings = torch.cat(embeddings)
        return embeddings

    def train(self):
        noise = Variable(torch.FloatTensor(config.batch_size, config.nz, 1, 1).cuda())
        label_real = Variable(torch.FloatTensor(config.batch_size, 1).fill_(1).cuda())
        label_fake = Variable(torch.FloatTensor(config.batch_size, 1).fill_(0).cuda())
        ds = Dataset(config)
        profile_data = get_infinite_batches(ds.load_dataset())
        front_data = get_infinite_batches(ds.load_front_dataset())
        first_person = Variable(torch.FloatTensor(config.batch_size, config.nc, 128, 128))

        for epoch in range(config.nepoch):
            for i in range(199):
                # train discriminator
                self.discriminator.zero_grad()

                profile_image = profile_data.__next__()
                front_image = front_data.__next__().repeat(config.batch_size, 1, 1, 1)

                batch_size = profile_image.size(0)
                label_real.data.resize(batch_size, 1).fill_(1)
                label_fake.data.resize(batch_size, 1).fill_(0)
                noise.data.resize_(batch_size, config.nz, 1, 1).normal_(0, 1)

                profile_image_cropped = self.get_cropped_image(profile_image)
                front_image_cropped = self.get_cropped_image(front_image)

                real = Variable(front_image_cropped.cuda())
                profile = Variable(profile_image_cropped.cuda())

                style = self.styleencoder(profile)
                style = Variable(style.cuda())
                #train discriminator
                d_real = self.discriminator(real, style)
                fake = self.generator(noise, style)
                d_fake = self.discriminator(fake.detach(), style)  # not update generator

                d_loss = self.loss(d_real, label_real) + self.loss(d_fake, label_fake)  # real label
                d_loss.backward()
                self.optimizer_d.step()

                # train generator, styleencoder
                self.generator.zero_grad()
                self.styleencoder.zero_grad()
                front_style = self.styleencoder(real)
                d_fake = self.discriminator(fake, style)
                emb_real = self.get_embedding_from_image(front_image_cropped)
                emb_fake = self.get_embedding_from_image(fake.cpu())
                #s1_loss = torch.mean(torch.abs(style-front_style))
                #s2_loss = torch.mean(torch.abs(real-fake))
                #s3_loss = torch.mean(torch.abs(self.styleencoder(previous_f)-front_style)) + torch.mean(torch.abs(self.styleencoder(previous_p)-style))
                s4_loss = torch.mean(torch.abs(emb_real-emb_fake)).cuda()
                g_loss = self.loss(d_fake, label_real)  # trick the fake into being real
                g_s_loss = g_loss + s4_loss
                g_s_loss.backward()
                self.optimizer_g.step()
                self.optimizer_s.step()
                previous_f = real
                previous_p = profile
                if i==0:
                    first_person = fake.data
            print("epoch{:03d} d_real: {}, d_fake: {}".format(epoch, d_real.mean(), d_fake.mean()))
            vutils.save_image(fake.data, '{}/fake_result_epoch_{:03d}.png'.format(config.result_dir, epoch), normalize=True)
            vutils.save_image(real.data, '{}/real_result_epoch_{:03d}.png'.format(config.result_dir, epoch), normalize=True)
            vutils.save_image(first_person, '{}/first_person_result_epoch_{:03d}.png'.format(config.result_dir, epoch), normalize=True)
            if epoch == 49:
                torch.save(self.generator.state_dict(), 'generator_param_50.pkl')
                torch.save(self.discriminator.state_dict(), 'discriminator_param_50.pkl')
                torch.save(self.styleencoder.state_dict(), 'styleencoder_param_50.pkl')
        torch.save(self.generator.state_dict(), 'generator_param_100.pkl')
        torch.save(self.discriminator.state_dict(), 'discriminator_param_100.pkl')
        torch.save(self.styleencoder.state_dict(), 'styleencoder_param_100.pkl')

import torch.utils.data
from dataset import *

trainer = Trainer()
trainer.train()