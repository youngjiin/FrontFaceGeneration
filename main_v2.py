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

class AdaptiveInstanceNorm(nn.Module):
    def __init__(self, in_channel, style_dim):
        super().__init__()

        self.norm = nn.InstanceNorm2d(in_channel)
        self.style = EqualLinear(style_dim, in_channel * 2) # 512 2048

        self.style.linear.bias.data[:in_channel] = 1
        self.style.linear.bias.data[in_channel:] = 0

    def forward(self, input, style):
        style = self.style(style).unsqueeze(2).unsqueeze(3)
        gamma, beta = style.chunk(2, 1)

        out = self.norm(input)
        out = gamma * out + beta

        return out

class EqualLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        linear = nn.Linear(in_dim, out_dim)
        linear.weight.data.normal_()
        linear.bias.data.zero_()

        self.linear = equal_lr(linear)

    def forward(self, input):
        return self.linear(input)

def equal_lr(module, name='weight'):
    EqualLR.apply(module, name)

    return module

class EqualLR:
    def __init__(self, name):
        self.name = name

    def compute_weight(self, module):
        weight = getattr(module, self.name + '_orig')
        fan_in = weight.data.size(1) * weight.data[0][0].numel()

        return weight * sqrt(2 / fan_in)

    @staticmethod
    def apply(module, name):
        fn = EqualLR(name)

        weight = getattr(module, name)
        del module._parameters[name]
        module.register_parameter(name + '_orig', nn.Parameter(weight.data))
        module.register_forward_pre_hook(fn)

        return fn

    def __call__(self, module, input):
        weight = self.compute_weight(module)
        setattr(module, self.name, weight)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.s1 = nn.Sequential(
            nn.ConvTranspose2d(config.nz + config.nfeature, 1024, 4, 1, 0, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(True),
        )
        self.s2 = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
        )
        self.s3 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
        )
        self.s4 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
        )
        self.s5 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )
        self.s6 = nn.Sequential(
            nn.ConvTranspose2d(64, config.nc, 4, 2, 1, bias=False),
            nn.Tanh(),
        )
        self.adain1 = AdaptiveInstanceNorm(1024, 512)
        self.adain2 = AdaptiveInstanceNorm(512, 512)
        self.adain3 = AdaptiveInstanceNorm(256, 512)
        self.adain4 = AdaptiveInstanceNorm(128, 512)
        self.adain5 = AdaptiveInstanceNorm(64, 512)

    def forward(self, x, attr): #attr ( 1, 512)
        attr = attr.view(-1, config.nfeature, 1, 1) #(10, 512, 1, 1)
        x = torch.cat([x, attr], 1) #(10 , 612, 1,1)

        x = self.s1(x)
        x = self.adain1(x, attr.squeeze(3).squeeze(2)) #x:(10, 1024, 4, 4) attr:(10,512,1,1)
        x = self.s2(x)
        x = self.adain2(x, attr.squeeze(3).squeeze(2))
        x = self.s3(x)
        x = self.adain3(x, attr.squeeze(3).squeeze(2))
        x = self.s4(x)
        x = self.adain4(x, attr.squeeze(3).squeeze(2))
        x = self.s5(x)
        x = self.adain5(x, attr.squeeze(3).squeeze(2))
        return self.s6(x)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.feature_input = nn.Linear(config.nfeature, 128 * 128)
        self.main = nn.Sequential(
            nn.Conv2d(config.nc + 1, 64, 4, 2, 1, bias=False),
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
            nn.Conv2d(1024, 1, 4, 1, 0, bias=False),
        )

    def forward(self, x, attr):
        attr = self.feature_input(attr).view(-1, 1, 128, 128)
        x = torch.cat([x, attr], 1)
        return self.main(x).view(-1, 1)

def get_infinite_batches(data_loader):
    while True:
        for i, (images, _) in enumerate(data_loader):
            yield images

class Trainer:
    def __init__(self):
        self.generator = Generator()
        self.discriminator = Discriminator()
        self.mtcnn = MTCNN(image_size=128, margin=20)
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval()
        self.loss = nn.MSELoss()
        self.optimizer_g = optim.Adam(self.generator.parameters(), lr=config.lr, betas=betas)
        self.optimizer_d = optim.Adam(self.discriminator.parameters(), lr=config.lr, betas=betas)

        self.generator.cuda()
        self.discriminator.cuda()
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

    def train(self, ds):
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

                attr = self.get_embedding_from_image(profile_image_cropped)

                attr = Variable(attr.cuda())
                real = Variable(front_image_cropped.cuda())

                d_real = self.discriminator(real, attr)


                fake = self.generator(noise, attr)
                #fake = torch.clamp(fake, 0, 1)
                d_fake = self.discriminator(fake.detach(), attr)  # not update generator

                d_loss = self.loss(d_real, label_real) + self.loss(d_fake, label_fake)  # real label
                d_loss.backward()
                self.optimizer_d.step()

                # train generator
                self.generator.zero_grad()
                d_fake = self.discriminator(fake, attr)
                g_loss = self.loss(d_fake, label_real)  # trick the fake into being real
                g_loss.backward()
                self.optimizer_g.step()
                if i==0:
                    first_person = fake.data
            print("epoch{:03d} d_real: {}, d_fake: {}".format(epoch, d_real.mean(), d_fake.mean()))
            vutils.save_image(fake.data, '{}/fake_result_epoch_{:03d}.png'.format(config.result_dir, epoch), normalize=True)
            vutils.save_image(real.data, '{}/real_result_epoch_{:03d}.png'.format(config.result_dir, epoch), normalize=True)
            vutils.save_image(first_person, '{}/first_person_result_epoch_{:03d}.png'.format(config.result_dir, epoch), normalize=True)
        torch.save(self.generator.state_dict(), 'generator_param.pkl')
        torch.save(self.discriminator.state_dict(), 'discriminator_param.pkl')

import torch.utils.data
from dataset import *

ds = Dataset(config)

trainer = Trainer()
trainer.train(ds)