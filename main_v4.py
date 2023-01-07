#attention 추가
import argparse
import time as t
from torch.utils.tensorboard import SummaryWriter
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
from facenet_pytorch import InceptionResnetV1, MTCNN# If required, create a face detection pipeline using MTCNN:
import cv2
import numpy as np

class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()
        #self.fc = nn.Linear(512, 100)
        #self.W = nn.Parameter(0.01 * torch.randn(1, 100, 8, 8, 100))
        self.main = nn.Sequential(
            nn.ConvTranspose2d(config.nz + config.nfeature, 1024, 4, 1, 0, bias=False),
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
            nn.ConvTranspose2d(64, config.nc, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, x, style): #attr ( 1, 512)
        '''
        style = squash(style)
        style = self.fc(style)
        style = style.view(-1, 1, 1, 100, 1)
        u = style.repeat(1, 1, 8, 1, 1)
        u_hat = torch.matmul(self.W, u)
        '''
        style = style.view(-1, config.nfeature, 1, 1) #(10, 512, 1, 1)
        x = torch.cat([x, style], 1) #(10 , 612, 1,1)

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
        #attr= torch.softmax(s,1)
        #attr= attr * s
        #
        return s

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
        #self.resnet.cuda()
        #self.mtcnn.cuda()
        self.loss.cuda()

        # Set the logger
        self.dir_name = t.strftime('~%Y%m%d~%H%M%S', t.localtime(t.time()))
        self.log_train = './paper_log/' + self.dir_name + '/train'
        self.writer = SummaryWriter(self.log_train)

    def get_cropped_image(self, image):
        images = []
        for i in range(image.size(0)):
            images.append(self.mtcnn(torchvision.transforms.ToPILImage()(image[i])))
        images = torch.cat(images).view(10, 3, 128, 128)
        return Variable(images.cuda())
        '''
        images = [transforms.ToPILImage()(image_) for image_ in image]
        images = self.mtcnn(images)
        images = torch.cat(images).view(10, 3, 128,128)
        return Variable(images.cuda())
        '''
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

        for epoch in range(config.nepoch):
            for i in range(199):
                # train discriminator
                self.discriminator.zero_grad()
                profile_image = profile_data.__next__()
                front_image = front_data.__next__().repeat(config.batch_size, 1, 1, 1)

                batch_size = config.batch_size
                label_real.data.resize(batch_size, 1).fill_(1)
                label_fake.data.resize(batch_size, 1).fill_(0)
                noise.data.resize_(batch_size, config.nz, 1, 1).normal_(0, 1)

                profile = self.get_cropped_image(profile_image)
                real = self.get_cropped_image(front_image)

                #profile = Variable(profile_image.cuda())
                #real = Variable(front_image.cuda())

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
                #fake = self.generator(noise, style)
                d_fake = self.discriminator(fake, style)
                #emb_real = self.get_embedding_from_image(front_image_cropped)
                #emb_fake = self.get_embedding_from_image(fake.cpu())
                #s1_loss = torch.mean(torch.abs(style-front_style))
                #s2_loss = torch.mean(torch.abs(real-fake))
                #s3_loss = torch.mean(torch.abs(self.styleencoder(previous_f)-front_style)) + torch.mean(torch.abs(self.styleencoder(previous_p)-style))
                #s4_loss = torch.mean(torch.abs(emb_real-emb_fake)).cuda()
                # recon_loss
                '''
                fake_front_style = self.styleencoder(fake)
                reconstruct = self.generator(noise, fake_front_style)
                reconstruct_loss = self.loss(reconstruct, real)
                '''
                g_loss = self.loss(d_fake, label_real)  # trick the fake into being real
                g_s_loss = g_loss  # s4_loss
                g_s_loss.backward()
                self.optimizer_g.step()
                self.optimizer_s.step()
                if i % 10 == 0:
                    # Testing
                    x_fake = fake + 1
                    x_real = real + 1
                    #x_recon = reconstruct + 1
                    x_fake = x_fake - x_fake.min()
                    x_real = x_real - x_real.min()
                    #x_recon = x_recon - x_recon.min()
                    x_fake = x_fake / (x_fake.max() - x_fake.min())
                    x_real = x_real / (x_real.max() - x_real.min())
                    #x_recon = x_recon / (x_recon.max() - x_recon.min())

                    mse = torch.mean((x_fake - x_real) **2)
                    psnr = 20 * torch.log10(255.0 / torch.sqrt(mse))

                    self.writer.add_scalar('d_loss', d_loss, ((epoch) * 190) + i)
                    self.writer.add_scalar('g_loss', g_loss, ((epoch) * 190) + i)
                    #self.writer.add_scalar('recon_loss', reconstruct_loss, ((epoch) * 190) + i)
                    self.writer.add_scalar('total_g_loss', g_s_loss, ((epoch) * 190) + i)
                    self.writer.add_scalar('psnr', psnr, ((epoch) * 190) + i)
                    self.writer.add_images('fake_image', x_fake, ((epoch) * 190) + i)
                    self.writer.add_images('real_image', x_real, ((epoch) * 190) + i)
                    #self.writer.add_images('recon_image', x_recon, ((epoch) * 190) + i)
                    print("d_loss:{}, g_loss:{},  g_s_loss:{}".format(d_loss, g_loss, g_s_loss))
            print("epoch{:03d} d_real: {}, d_fake: {}".format(epoch, d_real.mean(), d_fake.mean()))
            vutils.save_image(fake.data, '{}/fake_result_epoch_{:03d}.png'.format(config.result_dir, epoch), normalize=True)
            vutils.save_image(real.data, '{}/real_result_epoch_{:03d}.png'.format(config.result_dir, epoch), normalize=True)
            if epoch >= 40:
                torch.save(self.generator.state_dict(), 'paper_save_model/1_generator_param_%d.pth' % epoch)
                torch.save(self.discriminator.state_dict(), 'paper_save_model/1_discriminator_param_%d.pth' % epoch)
                torch.save(self.styleencoder.state_dict(), 'paper_save_model/1_styleencoder_param_%d.pth' % epoch)
        self.writer.close()

import torch.utils.data
from dataset import *

trainer = Trainer()
trainer.train()