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
import torch.nn.functional as F
import torch.optim as optim
import torchvision.utils as vutils
import torchvision.transforms as transforms
from torch.autograd import Variable
from math import sqrt
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1# If required, create a face detection pipeline using MTCNN:

import cv2

def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()


def squash(s, dim=-1):
    '''
    "Squashing" non-linearity that shrunks short vectors to almost zero length and long vectors to a length slightly below 1
    Eq. (1): v_j = ||s_j||^2 / (1 + ||s_j||^2) * s_j / ||s_j||

    Args:
        s: 	Vector before activation
        dim:	Dimension along which to calculate the norm

    Returns:
        Squashed vector
    '''
    squared_norm = torch.sum(s ** 2, dim=dim, keepdim=True)
    return squared_norm / (1 + squared_norm) * s / (torch.sqrt(squared_norm) + 1e-8)

class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()
        self.fc_layer = nn.Linear(512, 100)
        self.W = nn.Parameter(0.01 * torch.randn(1, 100, 8, 8, 100))
        self.main = nn.Sequential(
            nn.ConvTranspose2d(100, 256, 4, 2, 1, bias=False),
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

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self, x, style): #attr ( 1, 512)
        #style = style.view(-1, config.nfeature) #(10, 512, 1, 1)
        style = self.fc_layer(style) #(10, 100)
        style = squash(style) #(10,100)
        style = style.view(-1, 1, 1, 100, 1)
        #style = style.view(-1, 100, 1, 1)#torch.reshape(10, 100, 1, 1)
        u = style.repeat(1, 1, 8, 1, 1)
        u_hat = torch.matmul(self.W, u) # (10, 100, 16, 16)
        #u_hat = torch.cat([x, u_hat], 1)  # (10, 200, 16,16)
        #x = torch.cat([x, style], 1) #(10 , 612, 1,1)
        return self.main(u_hat.squeeze(-1))

'''
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
            nn.LeakyReLU(0.2, inplace=True), # ( 10, 1024, 4, 4)
            nn.MaxPool2d(4)
            #nn.Conv2d(1024, 512, 4, 1, 0, bias=False)
        )
        self.fc_layer = nn.Linear(1024, 512)

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
        s = self.fc_layer(s)
        return s
'''

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
        )
        self.last = nn.Sequential(
            nn.Conv2d(1024, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self, x, attr):
        attr = self.feature_input(attr).view(-1, 1, 128, 128)
        x = torch.cat([x, attr], 1)# (10, 4, 128, 128)
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
        self.generator.weight_init(mean=0.0, std=0.02)
        self.discriminator.weight_init(mean=0.0, std=0.02)
        #self.styleencoder = StyleEncoder()
        self.mtcnn = MTCNN(image_size=128, margin=20)
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval()
        self.loss = nn.BCELoss()
        self.recon_loss = nn.MSELoss()
        self.sigmoid = nn.Sigmoid()

        # Set the logger
        self.dir_name = t.strftime('~%Y%m%d~%H%M%S', t.localtime(t.time()))
        self.log_train = './log/' + self.dir_name + '/train'
        self.writer = SummaryWriter(self.log_train)

        self.optimizer_g = optim.Adam(self.generator.parameters(), lr=config.lr, betas=betas)
        self.optimizer_d = optim.Adam(self.discriminator.parameters(), lr=config.lr, betas=betas)
        #self.optimizer_s = optim.Adam(self.styleencoder.parameters(), lr=config.lr, betas=betas)
        self.generator.cuda()
        self.discriminator.cuda()
        #self.styleencoder.cuda()
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
        embeddings = self.sigmoid(embeddings)
        return embeddings

    def totalvariance_loss(self, img):
        """
        Compute total variation loss.
        Inputs:
        - img: PyTorch Variable of shape (1, 3, H, W) holding an input image.
        - tv_weight: Scalar giving the weight w_t to use for the TV loss.
        Returns:
        - loss: PyTorch Variable holding a scalar giving the total variation loss
          for img weighted by tv_weight.
        """
        w_variance = torch.sum(torch.pow(img[:, :, :, :-1] - img[:, :, :, 1:], 2))
        h_variance = torch.sum(torch.pow(img[:, :, :-1, :] - img[:, :, 1:, :], 2))
        loss = h_variance + w_variance
        return loss

    def train(self):
        noise = Variable(torch.FloatTensor(config.batch_size, config.nz, 1, 1).cuda())
        label_real = Variable(torch.FloatTensor(config.batch_size, 1).fill_(1).cuda())
        label_fake = Variable(torch.FloatTensor(config.batch_size, 1).fill_(0).cuda())
        ds = Dataset(config)
        profile_data = get_infinite_batches(ds.load_dataset())
        front_data = get_infinite_batches(ds.load_front_dataset())
        first_person = Variable(torch.FloatTensor(config.batch_size, config.nc, 128, 128))
        previous_f = Variable(torch.FloatTensor(config.batch_size, config.nc, 128, 128)).cuda()
        previous_p = Variable(torch.FloatTensor(config.batch_size, config.nc, 128, 128)).cuda()

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

                style = self.get_embedding_from_image(profile_image_cropped)#self.styleencoder(profile)
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
                #self.styleencoder.zero_grad()
                real_front_style = self.get_embedding_from_image(front_image_cropped)#self.styleencoder(real)
                fake_front_style = self.get_embedding_from_image(fake.cpu())#self.styleencoder(fake)
                d_fake = self.discriminator(fake, style)
                #emb_real = self.get_embedding_from_image(front_image_cropped)
                #emb_fake = self.get_embedding_from_image(fake.cpu())
                #pixel_loss = self.loss(real,fake)#torch.mean(torch.abs(torch.sub(real, fake)))
                real_front_style = Variable(real_front_style.cuda())
                fake_front_style =Variable(fake_front_style.cuda())
                #identity_loss = self.loss(real_front_style,fake_front_style)#torch.mean(torch.abs(torch.sub(real_front_style, fake_front_style)))
                #identity_loss_2 = self.loss(real_front_style, style)#torch.mean(torch.abs(torch.sub(real_front_style, style)))
                reconstruct = self.generator(noise, fake_front_style.cuda())
                #d_recon = self.discriminator(reconstruct, fake_front_style)
                reconstruct_loss = self.recon_loss(reconstruct, real)
                '''
                previous_style = self.styleencoder(previous_f)
                if epoch ==0 and i ==0 :
                    identity_max_loss = 0
                else :
                    identity_max_loss = self.loss(real_front_style, previous_style)#torch.mean(torch.abs(torch.sub(real_front_style, previous_style)))
                previous_f = real
                '''
                #tv_loss = self.totalvariance_loss(fake)
                g_loss = self.loss(d_fake, label_real)  # trick the fake into being real
                g_s_loss = g_loss + reconstruct_loss #- 0.01*identity_max_loss #+ 0.00000001*tv_loss
                #+ 0.01*identity_loss_2+ 0.001*pixel_loss - 0.01*identity_max_loss # s4_loss
                g_s_loss.backward()
                self.optimizer_g.step()
                #self.optimizer_s.step()
                if i==0:
                    first_person = fake.data
                if i%10 == 0:
                    # Testing
                    x_fake = fake+1
                    x_real = real+1
                    x_recon = reconstruct+1
                    x_fake = x_fake-x_fake.min()
                    x_real = x_real-x_real.min()
                    x_recon = x_recon-x_recon.min()
                    x_fake = x_fake / (x_fake.max()-x_fake.min())
                    x_real = x_real / (x_real.max()-x_real.min())
                    x_recon = x_recon / (x_recon.max()-x_recon.min())

                    self.writer.add_scalar('d_loss', d_loss, ((epoch)*190) + i)
                    self.writer.add_scalar('g_loss', g_loss, ((epoch)*190) + i)
                    #self.writer.add_scalar('pixel_loss', pixel_loss, ((epoch)*190) + i)
                    #self.writer.add_scalar('identity_loss', identity_loss, ((epoch)*190) + i)
                    #self.writer.add_scalar('identity_loss2', identity_loss_2, ((epoch)*190) + i)
                    #self.writer.add_scalar('identity_max_loss', identity_max_loss, ((epoch)*190) + i)
                    #self.writer.add_scalar('tv_loss', tv_loss, ((epoch)*190) + i)
                    self.writer.add_scalar('reconstruct_loss', reconstruct_loss, ((epoch) * 190) + i)
                    self.writer.add_scalar('total_loss', g_s_loss, ((epoch)*190) + i)
                    self.writer.add_images('fake_image', x_fake, ((epoch)*190) + i)
                    self.writer.add_images('real_image', x_real, ((epoch)*190) + i)
                    self.writer.add_images('recon_image', x_recon, ((epoch)*190) + i)
                    print("d_loss:{}, g_loss:{}, recon_loss:{}, g_s_loss:{}".format(d_loss, g_loss, reconstruct_loss, g_s_loss))
            print("epoch{:03d} d_real: {}, d_fake: {}".format(epoch, d_real.mean(), d_fake.mean()))
            vutils.save_image(fake.data, '{}/fake_result_epoch_{:03d}.png'.format(config.result_dir, epoch), normalize=True)
            vutils.save_image(real.data, '{}/real_result_epoch_{:03d}.png'.format(config.result_dir, epoch), normalize=True)
            vutils.save_image(first_person, '{}/first_person_result_epoch_{:03d}.png'.format(config.result_dir, epoch), normalize=True)
            torch.save(self.generator.state_dict(), 'save_model/my_generator_param_%d.pth' % epoch)
            torch.save(self.discriminator.state_dict(), 'save_model/my_discriminator_param_%d.pth' % epoch)
            #torch.save(self.styleencoder.state_dict(), 'save_model/my_styleencoder_param_%d.pth' % epoch)
        self.writer.close()

import torch.utils.data
from dataset import *

trainer = Trainer()
trainer.train()