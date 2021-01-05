#attention 추가
import argparse
import time as t
from torch.utils.tensorboard import SummaryWriter
parser = argparse.ArgumentParser("cDCGAN")

parser.add_argument('--dataset_dir', type=str, default='/home/ubuntu/dataset')
parser.add_argument('--result_dir', type=str, default='./cDCGAN_result')
parser.add_argument('--condition_file', type=str, default='./list_attr_cDCGAN.txt')

parser.add_argument('--batch_size', type=int, default=10)
parser.add_argument('--nepoch', type=int, default=32)
parser.add_argument('--nz', type=int, default=100) # number of noise dimension
parser.add_argument('--nc', type=int, default=3) # number of result channel
parser.add_argument('--nfeature', type=int, default=512) # num of embedding
parser.add_argument('--lr', type=float, default=0.0001)
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
from efficientnet_pytorch import EfficientNet
from model import Generator, StyleEncoder, Discriminator

def get_infinite_batches(data_loader):
    while True:
        for i, (images, _) in enumerate(data_loader):
            yield images

class Trainer:
    def __init__(self):
        self.generator = Generator()
        self.discriminator = Discriminator()
        self.styleencoder = StyleEncoder()
        self.mtcnn = MTCNN(image_size=128, margin=20, device='cuda:0')
        self.resnet = InceptionResnetV1(pretrained='vggface2', device='cuda:0').eval()
        self.efficient = EfficientNet.from_pretrained('efficientnet-b4').eval()
        #self.fc = nn.Linear(1000, 512)
        self.loss = nn.MSELoss()
        self.optimizer_g = optim.Adam(self.generator.parameters(), lr=0.001, betas=betas)
        self.optimizer_d = optim.Adam(self.discriminator.parameters(), lr=0.0001, betas=betas)
        #self.optimizer_fc = optim.SGD(self.fc.parameters(), lr=0.001, momentum=0.9)
        self.optimizer_s = optim.Adam(self.styleencoder.parameters(), lr=config.lr, betas=betas)
        self.generator.cuda()
        self.discriminator.cuda()
        self.styleencoder.cuda()
        #self.fc.cuda()
        self.efficient.cuda()
        self.resnet.cuda()
        self.mtcnn.cuda()
        self.loss.cuda()

        # Set the logger
        self.dir_name = t.strftime('~%Y%m%d~%H%M%S', t.localtime(t.time()))
        self.log_train = './paper_log/' + self.dir_name + '/train'
        self.writer = SummaryWriter(self.log_train)

    def get_cropped_image(self, image):
        images = [transforms.ToPILImage()(image_) for image_ in image]
        images = self.mtcnn(images)
        images = torch.cat(images).view(10, 3, 128, 128)
        return Variable(images.cuda())

    def save_image(self, epoch):
        ds = Dataset(config)
        profile_data = get_infinite_batches(ds.load_npy_dataset())
        front_data = get_infinite_batches(ds.load_front_npy_dataset())
        noise = Variable(torch.FloatTensor(config.batch_size, config.nz, 1, 1).cuda())
        for i in range(200):
            profile_image = profile_data.__next__()
            profile_image = Variable(profile_image.cuda())
            #style = self.resnet(profile_image)
            style = self.styleencoder(profile_image)
            style = Variable(style.cuda())
            # train discriminator
            fake = self.generator(style)

            vutils.save_image(fake.data, '{}/paper/8/{}_{:03d}_fake.png'.format(config.result_dir, epoch, i), normalize=True)
            if epoch is 30:
                real_image = front_data.__next__().repeat(10, 1, 1, 1)
                vutils.save_image(profile_image.data, '{}/paper/profile_{:03d}.png'.format(config.result_dir, i), normalize=True)
                vutils.save_image(real_image.data, '{}/paper/real_{:03d}.png'.format(config.result_dir, i), normalize=True)

    def train(self):
        noise = Variable(torch.FloatTensor(config.batch_size, config.nz, 1, 1).cuda())
        label_real = Variable(torch.FloatTensor(config.batch_size, 1).fill_(1).cuda())
        label_fake = Variable(torch.FloatTensor(config.batch_size, 1).fill_(0).cuda())
        ds = Dataset(config)
        profile_data = get_infinite_batches(ds.load_npy_dataset())
        front_data = get_infinite_batches(ds.load_front_npy_dataset())

        batch_size = config.batch_size
        label_real.data.resize(batch_size, 1).fill_(1)
        label_fake.data.resize(batch_size, 1).fill_(0)
        noise.data.resize_(batch_size, config.nz, 1, 1).normal_(0, 1)

        for epoch in range(config.nepoch):
            for i in range(200):
                # train discriminator
                self.discriminator.zero_grad()
                #self.fc.zero_grad()
                profile_image = profile_data.__next__()
                front_image = front_data.__next__().repeat(config.batch_size, 1, 1, 1)
                #profile_image = profile_data.__next__().repeat(3, 1, 1, 1)[:10, :, : ,:]
                #front_image = front_data.__next__()

                profile = Variable(profile_image.cuda())
                real = Variable(front_image.cuda())
                #profile = self.get_cropped_image(profile_image)
                #real = self.get_cropped_image(front_image)

                style = self.styleencoder(profile)
                style2 = self.resnet(profile)
                style = torch.cat((style,style2),1)
                #style = self.efficient(profile)
                #style = self.fc(style)
                style = Variable(style.cuda())
                #train discriminator
                d_real = self.discriminator(real, style)
                fake = self.generator(style)
                d_fake = self.discriminator(fake.detach(), style)  # not update generator

                d_loss = self.loss(d_real, label_real) + self.loss(d_fake, label_fake)  # real label
                d_loss.backward()
                self.optimizer_d.step()
                #self.optimizer_fc.step()

                # train generator, styleencoder
                #self.fc.zero_grad()
                self.generator.zero_grad()
                self.styleencoder.zero_grad()

                style = self.styleencoder(profile)
                style2 = self.resnet(profile)
                style = torch.cat((style,style2) , 1)
                #style = self.fc(style)
                style = Variable(style.cuda())

                fake = self.generator(style)
                d_fake = self.discriminator(fake, style)
                #emb_real = self.get_embedding_from_image(front_image_cropped)
                #emb_fake = self.get_embedding_from_image(fake.cpu())
                # recon_loss
                '''
                fake_front_style = self.styleencoder(fake)
                reconstruct = self.generator(noise, fake_front_style)
                reconstruct_loss = self.loss(reconstruct, real)
                '''
                g_loss = self.loss(d_fake, label_real)  # trick the fake into being real
                g_s_loss = g_loss  #+ reconstruct_loss# s4_loss
                g_s_loss.backward()
                self.optimizer_g.step()
                #self.optimizer_fc.step()
                self.optimizer_s.step()
                if epoch is config.nepoch-1:
                    vutils.save_image(fake.data, '{}/paper/s_1/{}_{:03d}_fake.png'.format(config.result_dir, epoch, i), normalize=True)
                    #vutils.save_image(profile_image.data, '{}/paper/2_profile_{:03d}.png'.format(config.result_dir, i), normalize=True)
                    #vutils.save_image(real.data, '{}/paper/2_real_epoch_{:03d}.png'.format(config.result_dir, i), normalize=True)

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

                    mse = torch.mean((x_fake*255 - x_real*255) **2)
                    psnr = 20 * torch.log10(255.0 / torch.sqrt(mse))

                    self.writer.add_scalar('d_loss', d_loss, ((epoch) * 190) + i)
                    self.writer.add_scalar('g_loss', g_loss, ((epoch) * 190) + i)
                    #.writer.add_scalar('recon_loss', reconstruct_loss, ((epoch) * 190) + i)
                    self.writer.add_scalar('total_g_loss', g_s_loss, ((epoch) * 190) + i)
                    self.writer.add_scalar('psnr', psnr, ((epoch) * 190) + i)
                    self.writer.add_images('fake_image', x_fake, ((epoch) * 190) + i)
                    self.writer.add_images('real_image', x_real, ((epoch) * 190) + i)
                    #self.writer.add_images('input_image', profile, ((epoch) * 190) + i)
                    #self.writer.add_images('recon_image', x_recon, ((epoch) * 190) + i)
                    print("d_loss:{}, g_loss:{},  g_s_loss:{}".format(d_loss, g_loss, g_s_loss))
            print("epoch{:03d} d_real: {}, d_fake: {}".format(epoch, d_real.mean(), d_fake.mean()))
            #vutils.save_image(fake.data, '{}/paper/5/fake_result_epoch_{:03d}.png'.format(config.result_dir, epoch), normalize=True)
            #vutils.save_image(real.data, '{}/paper/5/real_result_epoch_{:03d}.png'.format(config.result_dir, epoch), normalize=True)

            if epoch >= 20:
                #self.save_image(epoch)
                #'StyleEncoder': self.styleencoder.state_dict()
                #'fc':self.fc.state_dict()
                torch.save({
                    'Generator': self.generator.state_dict(),
                    #'fc':self.fc.state_dict()
                    'StyleEncoder': self.styleencoder.state_dict()
                }, 'paper_save_model/s_1/s1_model_param_%d.pth' % epoch)
                #torch.save(self.fc.state_dict(), 'paper_save_model/7/7_fc_param_%d.pth' % epoch)

        self.writer.close()

import torch.utils.data
from dataset import *

trainer = Trainer()
trainer.train()