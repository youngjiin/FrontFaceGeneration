#attention 추가
import argparse
import time as t
from torch.utils.tensorboard import SummaryWriter
parser = argparse.ArgumentParser("cDCGAN")

parser.add_argument('--dataset_dir', type=str, default='/home/ubuntu/dataset')
parser.add_argument('--result_dir', type=str, default='./test_output')
parser.add_argument('--batch_size', type=int, default=10)
parser.add_argument('--nepoch', type=int, default=100)
parser.add_argument('--nz', type=int, default=100) # number of noise dimension
parser.add_argument('--nc', type=int, default=3) # number of result channel
parser.add_argument('--nfeature', type=int, default=512) # num of embedding

config, _ = parser.parse_known_args()

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
import torchvision.transforms as transforms
from torch.autograd import Variable
from math import sqrt
from facenet_pytorch import InceptionResnetV1# If required, create a face detection pipeline using MTCNN:
import cv2
from model import Generator, StyleEncoder

def get_infinite_batches(data_loader):
    while True:
        for i, (images, _) in enumerate(data_loader):
            yield images

class Test:
    def __init__(self):
        self.generator = Generator()
        self.styleencoder = StyleEncoder()
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval()
        self.generator.cuda()
        self.styleencoder.cuda()
        self.resnet.cuda()

    def test(self):
        noise = Variable(torch.FloatTensor(config.batch_size, config.nz, 1, 1).cuda())
        ds = Dataset(config)
        profile_data = get_infinite_batches(ds.load_dataset())
        front_data = get_infinite_batches(ds.load_front_dataset())

        self.generator.load_state_dict(
            torch.load("./paper_save_model/5/5_generator_param_70.pth", map_location="cuda:0"), strict=True)
        self.styleencoder.load_state_dict(
            torch.load("./paper_save_model/5/5_styleencoder_param_70.pth", map_location="cuda:0"), strict=True)
        self.generator.eval()
        self.styleencoder.eval()

        for i in range(200):
            profile_image = profile_data.__next__().repeat(3, 1, 1, 1)[:10, :, :, :]
            real_image = front_data.__next__()
            profile_image = Variable(profile_image.cuda())
            #style = self.resnet(profile_image)
            style = self.styleencoder(profile_image)
            style = Variable(style.cuda())

            fake = self.generator(noise, style)

            vutils.save_image(fake.data, '{}/fake_{:03d}.png'.format(config.result_dir, i), normalize=True)
            vutils.save_image(profile_image.data, '{}/profile_{:03d}.png'.format(config.result_dir, i), normalize=True)
            vutils.save_image(real_image.data, '{}/real_{:03d}.png'.format(config.result_dir, i), normalize=True)

import torch.utils.data
from dataset2 import *

test_ = Test()
test_.test()