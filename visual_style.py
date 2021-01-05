#attention 추가
import argparse
import time as t
from torch.utils.tensorboard import SummaryWriter
parser = argparse.ArgumentParser("cDCGAN")

parser.add_argument('--dataset_dir', type=str, default='/home/ubuntu/dataset/crop_frontface/FACE/')
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
import glob
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.cbook import get_sample_data

def imscatter(x, y, image, ax=None, zoom=1):
    if ax is None:
        ax = plt.gca()
    try:
        img_tensor = np.load(image)
        img_tensor += 1
        img_tensor -= img_tensor.min()
        img_tensor /= (img_tensor.max() - img_tensor.min())
        image = 255 * img_tensor
        image = image.astype(np.uint8)
    except TypeError:
        # Likely already an array...
        pass
    im = OffsetImage(image.transpose(1,2,0), zoom=zoom)
    x, y = np.atleast_1d(x, y)
    artists = []
    for x0, y0 in zip(x, y):
        ab = AnnotationBbox(im, (x0, y0), xycoords='data', frameon=False)
        artists.append(ax.add_artist(ab))
    ax.update_datalim(np.column_stack([x, y]))
    ax.autoscale()
    return artists

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
        '''
        self.generator.load_state_dict(
            torch.load("./paper_save_model/5/5_generator_param_70.pth", map_location="cuda:0"), strict=True)
        self.styleencoder.load_state_dict(
            torch.load("./paper_save_model/5/5_styleencoder_param_70.pth", map_location="cuda:0"), strict=True)
        '''
        checkpoint = torch.load("./paper_save_model/9/9_model_param_31.pth")
        self.styleencoder.load_state_dict(checkpoint['StyleEncoder'])

        self.styleencoder.eval()

        paths = glob.glob(os.path.join(config.dataset_dir, '*.npy'))

        pca = PCA(n_components=2)
        xys = []
        for i, path in enumerate(paths):
            x = np.load(path)
            #print(path)
            reference_image = torch.from_numpy(x).float().cuda()
            style = self.styleencoder(reference_image.reshape(1, 3, 128, 128))
            style2 = self.resnet(reference_image.reshape(1,3,128,128))
            style2 *= 2
            style = torch.cat((style,style2), 1)
            xys.append(np.array(style.squeeze(0).cpu().detach()))

        xys = pca.fit_transform(xys)
        xs = xys[:, 0]
        ys = xys[:, 1]
        plt.rc('axes', unicode_minus=False)
        plt.rc('font', family='NanumGothic')
        plt.rcParams.update({'font.size': 25})
        plt.figure(figsize=(10, 10))
        colors = ["#7fc97f", "#beaed4", "#fdc086", "#ffff99", "#386cb0", "#f0027f", "#666666"]

        for i in range(len(xs)):
                plt.scatter(xs[i], ys[i], s=80, c=colors[1])
                #plt.text(xs[i]+0.1, ys[i], paths[i].split('/')[-1], fontsize=10)
                imscatter(xs[i]+0.1, ys[i]+0.1, paths[i], zoom=0.22)

        plt.savefig('fig.png')
        plt.show()

import torch.utils.data
from dataset2 import *

test_ = Test()
test_.test()