from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import glob
import os
import numpy as np
import torchvision.utils as vutils
import torchvision.transforms as transforms
from torchvision.utils import save_image
import torchvision
import torch
import io
# If required, create a face detection pipeline using MTCNN:
mtcnn = MTCNN(image_size=128, margin=20)

profile_img_dir = "/home/ubuntu/dataset/face/FACE/"
front_img_dir = "/home/ubuntu/dataset/frontface/FACE/"
profile_data_path = os.path.join(profile_img_dir, '*')
front_data_path = os.path.join(front_img_dir, '*')
profile_files = glob.glob(profile_data_path)
front_files = glob.glob(front_data_path)
f1_dir = "/home/ubuntu/dataset/crop_face/FACE/"
f2_dir = "/home/ubuntu/dataset/crop_frontface/FACE/"

for f1 in profile_files:
    profile_img = Image.open(f1)
    path = os.path.join(f1_dir, f1.split("/")[-1].split(".")[0])
    img_cropped = mtcnn(profile_img)
    np.save(path, img_cropped)

for f2 in front_files:
    front_img = Image.open(f2)
    path = os.path.join(f2_dir, f2.split("/")[-1].split(".")[0])
    img_cropped = mtcnn(front_img)
    np.save(path, img_cropped)

# Get cropped and prewhitened image tensor
# Calculate embedding (unsqueeze to add batch dimension)
