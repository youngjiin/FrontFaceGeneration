import torch
import os
import torchvision
import torchvision.transforms as transforms
import numpy as np


class Dataset:
    def __init__(self, flags):
        self.flags = flags
        self.data_path = os.path.join(self.flags.dataset_dir, 'face')
        self.front_data_path = os.path.join(self.flags.dataset_dir, 'frontface')
        self.crop_data_path = os.path.join(self.flags.dataset_dir, 'crop_face')
        self.crop_front_data_path = os.path.join(self.flags.dataset_dir, 'crop_frontface')

    def npy_loader(self, path):
        sample = torch.from_numpy(np.load(path))
        return sample

    def load_dataset(self):

        train_dataset = torchvision.datasets.ImageFolder(
            root=self.data_path,
            transform = transforms.Compose([
                transforms.ToTensor(),
            ])
            #transform=torchvision.transforms.ToTensor(),
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.flags.batch_size,
            #num_workers=4,
            shuffle=False,
            #pin_memory=True
        )

        return train_loader

    def load_npy_dataset(self):
        dataset = torchvision.datasets.DatasetFolder(
            root=self.crop_data_path,
            loader=self.npy_loader,
            extensions='.npy'
        )
        train_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.flags.batch_size,
            #num_workers=4,
            shuffle=False,
            #pin_memory=True
        )
        return train_loader

    def load_front_dataset(self):

        train_dataset = torchvision.datasets.ImageFolder(
            root=self.front_data_path,
            transform=transforms.Compose([
                transforms.ToTensor(),
            ])
            # transform=torchvision.transforms.ToTensor(),
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=1,
            #num_workers=4,
            shuffle=False,
            #pin_memory=True
        )

        return train_loader

    def load_front_npy_dataset(self):
        dataset = torchvision.datasets.DatasetFolder(
            root=self.crop_front_data_path,
            loader=self.npy_loader,
            extensions='.npy'
        )
        train_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            #num_workers=4,
            shuffle=False,
            #pin_memory=True
        )
        return train_loader
