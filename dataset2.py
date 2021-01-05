import torch
import os
import torchvision
import torchvision.transforms as transforms
import numpy as np


class Dataset:
    def __init__(self, flags):
        self.flags = flags
        self.data_path = os.path.join(self.flags.dataset_dir, 'cfp/profile')
        self.front_data_path = os.path.join(self.flags.dataset_dir, 'cfp/frontal')

    def load_dataset(self):

        train_dataset = torchvision.datasets.ImageFolder(
            root=self.data_path,
            transform = transforms.Compose([
                transforms.Scale(128),
                transforms.CenterCrop(128),
                transforms.ToTensor(),
            ])
            #transform=torchvision.transforms.ToTensor(),
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=4,
            #num_workers=4,
            shuffle=False,
            #pin_memory=True
        )

        return train_loader

    def load_front_dataset(self):

        train_dataset = torchvision.datasets.ImageFolder(
            root=self.front_data_path,
            transform=transforms.Compose([
                transforms.Scale(128),
                transforms.CenterCrop(128),
                transforms.ToTensor(),
            ])
            # transform=torchvision.transforms.ToTensor(),
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.flags.batch_size,
            #num_workers=4,
            shuffle=False,
            #pin_memory=True
        )

        return train_loader
