import random
import torch
import torch.utils.data
from PIL import Image
from glob import glob
import numpy as np
import torchvision.transforms as transforms
import os

class DataProcess(torch.utils.data.Dataset):
    def __init__(self, img_root, input_mask_root, ref_root, train=True):
        super(DataProcess, self).__init__()
        self.img_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        # mask should not normalize, is just have 0 or 1
        self.mask_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])
        self.img_root = img_root
        self.input_mask_root = input_mask_root
        self.ref_root = ref_root
        self.Train = False
        if train:
            self.img_paths = os.listdir(img_root)
            #self.img_paths = sorted(glob('{:s}/*'.format(img_root), recursive=True))
            #self.st_paths = sorted(glob('{:s}/*'.format(st_root), recursive=True))
            self.mask_paths = os.listdir(input_mask_root)
            #self.mask_paths = sorted(glob('{:s}/*'.format(input_mask_root), recursive=True))

            self.ref_paths = os.listdir(img_root)
            self.Train = True

        self.N_mask = len(self.mask_paths)
        print(self.N_mask)

    def __getitem__(self, index):
        img_ = Image.open(self.img_root+'/'+self.img_paths[index])
        #st_img = Image.open(self.st_paths[index])
        ref_img = Image.open(self.ref_root+'/'+self.ref_paths[index])
        mask_img = Image.open(self.input_mask_root+'/'+self.mask_paths[random.randint(0, self.N_mask - 1)])

        img = self.img_transform(img_.convert('RGB'))
        #gray = self.mask_transform(img_.convert('L'))
        #st_img = self.img_transform(st_img.convert('RGB'))
        ref_img = self.img_transform(ref_img.convert('RGB'))
        mask_img = self.mask_transform(mask_img.convert('RGB'))

        return img, mask_img, ref_img

    def __len__(self):
        return len(self.img_paths)
