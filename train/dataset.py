import os
import glob
import random
import numpy as np
import cv2

import scipy.io as scio    # MPIDataset
import pickle              # VideoDataset
import zipfile             # VideoDataset, MPIDataset

import torch
import torch.utils.data as data
from torch.autograd import Variable
import torch.nn.functional as F

from torchvision import datasets, transforms
import torchvision.utils as vutils


# class VideoDataset(data.Dataset):

# class MPIDataset(data.Dataset):

class FrameDataset(data.Dataset):

    ''' The Dataloader for the training '''

    def __init__(self, loadSize=288, fineSize=256, flip=True, content_path="data/content", style_path="data/style"):
        super(FrameDataset, self).__init__()

        # Data Lists
        self.content_img_list = glob.glob(content_path + "/*.jpg")
        self.style_img_list = glob.glob(style_path + "/*.jpg")
        self.style_img_list_len = len(self.style_img_list)

        # Parameters
        self.loadSize = loadSize
        self.fineSize = fineSize
        self.flip = flip

    def ProcessImg(self, img, size=None, x1=None, y1=None, flip_rand=None):
        """
        Given an image with channel [BRG] which values [0,255]
        The output values [-1,1]
        """
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        h,w,c = img.shape

        if (size != None):
            img = cv2.resize(img, (size,size))
            img = img[x1:x1+self.fineSize, y1:y1+self.fineSize, :]

        if (self.flip == 1):
            if flip_rand <= 0.25:
                img = cv2.flip(img, 1)  # 1은 좌우 반전
            elif flip_rand <= 0.5:
                img = cv2.flip(img, 0)  # 0은 상하 반전
            elif flip_rand <= 0.75:
                img = cv2.flip(img, -1) # -1은 좌우 & 상하 반전

        img = torch.from_numpy(img.transpose((2,0,1))).float()     # HWC to CHW
        mean = img.new_tensor([0.485, 0.456, 0.406]).view(-1,1,1)  # img.new_tensor <---- ??? | check
        std = img.new_tensor([0.229, 0.224, 0.225]).view(-1,1,1)
        img = img.div_(255.0)
        img = (img - mean) / std

        return img

    def Process(self, input):
        """ Given a numpy """
        return torch.from_numpy(input.transpose((2,0,1))).float()  # HWC to CHW

    def RandomBlur(self, img):
        img = img.data.numpy().transpose((1,2,0))  # CHW to HWC
        H,W,C = img.shape
        img = cv2.resize(img, (H+random.randint(-5,5), W+random.randint(-5,5)))
        img = cv2.resize(img, (H,W))
        return self.Process(img)

    def __getitem__(self, index):   # for loop 돌리기 위한 magic method
        # Read Data

        # Read content image
        first_img = self.content_img_list[index]
        first_frame = cv2.imread(first_img)
        # Read style image
        style_img_path = self.style_img_list[random.randint(0, self.style_img_list_len)]
        style_img = cv2.imread(style_img_path)

        Sequence = {}

        # Process Content Image
        x1 = random.randint(0, self.loadSize - self.fineSize)
        y1 = random.randint(0, self.loadSize - self.fineSize)
        flip_rand = random.random()

        first_frame = self.ProcessImg(first_frame, self.loadSize, x1, y1, flip_rand)
        Sequence['Content'] = first_frame

        # Process Style Image
        H,W,C = style_img.shape
        loadSize = max(H, W, self.loadSize)

        x1 = random.randint(0, loadSize - self.fineSize)
        y1 = random.randint(0, loadSize - self.fineSize)
        flip_rand = random.random()

        Sequence['Style'] = self.ProcessImg(style_img, loadSize, x1, y1, flip_rand)

        return Sequence

    def __len__(self):
        return len(self.content_img_list)


def get_loader(batch_size, loadSize=288, fineSize=256, flip=True, content_path="./data/content/", style_path="./data/style/", num_workers=16, use_mpi=False, use_video=False):

    # if use_mpi and use_video:

    # if use_mpi:

    # elif use_video:

    # else:
    dataset = FrameDataset(loadSize=loadSize, fineSize=fineSize, flip=flip, content_path=content_path, style_path=style_path)

    loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    return loader



