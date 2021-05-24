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

        img = torch.from_numpy(img.transpose((2,0,1))).float()
        mean = img.new_tensor([0.485, 0.456, 0.406]).view(-1,1,1)  # img.new_tensor <---- ??? | check
        std = img.new_tensor([0.229, 0.224, 0.225]).view(-1,1,1)
        img = img.div_(255.0)
        img = (img - mean) / std

        return img

    def Process(self, input):
        """ Given a numpy """
        return torch.



def get_loader(batch_size, loadSize=288, fineSize=256, flip=True, content_path="./data/content/", style_path="./data/style/", num_workers=16, use_mpi=False, use_video=False):

    # if use_mpi and use_video:

    # if use_mpi:

    # elif use_video:

    # else:
    dataset = FrameDataset()



