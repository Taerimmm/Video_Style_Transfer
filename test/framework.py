from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils
import torch.backends.cudnn as cudnn

import os
import time
import glob
import copy
import math
import numbers
import random

import numpy as np
import cv2

##############################
##  Tools
##############################

# Image to tensor tools
def numpy2tensor(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(img.transpose((2,0,1))).float()

def transform_image(img):
    mean = img.new_tensor([0.485, 0.456, 0.406]).view(-1,1,1)
    std = img.new_tensor([0.229, 0.224, 0.225]).view(-1,1,1)
    img = img.div_(255.0)
    img = (img - mean) / std
    return img.unsqueeze(0)

# Tensor to image tools
def tensor2numpy(img):
    img = img.data.cpu()
    img = img.numpy().transpose((1,2,0))
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

def transfrom_back_image(img):
    mean = img.new_tensor([0.485, 0.456, 0.406]).view(-1,1,1)
    std = img.new_tensor([0.229, 0.224, 0.225]).view(-1,1,1)
    img = img * std + mean
    img = img.clamp(0, 1)[0,:,:,:] * 255 # torch.clamp : min 혹은 max의 범주에 해당하도록 값을 변경하는 것
    return img


##############################
##  Framework
##############################

class Stylization():
    def __init__(self, checkpoint, cuda=False, use_Global=True):

        # ====== Prepare GPU and Seed ======

        if cuda:
            cudnn.benchmark = True
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # ====== Framework ======

        if use_Global:
            from style_network_global import TransformerNet
        else:
            from style_network_frame import TransformerNet

        self.model = TransformerNet().to(self.device)
        self.model.load_state_dict(torch.load(checkpoint, map_location=lambda storage, loc: storage))

        for param in self.model.parameters():
            param.requires_grad = False

    # ====== Sequence-Level Global Feature Sharing ======

    def add(self, patch):
        with torch.no_grad():
            patch = numpy2tensor(patch).to(self.device)
            self.model.add(transform_image(patch))
        torch.cuda.empty_cache()  # 사용하지 않으면서 캐시된 메모리들을 해제

    def compute(self):
        with torch.no_grad():
            self.model.compute()
        torch.cuda.empty_cache()

    def clean(self):
        self.model.clean()
        torch.cuda.empty_cache()

    def clean_(self):
        self.model.clean_()

    # ====== Style Transfer ======

    def prepare_style(self, style):
        with torch.no_grad():
            style = numpy2tensor(style).to(self.device)
            style = transform_image(style)
            self.model.generate_style_features(style)
        torch.cuda.empty_cache()

    def transfer(self, frame):
        with torch.no_grad():
            # Transform images into tensors
            frame = numpy2tensor(frame).to(self.device)
            frame = transform_image(frame)

            # Stylization
            frame = self.model(frame)

            frame_result = transfrom_back_image(frame)
            frame_result = tensor2numpy(frame_result)

        return frame_result
