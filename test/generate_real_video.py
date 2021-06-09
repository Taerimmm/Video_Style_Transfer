import os
import glob
import time
import random
import numpy as np
import scipy.io as scio
import cv2

import torch

from framework import Stylization


##############################
##  Parameters
##############################

# Target style
style_img = './inputs/plum_flower.jpg'  # Edit here.

# Target content video
# Use glob.lob() to search for all the frames
# Add sort them by sort()
content_video = './inputs/frame/*.png'  # Edit here.

# Path of the checkpoint
checkpoint_path = './Model/style_net-TIP-final.pth'

# Device settings, use cuda if available
cuda = torch.cuda.is_available()

# The proposed Sequence-Level Global Feature Sharing
use_Global = True

# Saving settings
save_video = True
fps = 24

# Where to save the results
result_frames_path = './result_frames/'
result_videos_path = './result_videos/'


##############################
##  Tools
##############################

if not os.path.exists(result_frames_path):
    os.mkdir(result_frames_path)
if not os.path.exists(result_videos_path):
    os.mkdir(result_videos_path)

def read_img(img_path):
    return cv2.imread(img_path)


class ReshapeTool():
    def __init__(self):
        self.record_H = 0
        self.record_W = 0

    def process(self, img):
        H, W, C = img.shape

        if self.record_H == 0 and self.record_W == 0:
            new_H = H + 128
            if new_H % 64 != 0:
                new_H += 64 - new_H % 64

            new_W = W + 128
            if new_W % 64 != 0:
                new_W += 64 - new_W % 64

            self.record_H = new_H
            self.record_W = new_W

        new_img = cv2.copyMakeBorder(img, 64, self.record_H - 64 - H,
                                          64, self.record_W - 64 - W, cv2.BORDER_REFLECT)

        return new_img


##############################
##  Preparation
##############################

# Read style image
if not os.path.exists(style_img):
    exit('Style image %s not exists' %(style_img))
style = cv2.imread(style_img)

# Build model
framework = Stylization(checkpoint_path, cuda, use_Global)
framework.prepare_style(style)

# Read content frames
frame_list = glob.glob(content_video)

# Name for this testing
style_name = (style_img.split('/')[-1]).split('.')[0]
video_name = (content_video.split('/')[-2])
name = 'ReReVST-' + style_name + '-' +video_name
if not use_Global:
    name = name + '-no-global'

# Mkdir corresponding folders
if not os.path.exists('{}/{}'.format(result_frames_path, name)):
    os.mkdir('{}/{}'.format(result_frames_path, name))

# Build tools
reshape = ReshapeTool()


##############################
##  Inference
##############################

frame_num = len(frame_list)

# Prepare for proposed Sequence-Level Global Feature Sharing

if use_Global:
    print('Preparationbs for Sequence-Level Global Feature Sharing')

    framework.clean()

    interval = 8    # input_batch
    sample_sum = (frame_num - 1) // interval

    for s in range(sample_sum):
        i = s * interval
        print('Add frame %d , %d frames in total' %(s + 1 , sample_sum + 1))
        input_frame = read_img(frame_list[i])
        framework.add(input_frame)

    print('Add frame %d , %d frames in total' %(sample_sum + 1, sample_sum + 1))
    input_frame = read_img(frame_list[-1])
    framework.add(input_frame)







