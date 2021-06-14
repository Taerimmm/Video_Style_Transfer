import os
import glob
import time
import random
import numpy as np
import scipy.io as scio
import cv2
import kornia

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils
import torch.backends.cudnn as cudnn

from framework import Stylization


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

# RGB to Gray scale
def RGB2Gray(image):
    mean = image.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = image.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)

    image = (image*std+mean)

    gray = image[:,2:3,:,:]*0.299 + image[:,1:2,:,:]*0.587 + image[:,0:1,:,:]*0.114
    gray = gray.expand(image.size())

    gray = (gray-mean)/std
    return gray

#

##############################
##  Transfer
##############################

# RGB to Gray scale
input_frame = './inputs/frame/4644.jpg'

frame = transform_image(numpy2tensor(cv2.imread(input_frame)))
print(frame.shape)

gray = RGB2Gray(frame)
print(gray.shape)

frame_result = transfrom_back_image(gray)
frame_result = tensor2numpy(frame_result)

cv2.imwrite('./result_frames/result_gray.jpg', frame_result)

# Reconstruction image
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

style_img = './inputs/resize_plum_flower.jpg'
style = transform_image(numpy2tensor(cv2.imread(style_img)))


checkpoint_path = './Model/style_net-TIP-final.pth'

cuda = torch.cuda.is_available()
use_Global = True

from framework import Stylization

framework = Stylization(checkpoint_path, cuda, use_Global)
framework.prepare_style(cv2.imread(style_img))

framework.clean()

'''
content_video = './inputs/frame/*.jpg'  # Edit here.
frame_list = glob.glob(content_video)

frame_num = len(frame_list)
print(frame_num)

interval = 8    # input_batch
sample_sum = (frame_num - 1) // interval

for s in range(sample_sum):
    i = s * interval
    print('Add frame %d , %d frames in total' %(s + 1 , sample_sum + 1))
    input_frame = cv2.imread(frame_list[i])
    framework.add(input_frame)

print('Add frame %d , %d frames in total' %(sample_sum + 1, sample_sum + 1))
input_frame = cv2.imread(frame_list[-1])
framework.add(input_frame)
'''

framework.add(cv2.imread(input_frame))
framework.compute()

print(framework.model.Encoder)

F_content = framework.model.Encoder(frame.to(device))
F_style = framework.model.EncoderStyle(style.to(device))
print(F_content.shape)
# print(F_style.shape)

recon_content = framework.model.Decoder(F_content, framework.model.EncoderStyle(frame.to(device)))
recon_style = framework.model.Decoder(framework.model.Encoder(RGB2Gray(style.to(device))), F_style)

recon_content_frame_result = transfrom_back_image(recon_content)
recon_content_frame_result = tensor2numpy(recon_content_frame_result)

recon_style_frame_result = transfrom_back_image(recon_style)
recon_style_frame_result = tensor2numpy(recon_style_frame_result)

cv2.imwrite('./result_frames/result_recon_content.jpg', recon_content_frame_result)
cv2.imwrite('./result_frames/result_recon_style.jpg', recon_style_frame_result)

# Warp image
def warp(x, flo, padding_mode='border'):
    B, C, H, W = x.size()

    # mesh grid
    xx = torch.arange(0, W).view(1,-1).repeat(H,1)
    yy = torch.arange(0, H).view(-1,1).repeat(1,W)
    xx = xx.view(1,1,H,W).repeat(B,1,1,1)
    yy = yy.view(1,1,H,W).repeat(B,1,1,1)
    grid = torch.cat((xx,yy),1).float()
    grid = grid.to(x.device)
    vgrid = grid - flo

    # scale grid to [-1,1]
    vgrid[:,0,:,:] = 2.0 * vgrid[:,0,:,:] / max(W-1, 1) - 1.0
    vgrid[:,1,:,:] = 2.0 * vgrid[:,1,:,:] / max(H-1, 1) - 1.0
    vgrid = vgrid.permute(0,2,3,1) # flow-field of size (N x OH x OW x 2)
    output = F.grid_sample(x, vgrid, padding_mode=padding_mode)
    return output

flow_max = 20
gauss = kornia.filters.GaussianBlur2d((101,101), (50.5,50.5))
def smooth_flow(flow, H, W):
    flow = F.interpolate(flow, (H,W), mode='bilinear', align_corners=False)
    flow = F.tanh(flow) * flow_max
    flow = gauss(flow)
    return flow

B, C, H, W = style.to(device).shape
flow_scale = 8

Flow = torch.zeros([B, 2, H//flow_scale, W//flow_scale]).to(style.device)

Bounded_Flow = smooth_flow(Flow, H, W)

warpped_tmp_style = warp(style, Bounded_Flow)
print(warpped_tmp_style.shape)

warpped_frame_result = transfrom_back_image(warpped_tmp_style)
warpped_frame_result = tensor2numpy(warpped_frame_result)

cv2.imwrite('./result_frames/result_warpped.jpg', warpped_frame_result)


## Temporal_Loss image
from torch.autograd import Variable

class TemporalLoss(nn.Module):
    def __init__(self, data_sigma=True, data_w=True, noise_level=0.001,
                       motion_level=8, shift_level=10):
        super(TemporalLoss, self).__init__()
        self.MSE = torch.nn.MSELoss()

        self.data_sigma = data_sigma
        self.data_w = data_w
        self.noise_level = noise_level
        self.motion_level = motion_level
        self.shift_level = shift_level

        """
        Flow should have most values in the range of [-1, 1].
        For example, values x = -1, y = -1 is the left-top pixel of input,
        and values x = 1, y = 1 is the right-bottom pixel of input.
        Flow should be from pre_frame to cur_frame
        """

    def GaussianNoise(self, ins, mean=0, stddev=0.001):
        stddev = stddev + random.random() * stddev
        noise = Variable(ins.data.new(ins.size()).normal_(mean, stddev))

        if ins.is_cuda:
            noise = noise.cuda()
        return ins + noise

    def GenerateFakeFlow(self, height, width):
        '''
        height : img.shape[0]
        width  : img.shape[1]
        '''

        if self.motion_level > 0:
            flow = np.random.normal(0, scale=self.motion_level, size = [height//100, width//100, 2])
            flow = cv2.resize(flow, (width, height))
            flow[:,:,0] += random.randint(-self.shift_level, self.shift_level)
            flow[:,:,1] += random.randint(-self.shift_level, self.shift_level)
            flow = cv2.blur(flow, (100,100))
        else:
            flow = np.ones([width,height,2])
            flow[:,:,0] = random.randint(-self.shift_level, self.shift_level)
            flow[:,:,1] = random.randint(-self.shift_level, self.shift_level)

        return torch.from_numpy(flow.transpose((2,0,1))).float()

    def GenerateFakeData(self, first_frame):
        '''
        Input should be a (H,W,3) numpy of value range [0,1].
        '''

        if self.data_w:
            forward_flow = self.GenerateFakeFlow(first_frame.shape[2], first_frame.shape[3])
            if first_frame.is_cuda:
                forward_flow = forward_flow.cuda()
            print(forward_flow.shape)
            forward_flow = forward_flow.expand(first_frame.shape[0], 2, first_frame.shape[2], first_frame.shape[3])
            second_frame = warp(first_frame, forward_flow)
        else:
            second_frame = first_frame.clone()
            forward_flow = None

        if self.data_sigma:
            second_frame = self.GaussianNoise(second_frame, stddev=self.noise_level)

        return second_frame, forward_flow

    def forward(self, first_frame, forward_flow):
        if self.data_w:
            first_frame = warp(first_frame, forward_flow)

        return first_frame

TemporalLoss = TemporalLoss()

frame = transform_image(numpy2tensor(cv2.imread(input_frame)))
print(frame.shape)

# Second Frame
SecondFrame, ForwardFlow = TemporalLoss.GenerateFakeData(frame)
print(SecondFrame.shape)

second_frame_result = transfrom_back_image(SecondFrame)
second_frame_result = tensor2numpy(second_frame_result)

cv2.imwrite('./result_frames/result_second_frame.jpg', second_frame_result)

# Styled Second Frmae
StyledSecondFrame = framework.model.Decoder(framework.model.Encoder(SecondFrame.to(device)),framework.model.EncoderStyle(style.to(device)))

StyledSecondFrame_result = transfrom_back_image(StyledSecondFrame)
StyledSecondFrame_result = tensor2numpy(StyledSecondFrame_result)

cv2.imwrite('./result_frames/result_StyledSecondFrame.jpg', StyledSecondFrame_result)

# Style

StyledFirstFrame = framework.model.Decoder(F_content, F_style)

StyledFirstFrame_result = transfrom_back_image(StyledFirstFrame)
StyledFirstFrame_result = tensor2numpy(StyledFirstFrame_result)

cv2.imwrite('./result_frames/result_StyledFirstFrame.jpg', StyledFirstFrame_result)

FakeStyledSecondFrame_1 = TemporalLoss(StyledFirstFrame, ForwardFlow.to(device))

FakeStyledSecondFrame_result = transfrom_back_image(FakeStyledSecondFrame_1)
FakeStyledSecondFrame_result = tensor2numpy(FakeStyledSecondFrame_result)

cv2.imwrite('./result_frames/result_FakeStyledSecondFrame.jpg', FakeStyledSecondFrame_result)


# L1 - Loss (temporal loss)
temporalloss = torch.mean(torch.abs(StyledSecondFrame - FakeStyledSecondFrame_1))
print(temporalloss)