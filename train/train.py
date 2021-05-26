from __future__ import print_function

import os
import time
import glob
import random
import argparse

import numpy as np
import cv2
import torch

import torch.optim as optim
import torch.nn as nn
import torch.utils as vutils
from torch.autograd import Variable, grad
from torch.utils.tensorboard import SummaryWriter

from dataset import get_loader
from loss_networks
from style_networks import TransformerNet
from other_networks


parser = argparse.ArgumentParser()

# GPU settings
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--gpu', action='store', type=int, default=0, help='gpu device')
parser.add_argument('--manualSeed', type=int, help='manual seed')

# Basic training settings
parser.add_argument('--batchSize', type=int, default=4, help='input batch size')
parser.add_argument('--epoches', type=int, default=2, help='number of epoches to train for')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--log', type=int, default=1000, help='number of iteration to save checkpoints and figures')
parser.add_argument('--continue_training', action='store_true', help='continue training')
parser.add_argument('--load_epoch', type=int, default=0, help='epoch of loaded checkpoint')
parser.add_argument('--start_iteration', type=int, default=0, help='can start at any iteration')
parser.add_argument('--num_workers', type=int, default=4, help='num of workers for data loader')

# Dataset settings
parser.add_argument('--content_data', default='./data/content/', help='path to content images')
parser.add_argument('--style_data', default='./data/style/', help='path to style images')

# Save path settings
parser.add_argument('--outf', default='result', help='path to output images and model checkpoints')
parser.add_argument('--valf', default='val', help='path to validation images')
parser.add_argument('--log_dir', default='log', help='path to event file of tensorboard')

# Data augmentation settings
parser.add_argument('--loadSize', type=int, default=512, help='the height / width of the input image to network')
parser.add_argument('--fineSize', type=int, default=256, help='the height / width of the input image to network')
parser.add_argument('--flip', type=int, default=1, help='1 for flipping image randomly, 0 for not')

# Model architecture settings (proposed model : --dynamic_filter --both_sty_con)
parser.add_argument('--dynamic_filter', action='store_true', help='use dynamic filter in the decoder')
parser.add_argument('--both_sty_con', action='store_true', help='use both style and content dynamic filter in the decoder')
parser.add_argument('--train_only_decoder', action='store_true', help='both content and style encoder are fixed pre-trained VGG')

# Loss settings (proposed model : --style_content_loss --recon_loss --tv_loss --temporal_loss --relax_style)
parser.add_argument('--temporal_loss')








opt = parser.parse_args()
print(opt)

# ==== Prepare GPU and Seed ====
if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed :", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)
    import torch.backends.cudnn as cudnn
    cudnn.benchmark = True


# ==== Models ====
style_net = TransformerNet(dynamic_filter=opt.dynamic_filter,
                           both_sty_con=opt.both_sty_con,
                           train_only_decoder=opt.train_only_decoder,
                           )