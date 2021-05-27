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
from loss_networks import warp, TemporalLoss
from style_networks import TransformerNet
from other_networks import define_D, init_weights, GANLoss


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
parser.add_argument('--temporal_loss', action='store_true', help='use temporal loss')
parser.add_argument('--style_content_loss', action='store_true', help='use style loss and content loss')
parser.add_argument('--recon_loss', action='store_true', help='use reconstruction loss')
parser.add_argument('--tv_loss', action='store_true', help='use tv loss')
parser.add_argument('--relax_style', action='store_true', help='use the proposed Relaxed Style Loss')
parser.add_argument('--old_style_loss', action='store_true', help='use the classical style loss')

parser.add_argument('--adaversarial_loss', action='store_true', help='use LSGAN (not included in the paper)')
'''
Adding LSGAN can make the stylization effect better.
The color can be more balanced, and textures can be more vivid
However, temporal consistency will be worse.
So I didn't include the LSGAN loss in the paper.
If you are interested, try adding --adaversarial_loss
'''

parser.add_argument()


# Specific settings for Compound Regularization (proposed model : --data_sigma --data_w)
parser.add_argument('--data_sigma', action='store_true', help='use noise in temporal loss')
parser.add_argument('--data_w', action='store_true', help='use warp in temporal loss')
parser.add_argument('--data_noise_level', type=float, default=0.001, help='noise level in temporal loss')
parser.add_argument('--data_motion_level', type=float, default=8, help='motion level in temporal loss')
parser.add_argument('--data_shift_level', type=float, default=10, help='shift level in temporal loss')

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

# Build model
style_net = TransformerNet(dynamic_filter=opt.dynamic_filter,
                           both_sty_con=opt.both_sty_con,
                           train_only_decoder=opt.train_only_decoder,
                           style_content_loss=opt.style_content_loss,
                           recon_loss=opt.recon_loss,
                           relax_style=opt.relax_style)

# Load checkpoints
def LoadPretrained(checkpoint):
    try:
        style_net.load_state_dict(torch.load(checkpoint[0], map_location=lambda storage, loc:storage))
    except:
        try:
            tmp_style_net = TransformerNet(dynamic_filter=False,
                                           both_sty_con=False,
                                           train_only_decoder=False,
                                           style_content_loss=False,
                                           recon_loss=False,
                                           relax_style=False)
            tmp_style_net.load_state_dict(torch.load(checkpoint[0], map_location=lambda storage, loc:storage))

            style_net.Encoder = tmp_style_net.Encoder
            style_net.EncoderStyle = tmp_style_net.EncoderStyle
            style_net.Decoder.slice4 = tmp_style_net.Decoder.slice4
            style_net.Decoder.slice3 = tmp_style_net.Decoder.slice3
            style_net.Decoder.slice2 = tmp_style_net.Decoder.slice2
            style_net.Decoder.slice1 = tmp_style_net.Decoder.slice1
        except:
            style_net.Decoder.conv_kernel = nn.Conv2d(512, 512, kernel_size=1)
            style_net.load_state_dict(torch.load(checkpoint[0], map_location=lambda storage, loc:storage))
            del style_net.Decoder.conv_kernel

if opt.continue_training:
    checkpoint = glob.glob('%s/style_net-epoch-%d*.pth' % (opt.outf, opt.load_epoch))
    if len(checkpoint) == 0:
        exit('Cannot find checkpoint.')
    if len(checkpoint) > 1:
        exit('Too many checkpoints.')
else:
    checkpoint = ['style_net-epoch-0.pth']

LoadPretrained(checkpoint)

# Set states
style_net.train()
print(style_net)

# Set devices
os.environ['CUDA_VISIBLE_DEVICES'] = "%d"%(opt.gpu)
device = torch.device("cuda" if opt.cuda else "cpu")
style_net = style_net.to(device)


# ==== Optimizer =====

optimizer = optim.Adam(filter(lambda p:p.requires_grad, style_net.parameters()), lr=opt.lr)


# ==== Useful functions ====

# Tensorboard
writer = SummaryWriter(log_dir=opt.log_dir)

# Savepath
if not os.path.exists(opt.outf):
    os.mkdir(opt.outf)

def save_figure(img, name, is_image=True):
    img = img.data.clone()
    if is_image:
        mean = img.new_tensor([0.485, 0.456, 0.406]).view(-1,1,1)
        std = img.new_tensor([0.229, 0.224, 0.225]).view(-1,1,1)
        img = (img * std + mean)
        vutils.save_image(img, '%s/%s.png' % (opt.outf, name))
    else:
        vutils.save_image(img, '%s/%s.png' % (opt.outf, name), normalize=True)

def adjust_learning_rate(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * 0.5
    return optimizer

def transform_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(img.transpose((2, 0, 1))).float()

    mean = img.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = img.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)

    img = img.div_(255.0)
    img = (img - mean) / std
    return img.unsqueeze(0)

# Save example results
class Validation():
    def __init__(self):
        content_img_list = glob.glob(opt.valf + '/content/*.jpg')
        self.content_image = []
        for i in range(6):
            img = cv2.imread(content_img_list[i])
            img = cv2.resize(img, (256,256))
            img = transform_image(img)
            self.content_image.append(img)

        style_img_list = glob.glob(opt.valf + '/style/*.jpg')
        self.style_image = []
        for i in range(6):
            img = cv2.imread(style_img_list[i])
            img = cv2.resize(img, (256,256))
            img = transform_image(img)
            self.style_image.append(img)

    def SaveResults(self, epoch):
        with torch.no_grad():
            for i in range(6):
                content = self.content_image[i].to(device)
                style = self.style_image[i].to(device)
                result = style_net.validation(content, style)
                result = torch.cat([result, content, style], dim=2)
                save_figure(result, 'Epoch[%d]-validation-%d'%(epoch, i))

Validation = Validation()
Validation.SaveResults(0)

# Name for this training
TimeName = time.asctime(time.localtime(time.time())) + " RandomID %d"%(random.randint(0,100))


# ==== Loss functions ====

def TV(x):
    b, c, h_x, w_x = x.shape
    h_tv = torch.mean(torch.abs(x[:,:,1:,:] - x[:,:,:h_x-1,:]))
    w_tv = torch.mean(torch.abs(x[:,:,:,1:] - x[:,:,:,:w_x-1]))
    return h_tv + w_tv

if opt.temporal_loss:
    TemporalLoss = TemporalLoss(data_sigma=opt.data_sigma,
                                data_w=opt.data_w,
                                noise_level=opt.data_noise_level,
                                motion_level=opt.data_motion_level,
                                shift_level=opt.data_shift_level)

if opt.adaversarial_loss:
    netD = define_D(3).to(device)
    init_weights(netD)
    netD.train()
    print(netD)

    if opt.continue_training:
        checkpoint = '%s/netD-epoch-%d.pth' % (opt.outf, opt.load_epoch)
        if os.path.exists(checkpoint):
            netD.load_state_dict(torch.load(checkpoint, map_location=lambda storage, loc:storage))

    criterionGAN = GANLoss('lsgan').to(device)
    optimizerD = optim.Adam(filter(lambda p:p.requires_grad, netD.parameters()), lr=1e-4, betas=(0.5, 0.9))


# ==== Dataloader ====

loader = get_loader(opt.batchSize, loadSize=opt.loadSize, fineSize=opt.fineSize, flip=opt.flip,
                    content_path=opt.content_data, style_path=opt.style_data,
                    num_workers=opt.num_workers, use_mpi=False, use_video=False)

print('Data Load Success.')
iteration_sum = len(loader)


# ==== Training ====

print('Training Start.')

min_total_loss = np.inf
cur_total_loss = 0.























