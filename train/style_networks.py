import torch
import torch.nn as nn
from torch.nn import init  # Initializer (초기화 설정) ....?
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.models as models
import torchvision.utils as vutils

import kornia
from collections import namedtuple


##############################
##  Tools
##############################

mean_std = namedtuple("mean_std", ['mean', 'std'])
vgg_outputs = namedtuple("VggOutputs", ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1'])
vgg_outputs_super = namedtuple("VggOutputs", ['map', 'relu1_1', 'relu2_1', 'relu3_1', 'relu4_1'])












##############################
##  Layers and Blocks
##############################

class InstanceNorm(nn.Module):
    def __init__(self, epsilon=1e-8):
        super(InstanceNorm, self).__init__()
        self.epsilon = epsilon

    def forward(self, x):
        x = x - torch.mean(x, (2,3), True)
        tmp = torch.mul(x, x)
        tmp = torch.rsqrt(torch.mean(tmp, (2,3), True) + self.epsilon)
        return x * tmp


class FilterPredictor(nn.Module):
    def __init__(self, vgg_channel=512, inner_channel=32):
        super(FilterPredictor, self).__init__()
        self.down_sample = nn.Sequential(nn.Conv2d(vgg_channel, inner_channel, kernel_size=3, padding=1))
        self.inner_channel = inner_channel
        self.FC = nn.Linear(inner_channel*2, inner_channel*inner_channel)

    def forward(self, content, style):
        content = self.down_sample(content)
        style = self.down_sample(style)

        content = torch.mean(content.view(content.size(0), content.size(1), -1), dim=2) # Global Average Pooling
        style = torch.mean(style.view(style.size(0), style.size(1), -1), dim=2)

        filter = self.FC(torch.cat([content, style], 1))
        filter = filter.view(-1, self.inner_channel, self.inner_channel).unsqueeze(3)
        return filter


class KernelFilter(nn.Module):
    def __init__(self, vgg_channel=512, inner_channel=32):
        super(KernelFilter, self).__init__()
        self.down_sample = nn.Sequential(
            nn.Conv2d(vgg_channel, inner_channel, kernel_size=3, padding=1),
        )
        self.upsample = nn.Sequential(
            nn.Conv2d(inner_channel, vgg_channel, kernel_size=3, padding=1),
        )

        self.F1 = FilterPredictor(vgg_channel, inner_channel)
        self.F2 = FilterPredictor(vgg_channel, inner_channel)

        self.relu = nn.LeakyReLU(0.2)

    def apply_filter(self, input_, filter_):
        """
        input_  : [B, inC, H, W]
        filter_ : [B, inC, outC, 1]
        """

        B = input_.shape[0]
        input_chunk = torch.chunk(input_, B, dim=0)     # Tensor를 쪼개는 함수. (B개로 dim 차원으로 쪼갤지 설정)
        filter_chunk = torch.chunk(filter_, B, dim=0)

        results = []

        for input, filter_ in zip(input_chunk, filter_chunk):
            input = F.conv2d(input, filter_.permute(1,2,0,3), groups=1)  # 1x1 convolution
            # transpose()는 딱 두 개의 차원을 맞교환할 수 있다. 그러나 permute()는 모든 차원들을 맞교환할 수 있다
            results.append(input)

        return torch.cat(results, 0)

    def forward(self, content, style):
        content_ = self.down_sample(content)

        content_ = self.apply_filter(content_, self.F1(content, style))
        content_ = self.relu(content_)

        content_ = self.apply_filter(content_, self.F2(content, style))

        return content + self.upsample(content_)




class ResidualBlock(nn.Module):
    def __init__(self, input_channel, output_channel, upsample=True):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(input_channel, output_channel, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(output_channel, output_channel, kernel_size=3, padding=1)
        self.conv_shortcut = nn.Conv2d(input_channel, output_channel, kernel_size=1, bias=False)
        self.relu = nn.LeakyReLU(0.2)
        self.norm = InstanceNorm()
        self.upsample = upsample

    def forward(self, x):
        if self.upsample:
            x = F.interpolate(x, mode='nearest', scale_factor=2)

        x_s = self.conv_shortcut(x)

        x = self.conv1(x)
        x = self.relu(x)
        x = self.norm(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.norm(x)

        return x_s + x


##############################
##  Networks
##############################



class Decoder(nn.Module):
    def __init__(self, dynamic_filter=True, both_sty_con=True):
        super(Decoder, self).__init__()

        self.slice4 = ResidualBlock(512, 256)
        self.slice3 = ResidualBlock(256, 128)
        self.slice2 = ResidualBlock(128, 64)
        self.slice1 = nn.Conv2d(64, 3, kernel_size=3, padding=1)
        self.norm = InstanceNorm()

        self.dynamic_filter = dynamic_filter
        if dynamic_filter:
            if both_sty_con:
                self.Filter1 = KernelFilter()
                self.Filter2 = KernelFilter()
                self.Filter3 = KernelFilter()
            else:
                self.Fi


class TransformerNet(nn.Module):
    def __init__(self, dynamic_filter=True, both_sty_con=True, train_only_decoder=False,
                       style_content_loss=True, recon_loss=True, relax_style=True):
        super(TransformerNet, self).__init__()

        # Sub-models
        self.Decoder = Dec