import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.utils as vutils
from torch.autograd import Variable

import random
import numpy as np
import cv2
from collections import namedtuple

''' Optical flow warping function '''

def warp