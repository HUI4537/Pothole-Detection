import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
from torchvision.transforms import functional as F
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw
import os
import numpy as np

if torch.cuda.is_available():
    print("Ye")
else:
    print("No")