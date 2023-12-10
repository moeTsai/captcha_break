import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import to_tensor, to_pil_image
from PIL import Image

from captcha.image import ImageCaptcha
from tqdm import tqdm
import random
import numpy as np
from collections import OrderedDict

import string

from CaptchaDataset import CaptchaDataset
from Model import Model
from Utilss import train
from Utilss import valid



def decode(sequence):
    a = ''.join([characters[x] for x in sequence])
    s = ''.join([x for j, x in enumerate(a[:-1]) if x != characters[0] and x != a[j+1]])
    if len(s) == 0:
        return ''
    if a[-1] != characters[0] and s[-1] != a[-1]:
        s += a[-1]
    return s

def decode_target(sequence):
    return ''.join([characters[x] for x in sequence]).replace(' ', '')


characters = '-' + string.digits + string.ascii_uppercase
width, height, n_len, n_classes = 192, 64, 4, len(characters)#192 64
n_input_length = 12

# dataset = CaptchaDataset(characters, 1, width, height, n_input_length, n_len)



image_path = r'C:\Users\a0955\OneDrive\文件\GitHub\captcha_break\imgs\wmQI.png'
image = Image.open(image_path)
image = image.resize((128, 64), Image.BILINEAR)
    # 將圖片轉換為 RGB 模式
image_rgb = image.convert('RGB')

    # 將 RGB 圖片轉換為 PyTorch 張量
image_tensor = to_tensor(image_rgb)

print("image_shape = ", image_tensor.shape)

model = torch.load('ctc3.pth')
model = model.cuda()
model.eval()


output = model(image_tensor.unsqueeze(0).cuda())
output_argmax = output.detach().permute(1, 0, 2).argmax(dim=-1)
predict = decode(output_argmax[0])
print('pred:', predict)
image.show()