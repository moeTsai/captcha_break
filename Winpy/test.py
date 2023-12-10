from torchvision.transforms.functional import to_tensor
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import to_tensor, to_pil_image

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



if __name__ == '__main__':

    # 讀取 PNG 文件
    # 讀取 PNG 文件
    image_path = r'C:\Users\a0955\OneDrive\文件\GitHub\captcha_break\imgs\wmQI.png'
    image = Image.open(image_path)

    # 將圖片轉換為 RGB 模式
    image_rgb = image.convert('RGB')

    # 將 RGB 圖片轉換為 PyTorch 張量
    image_tensor = to_tensor(image_rgb)

    

    characters = '-' + string.digits + string.ascii_uppercase# 讀取 PNG 文件


    width, height, n_len, n_classes = 192, 64, 4, len(characters)  # 192 64
    n_input_length = 12
    print(characters, width, height, n_len, n_classes)

    dataset = CaptchaDataset(characters, 1, width, height, n_input_length, n_len)

    to_tensor_image = to_tensor(image)
    
    image_shape = to_tensor_image.shape
    print("image_shape = ", image_shape)

    to_pil_image(to_tensor_image).show()

    batch_size = 70
    # train_set = CaptchaDataset(characters, 1000 * batch_size, width, height, n_input_length, n_len)
    valid_set = CaptchaDataset(characters, 100 * batch_size, width, height, n_input_length, n_len)
    #　train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=2)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, num_workers=2)

    model = Model(n_classes, input_shape=(3, height, width))
    inputs = torch.zeros((32, 3, height, width))
    outputs = model(inputs)
    print(outputs.shape)

    model = Model(n_classes, input_shape=(3, height, width))
    model = model.cuda()

    # 轉換為 PyTorch 張量
    image_tensor = to_tensor(image)

    # 將預測的輸入形狀轉換為模型期望的形狀
    # 注意: 這裡的形狀應該符合你模型的預期輸入形狀
    image_tensor = image_tensor.unsqueeze(0)  # 添加 batch 維度

    # 使用模型進行預測
    model.eval()
    with torch.no_grad():
        output = model(image_tensor.cuda())  # 如果你的模型在 GPU 上，記得將張量移動到 GPU 上

    # 解碼預測結果
    output_argmax = output.detach().permute(1, 0, 2).argmax(dim=-1)
    predicted_text = decode(output_argmax[0], characters)

    print(f"Predicted Label: {predicted_text}")

    # 將圖片轉換回 PIL 格式
    image_pil = to_pil_image(image_tensor.squeeze(0))

    # 顯示圖片（可選）
    image_pil.show()