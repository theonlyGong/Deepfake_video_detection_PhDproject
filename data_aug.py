# -*- coding: utf-8 -*-
# @Time : 2024/3/28 16:10
# @Author : Liangyu Gong
# @FileName: data_aug.py
# @Software: PyCharm

import torchvision.transforms as transform
import numpy as np
import albumentations as alb
from albumentations.pytorch import ToTensorV2
import cv2
import random
from PIL import Image


class ALBU_AUG:
    def __init__(self, base_transform):
        self.transform = base_transform

    def __call__(self, x):
        if isinstance(x, Image.Image):
            x = np.asarray(x)
        return self.transform(image=x)['image']

class OneOfTrans:
    """random select one of from the input transform list"""

    def __init__(self, base_transforms):
        self.base_transforms = base_transforms

    def __call__(self, x):
        return self.base_transforms[random.randint(0,len(self.base_transforms)-1)](x)

def get_aug(name = 'base',img_size = 224):
    # 1 常规数据增强方式
    if name == 'base':
        return transform.Compose([transform.Resize([img_size,img_size]),
                                  transform.ToTensor(),
                                  transform.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    # 2 Random Erasing 数据增强
    elif name == 'RE':
        return transform.Compose([transform.Resize([img_size,img_size]),
                                  transform.RandomHorizontalFlip(),
                                  transform.ToTensor(),
                                  transform.RandomErasing(p=0.8, scale=(0.02, 0.20), ratio=(0.5, 2.0),inplace=True),
                                  transform.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                  ])
    # 3 Random Crop 数据增强
    elif name == "RandCrop":
        return transform.Compose([transform.RandomResizedCrop(img_size, scale=(1 / 1.3, 1.0), ratio=(0.9, 1.1)),
                                  transform.RandomHorizontalFlip(),
                                  transform.ToTensor(),
                                  transform.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    # 4 Alb Augmented 数据增强
    elif name == 'DFDC_Selium':
        return ALBU_AUG(alb.Compose([
            alb.ImageCompression(quality_lower=60, quality_upper=100, p=0.5),
            alb.GaussNoise(p=0.1),
            alb.GaussianBlur(blur_limit=3, p=0.05),
            alb.HorizontalFlip(),
            alb.OneOf([
                alb.LongestMaxSize(max_size=img_size, interpolation=cv2.INTER_CUBIC),
                alb.LongestMaxSize(max_size=img_size, interpolation=cv2.INTER_AREA),
                alb.LongestMaxSize(max_size=img_size, interpolation=cv2.INTER_LINEAR)
            ], p=1.0),
            alb.PadIfNeeded(min_height=img_size, min_width=img_size, border_mode=cv2.BORDER_CONSTANT, value=0),
            alb.OneOf([alb.RandomBrightnessContrast(),alb.FancyPCA(), alb.HueSaturationValue()], p=0.7),
            alb.ToGray(p=0.2),
            alb.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=10,
                               border_mode=cv2.BORDER_CONSTANT, value=0, p=0.5),
            alb.Normalize(mean=tuple([0.485, 0.456, 0.406]), std=tuple([0.229, 0.224, 0.225])),
            ToTensorV2()
        ]))

    # 5 Random Augmented 数据增强
    elif name == 'RA':
        return OneOfTrans([
            transform.Compose([
                transform.Resize([img_size,img_size]),
                transform.RandomHorizontalFlip(),
                transform.ToTensor(),
                transform.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]),
            transform.Compose([
                transform.Resize([img_size,img_size]),
                transform.RandomHorizontalFlip(),
                transform.ToTensor(),
                transform.RandomErasing(p=1.0, scale=(0.02, 0.20), ratio=(0.5, 2.0)),
                transform.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]),
            transform.Compose([
                transform.RandomResizedCrop(img_size, scale=(1 / 1.3, 1.0), ratio=(0.9, 1.1)),
                transform.RandomHorizontalFlip(),
                transform.ToTensor(),
                transform.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])])

if __name__ == '__main__':
    aug_list = ['base', 'RE', 'DFDC_Selium', 'RA', 'RandCrop']
    trans_name = random.sample(aug_list, 1)[0]



    img = Image.open('./samples/gal.jpg')
    to_tensor = get_aug(name = 'RA', img_size= 224)
    print(to_tensor)
    img_t = to_tensor(img)
    print(img_t.shape)
