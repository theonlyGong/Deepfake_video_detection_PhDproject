# -*- coding: utf-8 -*-
# @Time : 2024/6/22 18:22
# @Author : Liangyu Gong
# @FileName: image_dataset.py
# @Software: PyCharm

# import os
# from torch.utils.data import Dataset

# data_dir = './data/val_imgs'
# for root, dirs, files in os.walk(data_dir):
#     for filename in files:
#         img_path = root+'\\'+filename
#         with open('img_test.txt','a') as f:
#             if img_path.split('\\')[-2] == 'real':
#                 f.write(img_path+' 0'+'\n')
#             if img_path.split('\\')[-2] == 'fake':
#                 f.write(img_path+' 1'+'\n')

from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch
from PIL import Image


class MyDataset(Dataset):
    def __init__(self, train_dir, transforms=None):
        # 初始化部分只需要初始化图片路径和transforms预处理即可
        self.data_info = self.get_info(train_dir)
        self.transforms = transforms

    def __len__(self):
        # len部分更为简单，只需要返回数据集的数据个数就好，通常都是len一个列表，看列表内元素个数
        return len(self.data_info)

    def __getitem__(self, item):
        # 这个函数主要是读取图片 + 预处理图片。
        img_path, label = self.data_info[item]
        label = torch.tensor(int(label))
        # print(img_path)
        img = Image.open(img_path).convert('RGB')
        if self.transforms is not None:
            image = self.transforms(img)
        return image, label

    @staticmethod
    def get_info(train_dir):
        # 处理txt文件的标准形式，将每一行的图像路径给提取出来
        data_info = []
        with open(train_dir) as file:
            lines = file.readlines()
            for line in lines:
                data_info.append(line.strip('\n').split(' '))
        return data_info

if __name__ == '__main__':
    train_transforms = transforms.Compose([transforms.ToTensor(),
                                           transforms.RandomHorizontalFlip(0.5)])
    train_dataset = MyDataset(train_dir='./img_test.txt', transforms = train_transforms)
    print(train_dataset.__getitem__(1))