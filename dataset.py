# -*- coding: utf-8 -*-
# @Time : 2024/4/1 8:38
# @Author : Liangyu Gong
# @FileName: dataset.py
# @Software: PyCharm

import os
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
import math
from PIL import Image
from decord import VideoReader

from scrfd_opencv_gpu.scrfd_face_detect import SCRFD
from data_aug import *

# Load dataset dir
train_files = os.path.join(os.getcwd(), 'txt_files', 'new_train.txt')
val_files = os.path.join(os.getcwd(), 'txt_files', 'new_test.txt')
# test_files = os.getcwd() + '\\txt_files\\test.txt'

# Load face detector
detector = SCRFD('./scrfd_opencv_gpu/weights/scrfd_10g_kps.onnx', confThreshold=0.5, nmsThreshold=0.5)


# Facial detector SCRFD
def predict_landmarks(res, img):
    # res = net.detect(img)
    max_value = None
    max_idx = None

    for idx, (box, _) in enumerate(res):
        value = (box[2] - box[0]) * (box[3] - box[1])
        if (max_value is None or value > max_value):
            max_value = value
            max_idx = idx
    # 人脸框预测
    output_box = res[max_idx][0]
    # 5个关键点坐标预测
    kpss = res[max_idx][1]

    return kpss, output_box


def face_allign(landmarks, img):
    # 获取左右人眼中心点
    left_eye = landmarks[0]
    right_eye = landmarks[1]
    # 计算dy，dx从而获取角度
    dy = right_eye[1] - left_eye[1]
    dx = right_eye[0] - left_eye[0]
    angle = math.atan2(dy, dx) * 180. / math.pi
    # print('Rotating angle is:',angle)
    # 计算人眼中心点
    eye_center = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)
    rotate_matrix = cv2.getRotationMatrix2D(eye_center, angle, scale=1)
    rotated_img = cv2.warpAffine(img, rotate_matrix, (img.shape[1], img.shape[0]))
    rotated_img = cv2.cvtColor(rotated_img, cv2.COLOR_BGR2RGB)
    return rotated_img, angle


def face_crop(img, box):
    xmin, ymin, xmax, ymax = box
    # 外扩0.2宽高
    h = int(0.1 * (ymax - ymin))
    w = int(0.1 * (xmax - xmin))
    crop_img = img[int(ymin) - h:int(ymax) + h, int(xmin) - w:int(xmax) + w]
    # 如果外扩面积超出范围，则选择性不外扩，否则会出现scr.empty()的opencv报错！
    if crop_img.shape[1] <= 0 or crop_img.shape[0] <= 0:
        crop_img = img[int(ymin):int(ymax), int(xmin):int(xmax)]
    return crop_img


class Deepfake_Dataset(Dataset):

    # initialize required parameters
    def __init__(self, video_files, transform=None):
        self.video_dir = self.get_path_info(video_files)
        self.transform = transform

    # Return the Dataset amount
    def __len__(self):
        # print('Dataset amount is :{}'.format(len(self.video_dir)))
        return len(self.video_dir)

    # Return the separated frames with corresponding labels.
    def __getitem__(self, video_idx):
        # 读取视频路径
        video_name = self.video_dir[video_idx]
        # print('===>',video_name)
        # Define labels 0: Real; 1: Fake
        label = 0 if 'Real' in video_name else 1
        img_list = self.frame_divider(detector, video_name)
        if len(img_list) != 8:
            print('=====>',video_name)
        # print('This video is divided into:{} frames'.format(len(img_list)))
        trans_img_list = []
        label_list = []
        for img in img_list:
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            if self.transform is not None:
                img = self.transform(img)
            trans_img_list.append(img)
            label_list.append(label)

        img_tensor_list = torch.stack(trans_img_list)
        label_tensor_list = torch.tensor(label_list, dtype=torch.float32)
        return img_tensor_list, label_tensor_list

    @staticmethod
    def get_path_info(files):
        video_root_list = []
        with open(files, 'r') as f:
            lines = f.readlines()
            for line in lines:
                video_path = line.split(' ')[0]
                video_root_list.append(video_path)
        return video_root_list

    @staticmethod
    def frame_divider(detector, video_path):
        vr = VideoReader(video_path)
        total_frames = len(vr)
        # 这里特意引用向上取整，如果使用//8会得到9帧！
        frames = vr.get_batch(np.arange(0, total_frames, math.ceil(total_frames / 8))).asnumpy()

        imgs = []
        for idx in range(len(frames)):
            img = frames[idx]
            res = detector.detect(img)
            # 如果某一帧没有人脸,取上一索引视频帧
            if res == []:
                img = frames[idx - 1]
                res = detector.detect(img)
                landmarks, output_box = predict_landmarks(res, img)
                rotated_img, angle = face_allign(landmarks, img)
                crop_img = face_crop(rotated_img, output_box)
                if len(crop_img) == 0:
                    print(idx, crop_img)
                    imgs.append(rotated_img)
                else:
                    imgs.append(crop_img)
            else:
                # 需要加入人脸检测模块
                landmarks, output_box = predict_landmarks(res, img)
                rotated_img, angle = face_allign(landmarks, img)
                crop_img = face_crop(rotated_img, output_box)
                if len(crop_img) == 0:
                    print(idx, crop_img)
                    imgs.append(rotated_img)
                else:
                    imgs.append(crop_img)
        return imgs


if __name__ == '__main__':
    # Load Data Augmentation
    aug_list = ['base', 'RE', 'DFDC_Selium', 'RA', 'RandCrop']
    trans_name = random.sample(aug_list, 1)[0]
    transforms = get_aug(name=trans_name, img_size=224)

    # Define training set and validation set.
    transval_set = Deepfake_Dataset(val_files, transform=transforms)
    val_set = Deepfake_Dataset(val_files, transform=None)

    transtrain_set = Deepfake_Dataset(train_files, transform=transforms)

    # Load face detector
    detector = SCRFD('./scrfd_opencv_gpu/weights/scrfd_10g_kps.onnx', confThreshold=0.5, nmsThreshold=0.5)
    # 通过调用getitem内置函数来获取PIL格式图片和相应的标签
    # print(val_set.__getitem__(3300))

    from matplotlib import pyplot as plt

    # img_list = val_set.__getitem__(2000)[0]
    titles = ['Frame 1', 'Frame 2', 'Frame 3', 'Frame 4', 'Frame 5','Frame 6', 'Frame 7', 'Frame 8']
    # for i in range(8):
    #     plt.subplot(2,4,i+1)
    #     plt.title(titles[i])
    #     plt.axis('off')
    #     plt.imshow(img_list[i])
    # plt.show()
    # for i in range(3576):
    #     img_tensor, label_tensor = transval_set.__getitem__(i)
        # print(img_tensor.shape)          # 【8，3，224，224】
        # print(label_tensor)
        # print(type(img_tensor))
    import matplotlib.pyplot as plt
    for i in range(2500,2501):
        img_tensor, label_tensor = transval_set.__getitem__(i)
        for q in range(8):
            plt.subplot(2,4,q+1)
            plt.title(titles[q])
            plt.axis('off')
            plt.imshow(img_tensor[q].permute(1,2,0))
        plt.show()



    # transimg_list = []
    #
    # for t in trans_tensor_list:
    #     to_PIL = transform.ToPILImage(mode = 'RGB')
    #     img = to_PIL(t)
    #     transimg_list.append(img)
    #
    # titles = ['Frame 1', 'Frame 2', 'Frame 3', 'Frame 4', 'Frame 5','Frame 6', 'Frame 7', 'Frame 8']
    # for i in range(8):
    #     plt.subplot(2,4,i+1)
    #     plt.title(titles[i])
    #     plt.axis('off')
    #     plt.imshow(transimg_list[i])
    # plt.show()
