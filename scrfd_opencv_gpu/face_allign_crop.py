# -*- coding: utf-8 -*-
# @Time : 2023/6/14 11:54
# @Author : Liangyu Gong
# @FileName: face_allign_crop.py
# @Software: PyCharm

from scrfd_face_detect import SCRFD
import cv2
import matplotlib.pyplot as plt
import math


def predict_landmarks(net, img):
    res = net.detect(img)
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
    # 外扩0.1宽高
    h = int(0.05 * (ymax - ymin))
    w = int(0.05 * (xmax - xmin))
    crop_img = img[int(ymin) - h:int(ymax) + h, int(xmin) - w:int(xmax) + w]
    return crop_img


if __name__ == '__main__':

    mynet = SCRFD('weights/scrfd_10g_kps.onnx', confThreshold=0.5, nmsThreshold=0.5)
    img = cv2.imread("test_imgs/test3.jpg")
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    landmarks, output_box = predict_landmarks(mynet, img)
    print('Five_points_landmarks are===>', landmarks)
    rotated_img, angle = face_allign(landmarks, img)
    crop_img = face_crop(rotated_img, output_box)
    crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
    cv2.imshow('Crop image',crop_img)
    cv2.waitKey(0)
    # imgs = [img, rotated_img, crop_img]
    # names = ['Original', 'Rotated Img', 'Crop img']
    # for i in range(3):
    #     plt.subplot(1, 3, i + 1)
    #     plt.imshow(imgs[i])
    #     plt.axis('off')
    #     plt.title(names[i])