# -*- coding: utf-8 -*-


import numpy as np
import cv2
import utils
import os
from matplotlib import pyplot as plt

img_path = r"D:\code\python\EMHD\own1\val\lr\1.png"
def sharpening(img_path, save_path):
    # 加载图像
    image = cv2.imread(img_path)
    # 自定义卷积核
    kernel_sharpen_1 = np.array([
        [-1, -1, -1],
        [-1, 9, -1],
        [-1, -1, -1]])
    kernel_sharpen_2 = np.array([
        [1, 1, 1],
        [1, -7, 1],
        [1, 1, 1]])
    kernel_sharpen_3 = np.array([
        [-1, -1, -1, -1, -1],
        [-1, 2, 2, 2, -1],
        [-1, 2, 12, 2, -1],
        [-1, 2, 2, 2, -1],
        [-1, -1, -1, -1, -1]]) / 8.0
    # 卷积
    output_1 = cv2.filter2D(image, -1, kernel_sharpen_1)
    output_2 = cv2.filter2D(image, -1, kernel_sharpen_2)
    output_3 = cv2.filter2D(image, -1, kernel_sharpen_3)
    # 显示锐化效果
    # cv2.imshow('Original Image', image)
    # cv2.imshow('sharpen_1 Image', output_1)
    # cv2.imshow('sharpen_2 Image', output_2)
    # cv2.imshow('sharpen_3 Image', output_3)
    # 停顿
    if cv2.waitKey(0) & 0xFF == 27:
        cv2.destroyAllWindows()

    img_name = os.path.basename(img_path).split('.')[0]
    save_fold = os.path.join(save_path, img_name)
    utils.makedirs(save_fold)
    shap3_path = os.path.join(save_fold, 'shrp3.png')
    # cv2.imwrite(shap3_path, output_3)

if __name__ == "__main__":
    root_path = r'D:\val'
    save_path = r'D:\code\python\EMHD\own1\val\sharp'
    utils.makedirs(save_path)
    imgs = os.listdir(root_path)
    for name in imgs:
        img_path = os.path.join(root_path, name)
        sharpening(img_path, save_path)
