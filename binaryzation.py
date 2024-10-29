import sys
import cv2
import matplotlib.pyplot as plt
import numpy as np
import utils
import os


def fill_color(img, color=0):
    print(img[:10, :10])
    height, width = img.shape[:2]
    print(height,width)
    for i in range(1, height-1):
        for j in range(1, width-1):
            if (color == img[i-1, j]) and (color == img[i, j-1]) \
                    and (color == img[i+1, j]) and (color == img[i, j+1]) and (img[i, j] == 255):
                print(img[i,j])
                img[i, j] = color
                print(img[i,j])
    return img


def binary(image):
    mean = image.mean()
    # image = cv2.bilateralFilter(image, 5, 160, 200)
    # image = cv2.pyrMeanShiftFiltering(image, 3, 10)
    # image = cv2.blur(image, (3,3))

    # image = cv2.filter2D(image, -1, kernel_sharpen_3)


    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3,3), 0, 0)
    # cv2.imshow('gr', gray)
    # print(mean)
    # (_, thresh) = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_TRIANGLE)
    (_, thresh) = cv2.threshold(gray, mean, 255, cv2.THRESH_BINARY)
    # thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 2)
    # thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 4)
    # 显示预处理后图像.
    # cv2.imshow('Pretreatment', thresh)
    # cv2.waitKey()
    thresh = cv2.resize(thresh,(480, 384))
    return thresh

if __name__ == "__main__":
    root_path = r'D:\code\python\EMHD\SR\all'
    save_path = r'D:\code\python\EMHD\SR\sr_binary'
    utils.makedirs(save_path)
    imgs = os.listdir(root_path)
    for name in imgs:
        img_path = os.path.join(root_path, name)
        img = cv2.imread(img_path)
        result = binary(img)
        cv2.imwrite(os.path.join(save_path, name), result)

