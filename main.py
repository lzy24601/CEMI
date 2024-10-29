import os.path
import sys

import numpy as np
from PIL import Image
import cv2
import shutil
from utils import *
import matplotlib.pyplot as plt


def prepro_video(video_root_path=None):
    video_list = getFileList(video_root_path)
    for video_path in video_list:
        getFrameFromVideo(video_path, frame_frequency=1)


def prepro_traindata(data_root_path=None, save_path=None):
    fold_list = os.listdir(data_root_path)
    count = 96
    lr_save_path = os.path.join(save_path, 'lr')
    hr_save_path = os.path.join(save_path, 'hr')
    makedirs(lr_save_path)
    makedirs(hr_save_path)
    for fold_name in fold_list:
        fold_path = os.path.join(data_root_path, fold_name)
        fold_name = os.path.basename(fold_path)
        lr_path = os.path.join(fold_path, fold_name + '-01.tif')
        hr_path = os.path.join(fold_path, fold_name + '-02.tif')
        lr = cv2.imread(lr_path)
        lr = cutImage(lr, x=[0, 1020], y=[0, 692])
        cv2.imwrite(os.path.join(lr_save_path, str(count) + '.png'), lr)
        hr = cv2.imread(hr_path)
        hr = cutImage(hr, x=[0, 1020], y=[0, 692])
        cv2.imwrite(os.path.join(hr_save_path, str(count) + '.png'), hr)
        count += 1


def move_img():
    img_fold_root = r'D:\code\python\EMHD\SR\SR\syn_22_39'
    img_sava_fold = r'D:\code\python\EMHD\SR\SR\syn_22_39_all'
    makedirs(img_sava_fold)
    fold_list = os.listdir(img_fold_root)
    img_cnt = 0
    for fold in fold_list:
        img_flod_path = os.path.join(img_fold_root, fold)
        img_names = os.listdir(img_flod_path)
        for img_name in img_names:
            img_path = os.path.join(img_flod_path, img_name)
            print(img_path)
            img_sava_path = os.path.join(img_sava_fold, '%04d.png' % img_cnt)
            print(img_path)
            print(img_sava_path)
            shutil.copyfile(img_path, img_sava_path)
            img_cnt += 1


def binary_compare_lr_hr():
    lr_root = r'D:\code\python\EMHD\own1\train\lr'
    hr_root = r'D:\code\python\EMHD\own1\train\hr'
    sr_root = r'C:\SR\Real-ESRGAN\results\lr'
    lr_bccs = []
    hr_bccs = []
    sr_bccs = []
    for i in range(1, 101):
        lr_path = os.path.join(lr_root, str(i) + '.png')
        hr_path = os.path.join(hr_root, str(i) + '.png')
        sr_path = os.path.join(sr_root, str(i) + '_out.png')
        print(lr_path, hr_path, sr_path)
        lr_img = cv2.imread(lr_path)
        hr_img = cv2.imread(hr_path)
        sr_img = cv2.imread(sr_path)
        lr_bcc, lr_thresh = binary(lr_img)
        hr_bcc, hr_thresh = binary(hr_img)
        sr_bcc, sr_thresh = binary(sr_img)
        lr_bccs.append(lr_bcc)
        hr_bccs.append(hr_bcc)
        sr_bccs.append(sr_bcc)
    x = [i for i in range(len(lr_bccs))]
    plt.title("7*7")
    plt.scatter(x, lr_bccs, label='lr', c='r', s=5)
    plt.scatter(x, sr_bccs, label='sr', c='b', s=5)
    plt.scatter(x, hr_bccs, label='hr', c='g', s=5)
    plt.legend()
    plt.show()
    for i in range(len(lr_bccs)):
        print(lr_bccs[i], hr_bccs[i], sr_bccs[i])
    # print("avg lr:{} hr:{} sr:{}".format(sum(lr_bccs) / len(lr_bccs), sum(hr_bccs) / len(hr_bccs), sum(sr_bccs) / len(sr_bccs)))
    plt.scatter(x, [abs(hr_bccs[i] - sr_bccs[i]) for i in range(len(lr_bccs))], label='abs(sr - hr)', c='b', s=10)
    plt.legend()
    plt.show()
    print(lr_bccs)
    print(hr_bccs)
    print(sr_bccs)

    print("lr:{}".format((sum(lr_bccs) - sum(hr_bccs)) / len(hr_bccs)))
    print("sr:{}".format((sum(sr_bccs) - sum(hr_bccs)) / len(hr_bccs)))

def binary(image):
    mean = image.mean()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0, 0)
    # (_, thresh) = cv2.threshold(gray, mean - 1, 255, cv2.THRESH_BINARY)

    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

    # thresh = cv2.erode(thresh, None, iterations=1)
    # thresh = cv2.dilate(thresh, None, iterations=1)

    # 比较合适的参数
    # thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)


    # 显示预处理后图像.
    # print(thresh)
    # cv2.imshow("111", thresh)
    # plt.show()
    # cv2.waitKey()
    # thresh = cv2.resize(thresh,(448,448))
    result = np.sum(thresh == 0.)
    return result / (thresh.shape[0] * thresh.shape[1]), thresh


def main():
    # binary_compare_lr_hr()
    # move_img()
    prepro_video(video_root_path=r'D:\code\python\EMHD\SR\2024.0314\video')
    # prepro_traindata(data_root_path=r'D:\code\python\EMHD\own1\mi\mi', save_path=r'D:\code\python\EMHD\own1')
    # show_histogram(img_path=r'C:\SR\own12\val\0\hr\93.png')
    # getFrameFromVideo(r'D:\code\python\EMHD\SR\20230523-80%\20221105-1-01\bandicam 2023-05-23 15-23-10-656.mp4',
    #                   save_path=r'D:\code\python\EMHD\SR\20230523-80%', frame_frequency=50)


if __name__ == '__main__':
    main()
