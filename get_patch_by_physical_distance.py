import sys
import os

os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2, 40).__str__()
import cv2
import numpy as np
import utils
import matplotlib.pyplot as plt
from PIL import Image
from jpg2png import resize_img_keep_ratio
import matplotlib
from crop_img_MN import binary, cnt_batten, plot_mat

matplotlib.rc("font", family='YouYuan')


def crop_img(org_img, savepath):
    height, width = org_img.shape[:2]
    patch_size = [480, 384]
    column = int(org_img.shape[0] / patch_size[1])
    absolut_x = [399.0, 698.99, 1106.279, 1406.269, 2084.115, 2384.105, 2853.402, 3153.392, 3615.354, 3915.344, 4533.66, 4833.649, 5463.776, 5763.766, 6387.891, 6687.881, 7223.139, 7523.128, 7995.711, 8295.701, 8697.751, 8997.741, 9511.186, 9811.176, 10293.856, 10593.846, 11692.364, 12297.678, 12597.667, 13305.851, 13605.841, 14124.525, 14424.515, 15369.919, 16009.284, 16309.274, 16891.918,16937.268, 17237.258, 17919.153, 18219.143, 18621.526, 18921.516, 19620.651, 19920.641, 20760.127, 21060.117, 22005.044, 22305.034, 22645.172, 22945.162, 23678.015, 23978.005, 24309.237, 24609.227, 24958.366, 25258.356, 25625.592, 25925.582, 26216.333, 26516.322, 26950.567, 27250.557]

    k_x = 0.238164
    row_step = patch_size[1]
    column_step = int(patch_size[0] / 2)
    utils.makedirs(savepath)
    for i, x in enumerate(absolut_x):
        x_centre = int(x / k_x)
        for j in range(column):
            pic_name = savepath + '\\' + str(j) + "_" + str(i) + ".png"
            col_start = int(x_centre - column_step)
            col_end = int(x_centre + column_step if x_centre + column_step < width else width - 1)
            row_start = j * row_step
            row_end = (j * row_step + row_step)
            print(row_start, row_end, col_start, col_end)
            # 使用org_img会使得分割的小图大小不一致，但不会漏裁。img分割的小图大小一致
            tmp_img = org_img[row_start:row_end, col_start:col_end]
            cv2.imwrite(pic_name, tmp_img)


def plot_binary():
    row = 20
    col = 67
    ans = 0.0
    img_fold = r'D:\code\python\EMHD\SR\patch_by_physical_distance'
    binary_root = r'D:\code\python\EMHD\SR\physical_distance_binary'
    utils.makedirs(binary_root)
    mat = np.zeros((row, col), dtype='float32')
    imgs = os.listdir(img_fold)
    for img in imgs:
        img_path = os.path.join(img_fold, img)
        i, j = [int(i) for i in img.split('.')[0].split('_')]
        print(i, j, img)
        im = cv2.imread(img_path)
        mat[i][j], tmp = binary(im)
        # tmp = resize_img_keep_ratio(tmp, (384, 480))
        cv2.imwrite(os.path.join(binary_root, img), tmp)
        re, _ = binary(im)
        ans += re
    # print(ans / row / col)
    _sum = np.sum(mat, axis=0)
    _sum = _sum / 20
    plt.scatter([i for i in range(len(_sum))], _sum)
    plt.show()
    np.save('binary_physical.npy', mat)
    plot_mat(mat)


def main():
    # org_img = cv2.imread(r'4k_crop.tif')
    # crop_img(org_img, "./patch_by_physical_distance_2")
    plot_binary()


if __name__ == '__main__':
    main()

