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

matplotlib.rc("font", family='YouYuan')
np.seterr(all='ignore')

# (10, 121) (19,243) (38,486)

def crop_img(org_img, savepath, size=(19, 243)):
    row, column = size
    height, width = org_img.shape[:2]
    print('height %d widht %d' % (height, width))

    row_step = (int)(height / row)
    column_step = (int)(width / column)

    print('row step %d col step %d' % (row_step, column_step))

    print('height %d widht %d' % (row_step * row, column_step * column))

    img = org_img[0:row_step * row, 0:column_step * column]
    utils.makedirs(savepath)
    for i in range(row):
        for j in range(column):
            pic_name = savepath + '\\' + str(i) + "_" + str(j) + ".png"
            row_start = i * row_step
            row_end = (i * row_step + row_step) if i < row - 1 else height
            col_start = j * column_step
            col_end = (j * column_step + column_step) if j < column - 1 else width
            # 使用org_img会使得分割的小图大小不一致，但不会漏裁。img分割的小图大小一致
            tmp_img = org_img[row_start:row_end, col_start:col_end]
            cv2.imwrite(pic_name, tmp_img)


def get_specific_pixel_area(img, color=(0, 255, 170)):
    height, width = img.shape[:2]
    cnt = 0
    for i in range(height):
        for j in range(width):
            if (color == img[i, j]).all():
                cnt += 1
    return cnt


def cnt_batten(img):
    mean = img.mean()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('gr', gray)
    gray = cv2.GaussianBlur(gray, (5, 5), 0, 0)
    # gray = cv2.blur(gray, (1, 1))
    (_, thresh) = cv2.threshold(gray, mean, 255, cv2.THRESH_BINARY)
    result = np.sum(thresh == 0.)
    return result / (thresh.shape[0] * thresh.shape[1])


def plot_mat1(mat):
    img = plt.imread(r"D:\code\python\EMHD\SR\all2.PNG")
    fig, ax = plt.subplots()
    ax.imshow(img, extent=[-0.5, 29.5, -0.5, 5.5])
    # ax.matshow(mat, cmap=plt.cm.Reds)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            plt.text(x=j - 0.5, y=i - 0.1, s='{:>.2f}%'.format(mat[mat.shape[0] - i - 1, j] * 100), fontsize=8)

    plt.show()


def plot_mat(mat):
    fig, ax = plt.subplots()
    ax.matshow(mat, cmap=plt.cm.Reds)
    # for i in range(mat.shape[0]):
    #     for j in range(mat.shape[1]):
    #         plt.text(x=j-0.4, y=i+0.1, s='{:>.2f}%'.format(mat[i, j]*100), fontsize=8)
    plt.show()


def plot_batten():  # row step 206 col step 254
    row = 10
    col = 121
    cnt = 0.0
    img_fold = r'D:\code\python\EMHD\SR\save_mask_crop'
    mat = np.zeros((row, col), dtype='float32')
    imgs = os.listdir(img_fold)
    for img in imgs:
        img_path = os.path.join(img_fold, img)
        i, j = [int(i) for i in img.split('.')[0].split('_')]
        print(i, j)
        im = cv2.imread(img_path)
        mat[i][j] = cnt_batten(im)
        print(mat[i][j])
        cnt += mat[i][j]
    plot_mat(mat)
    _sum = np.sum(mat, axis=0) / 19
    plt.plot(_sum)

    # fig, ax1 = plt.subplots()
    # plt.xticks(rotation=45)

    # ax1.plot(xx, _sum, label = "板条状占比")
    # ax1.scatter([62,75,86,112,125,137,149,162,174], [0.263,0.285,0.2743,0.294,0.2328,0.2325,0.2353,0.2650,0.2462], color = 'r', marker="x", label = "实验点板条状占比")
    # ax1.set_xlabel("长度(自内向外)")
    # ax1.set_ylabel("板条状占比")

    # ax2 = ax1.twinx()
    # ax2.scatter([62,75,86,112,125,137,149,162,174],[5.53, 5.558, 5.553, 5.399, 5.431, 5.354, 5.29, 5.332, 5.22], color = 'g', marker="o", label = "实验点硬度")
    # ax2.set_ylabel("")

    # fig.legend(loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)
    # plt.show()

    # plt.plot(xx, _sum)
    # plt.xlabel('长度(自内向外)') #x轴坐标名称及字体样式
    # plt.ylabel('板条状占比') #y轴坐标名称及字体样式
    # plt.scatter([62,75,86,112,125,137,149,162,174], [0.263,0.285,0.2743,0.294,0.2328,0.2325,0.2353,0.2650,0.2462], color = 'r', marker="x", label = "板条状占比")
    # plt.scatter([62,75,86,112,125,137,149,162,174],[5.53, 5.558, 5.553, 5.399, 5.431, 5.354, 5.29, 5.332, 5.22], color = 'g', marker="o", label = "硬度")
    # plt.legend()
    # plt.show()

    np.save('batten.npy', mat)
    print(cnt / len(imgs))


def binary(image):
    mean = image.mean()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0, 0)
    # (_, thresh) = cv2.threshold(gray, mean - 1, 255, cv2.THRESH_BINARY)

    # thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

    # thresh = cv2.erode(thresh, None, iterations=1)
    # thresh = cv2.dilate(thresh, None, iterations=1)

    # 比较合适的参数
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)


    # 显示预处理后图像.
    # print(thresh)
    cv2.imshow("111", thresh)
    plt.show()
    cv2.waitKey()
    # thresh = cv2.resize(thresh,(448,448))
    result = np.sum(thresh == 0.)
    return result / (thresh.shape[0] * thresh.shape[1]), thresh


def plot_binary():
    row = 19
    col = 243
    ans = 0.0
    img_fold = r'D:\code\python\EMHD\SR\all'
    binary_root = r'D:\code\python\EMHD\SR\binary'
    utils.makedirs(binary_root)
    mat = np.zeros((row, col), dtype='float32')
    imgs = os.listdir(img_fold)
    for img in imgs:
        img_path = os.path.join(img_fold, img)
        i, j = [int(i) for i in img.split('.')[0].split('_')]
        im = cv2.imread(img_path)
        mat[i][j], tmp = binary(im)
        if mat[i][j] == 0:
            print(i, j)
        tmp = resize_img_keep_ratio(tmp, (384, 480))
        cv2.imwrite(os.path.join(binary_root, img), tmp)
        # re, _ = binary(im)
    # print(ans / row / col)
    _sum = np.sum(mat, axis=0) / 19
    # print(_sum)
    plt.scatter([i for i in range(len(_sum))], _sum)
    plt.show()
    np.save('binary.npy', mat)
    print(mat)
    plot_mat(mat)


def binary_sync(image):
    m, n, _ = image.shape
    # print(m, n, image[0][0])
    white_size = 0
    for i in range(m):
        for j in range(n):
            if (image[i][j] == [0, 255, 0]).all():
                white_size += 1
                image[i][j] = [255, 255, 255]
    mean = image.mean()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # gray = cv2.GaussianBlur(gray, (3, 3), 0, 0)
    # print(mean)
    # (_, thresh) = cv2.threshold(gray, mean - 2.5, 255, cv2.THRESH_BINARY)

    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

    # thresh = cv2.erode(thresh, None, iterations=1)
    # thresh = cv2.dilate(thresh, None, iterations=1)

    # 显示预处理后图像.
    # cv2.imshow('Pretreatment', thresh)

    # thresh = cv2.resize(thresh,(448,448))
    result = np.sum(thresh == 0.)
    # if result != 0:
    #     print(result / ((m * n) - white_size))
    #     plt.imshow(thresh)
    #     plt.show()
    #     cv2.waitKey()
    if white_size > ((m * n) / 10 * 9):
        return 0, thresh
    return result / ((m * n) - white_size), thresh


def plot_binary_sync():
    row = 19
    col = 243
    ans = 0.0
    img_fold = r'D:\code\python\EMHD\SR\all_sync'
    binary_root = r'D:\code\python\EMHD\SR\binary'
    utils.makedirs(binary_root)
    mat = [[] for i in range(col)]
    imgs = os.listdir(img_fold)
    for img in imgs:
        img_path = os.path.join(img_fold, img)
        i, j = [int(i) for i in img.split('.')[0].split('_')]
        # print(img_path)
        im = cv2.imread(img_path)
        re, tmp = binary(im)
        if re != 0:
            mat[j].append(re)
    np.save('binary_sync.npy', mat)
    print(mat)
    # plot_mat(mat)


def remove_fill(img, color=(0, 255, 170)):
    height, width = img.shape[:2]
    for i in range(height):
        for j in range(width):
            if (img[i, j] == color).all():
                img[i, j] = (255, 255, 255)
    return img


def remove_fills():
    root_path = r'D:\code\python\EMHD\SR\all'
    save_path = r'D:\code\python\EMHD\SR\all_no_fill'
    utils.makedirs(save_path)
    imgs = os.listdir(root_path)
    for name in imgs:
        img_path = os.path.join(root_path, name)
        img = cv2.imread(img_path)
        tmp = remove_fill(img)
        save = os.path.join(save_path, name)
        cv2.imwrite(save, tmp)

def cal_binary():

    img_fold = r'D:\core\img'
    imgs = os.listdir(img_fold)
    binary_arr = []
    for img in imgs:
        
        img_path = os.path.join(img_fold, img)
        idx = int(img.split('_')[1].split('.')[0])
        im = cv2.imread(img_path)
        print(img)
        re, _ = binary(im)
        binary_arr.append([idx, re])
        print("img_name:{} bcc_percent:{}".format(img, re))
    #np.save('binary_arr.npy', binary_arr)

def main():
    # org_img = cv2.imread(r'D:\code\python\EMHD\SR\stitch_result\syn_22-39\sync\crop.tif')
    # print(type(org_img))
    # crop_img(org_img, savepath='./all_sync')
    # plot_binary_sync()
    # plot_binary()
    # plot_batten()
    # print(cnt_batten(cv.imread(r'D:\code\python\EMHD\SR\save_mask_crop\0_9.png')))
    # val, _ = binary(cv2.imread(r'D:\code\python\EMHD\SR\all\0_2.png'))
    # print(val)
    # remove_fills()

    # im = cv2.imread(r'D:\code\python\EMHD\SR\all\0_144.png')
    # re, tmp = binary_sync(im)
    cal_binary()

if __name__ == '__main__':
    main()
