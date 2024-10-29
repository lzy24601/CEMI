import cv2
import numpy
import numpy as np
import datetime
import re
import os
from shutil import copy
import random
import skimage.data
import selectivesearch
from PIL import ImageFont
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
from pylab import *
import cv2

from Image_Stitching import imageStitch


def makedirs(fold):
    if not os.path.exists(fold):
        os.makedirs(fold)


def cutImage(img, x=[1, 1023], y=[2, 698]):
    """
    :param img: 待截取的图像
    :param x: 截取的X范围
    :param y: 截取的Y范围
    """
    cropped = img[y[0]:y[1], x[0]:x[1]]  # 裁剪坐标为[y0:y1, x0:x1]
    return cropped


def getFileList(file_dir, sort=False):
    file_list = []
    for files in os.listdir(file_dir):  # 不仅仅是文件，当前目录下的文件夹也会被认为遍历到
        file_path = os.path.join(file_dir, files)
        if os.path.isdir(file_path):
            getFileList(file_path, file_list)
        else:
            file_list.append(file_path)
    if sort:
        file_list.sort(key=lambda path: int(re.findall(r'\d+', path)[0]))

    return file_list


# https://blog.csdn.net/allway2/article/details/122628800?utm_medium=distribute.pc_relevant.none-task-blog-2~default~baidujs_baidulandingword~default-1-122628800-blog-122887195.pc_relevant_3mothn_strategy_and_data_recovery&spm=1001.2101.3001.4242.2&utm_relevant_index=4
def check_fault(img):
    into_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # changing the color format from BGr to HSV
    # This will be used to create the mask
    #蓝色
    # L_limit = np.array([50, 50, 50])  # setting the blue lower limit
    # U_limit = np.array([139, 255, 255])  # setting the blue upper limit
    #绿色
    L_limit = np.array([35, 50, 50])  # HSV中绿色的下限值
    U_limit = np.array([90, 255, 255])  # HSV中绿色的上限值
    b_mask = cv2.inRange(into_hsv, L_limit, U_limit)
    # creating the mask using inRange() function
    # this will produce an image where the color of the objects
    # falling in the range will turn white and rest will be black
    blue = cv2.bitwise_and(img, img, mask=b_mask)
    # cv2.imshow('Original', img)  # to display the original frame
    # cv2.imshow('Blue Detector', blue)  # to display the blue object output
    
    # cv2.waitKey()
    # print(blue)
    if (blue == 0).all():
        return True
    return False
    # this will give the color to mask.


def getFrameFromVideo(path, save_path=r'D:\code\python\EMHD\SR\2024.0314\extract_image', frame_frequency=300):
    """
    :param path: 视频路径
    :param save_path: 保存的目录，默认放在视频同目录下的文件夹，文件夹名称同视频名称
    :param frame_frequency: 每隔多少帧取一次
    """
    # 要提取视频的文件名，隐藏后缀
    sourceFileName = path
    video_path = os.path.join("", "", sourceFileName)
    times = 0
    if save_path == '':
        outPutDirName = os.path.join('', sourceFileName + '\\')
    else:
        outPutDirName = os.path.join(save_path, sourceFileName.split('\\')[-1][:-4])
    if not os.path.exists(outPutDirName):
        # 如果文件目录不存在则创建目录
        os.makedirs(outPutDirName)
    camera = cv2.VideoCapture(video_path)
    flag = 0
    while True:
        res, image = camera.read()
        if not res:
            save_path = os.path.join(outPutDirName, '%06d' % times + '.png')
            img = cutImage(tmp_img, x=[5, 1017], y=[2, 630])
            if check_fault(img):
                cv2.imwrite(save_path, img)
                print(save_path)
            print('not res , not image')
            break
        if times % frame_frequency == 0:
            save_path = os.path.join(outPutDirName, '%06d' % times + '.png')
            img = cutImage(image, x=[5, 1017], y=[2, 630])
            if check_fault(img):
                if flag % 30 == 0:
                    cv2.imwrite(save_path, img)
                    print(save_path)
                flag += 1
        times += 1
        tmp_img = image
    print('{}  提取结束'.format(path))
    camera.release()


def ImageDraw(img, text=''):
    font = ImageFont.truetype(r"C:\Windows\Fonts\simsun.ttc", 128)
    im = Image.open(img)
    draw = ImageDraw.Draw(im)
    draw.text((510, 200), text, (255, 0, 0), font=font)  # 设置文字位置/内容/颜色/字体
    draw = ImageDraw.Draw(im)  # Just draw it!
    return im


# 将文件夹内图片拼接为一张图
def imagesStitch(path):
    pass


def removeBlackBorder(read_file):
    image = cv2.imread(read_file, 1)  # 读取图片 image_name应该是变量
    img = cv2.medianBlur(image, 5)  # 中值滤波，去除黑色边际中可能含有的噪声干扰
    b = cv2.threshold(img, 15, 255, cv2.THRESH_BINARY)  # 调整裁剪效果
    binary_image = b[1]  # 二值图--具有三通道
    binary_image = cv2.cvtColor(binary_image, cv2.COLOR_BGR2GRAY)
    print(binary_image.shape)  # 改为单通道

    indexes = np.where(binary_image == 255)  # 提取白色像素点的坐标

    left = min(indexes[0])  # 左边界
    right = max(indexes[0])  # 右边界
    width = right - left  # 宽度
    bottom = min(indexes[1])  # 底部
    top = max(indexes[1])  # 顶部
    height = top - bottom  # 高度

    pre1_picture = image[left:left + width, bottom:bottom + height]  # 图片截取
    return pre1_picture  # 返回图片数据


def testremoveBlackBorder():
    source_file = "./images_cut/merger/merger300_cut.jpg"  # 原始图片
    save_path = "./images_cut/merger/final_remove_merger300_cut.jpg"  # 裁剪后图片

    starttime = datetime.datetime.now()
    x = removeBlackBorder(source_file)
    cv2.imwrite(save_path, x)
    print("裁剪完毕")
    endtime = datetime.datetime.now()  # 记录结束时间
    endtime = (endtime - starttime).seconds
    print("裁剪总用时", endtime)


def split_data():
    # split_data.py
    # 划分数据集flower_data，数据集划分到flower_datas中，训练验证比例为8：2

    # 获取data文件夹下所有文件夹名（即需要分类的类名）
    # 划分数据集flower_data，数据集划分到flower_datas中
    file_path = r'D:\eudt100_for_classfication\val'
    new_file_path = r'D:\eudt100_for_classfications\val'

    # 划分比例，训练集 : 验证集 = 8 : 2
    split_rate = 0.2

    data_class = [cla for cla in os.listdir(file_path)]

    train_path = new_file_path + '/test/'
    val_path = new_file_path + '/val/'
    # 创建 训练集train 文件夹，并由类名在其目录下创建子目录
    makedirs(new_file_path)
    for cla in data_class:
        makedirs(train_path + cla)

    # 创建 验证集val 文件夹，并由类名在其目录下创建子目录
    makedirs(new_file_path)
    for cla in data_class:
        makedirs(val_path + cla)

    # 遍历所有类别的全部图像并按比例分成训练集和验证集
    for cla in data_class:
        cla_path = file_path + '/' + cla + '/'  # 某一类别的子目录
        images = os.listdir(cla_path)  # iamges 列表存储了该目录下所有图像的名称
        num = len(images)
        # eval_index = random.sample(images, k=int(num * split_rate))  # 从images列表中随机抽取 k 个图像名称
        eval_index = random.sample(images, 20)  # 从images列表中随机抽取 k 个图像名称
        for index, image in enumerate(images):
            # eval_index 中保存验证集val的图像名称
            if image in eval_index:
                image_path = cla_path + image
                new_path = val_path + cla
                copy(image_path, new_path)  # 将选中的图像复制到新路径

            # 其余的图像保存在训练集train中
            else:
                image_path = cla_path + image
                new_path = train_path + cla
                # copy(image_path, new_path)
            print("\r[{}] processing [{}/{}]".format(cla, index + 1, num), end="")  # processing bar
        print()

    print("processing done!")


# 这种的只能处理 一个样本数据的 图片名字， 有待改进
def rename(path):
    filelist = os.listdir(path)  # 获取指定的文件夹包含的文件或文件夹的名字的列表
    print(filelist)
    total_num = len(filelist)  # 获取文件夹内所有文件个数

    i = 0  # 图片名字从 0 开始
    c = 0
    for item in filelist:  # 遍历这个文件夹下的文件,即 图片
        if item.endswith('.jpg'):
            src = os.path.join(os.path.abspath(path), item)
            dst = os.path.join(os.path.abspath(path), str(i) + '.jpg')

            try:
                os.rename(src, dst)
                print('converting %s to %s ...' % (src, dst))
                i = i + 1
                c = c + 1
            except:
                continue
    print('total %d to rename & converted %d jpgs' % (total_num, i))
    print('total %d to rename & converted %d jpgs' % (total_num, c))


def get_bounding_box(file):
    # loading astronaut image
    # img = skimage.data.astronaut()

    img = Image.open(file)
    img = img.convert("RGB")
    # dtype = {'F': np.float32, 'L': np.uint8}[img.mode]
    w, h = img.size
    img = np.array(img.getdata())
    print(img.size)
    img.shape = (h, w, img.size // (w * h))
    print(img.shape)

    # perform selective search
    img_lbl, regions = selectivesearch.selective_search(
        img, scale=500, sigma=0.9, min_size=10)
    print(regions)
    candidates = set()
    for r in regions:
        # excluding same rectangle (with different segments)
        if r['rect'] in candidates:
            continue
        # excluding regions smaller than 2000 pixels
        if r['size'] < 2000:
            continue
        # distorted rects
        x, y, w, h = r['rect']
        if w / h > 1.2 or h / w > 1.2:
            continue
        candidates.add(r['rect'])
    # draw rectangles on the original image
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
    ax.imshow(img)
    for x, y, w, h in candidates:
        print(x, y, w, h)
        rect = mpatches.Rectangle(
            (x, y), w, h, fill=False, edgecolor='red', linewidth=1)
        ax.add_patch(rect)

    plt.show()


def show_histogram(img_path=None):
    # 读取图像到数组中，并灰度化
    im = array(Image.open(img_path).convert('L'))
    # 直方图图像
    # flatten可将二维数组转化为一维
    hist(im.flatten(), 128)
    # 显示
    show()
