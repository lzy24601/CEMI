# 图像去噪平滑滤波
# 使用opencv的自带函数实现，与自编写作比较
# 产生椒盐噪声，高斯噪声等
# 使用中值滤波，平均滤波，高斯滤波，方框滤波

import numpy as np
import cv2
import matplotlib.pyplot as plt
import utils
import os


# 加噪声
def noise(img):
    out = img
    rows, cols, chn = img.shape
    for i in range(5000):
        x = np.random.randint(0, rows)
        y = np.random.randint(0, cols)
        out[x, y, :] = 255
    return out


def denoising(img_path, save_path):
    noise_img = cv2.imread(img_path)
    plt.subplot(3, 2, 1)
    plt.imshow(noise_img)
    plt.axis('off')
    plt.title('Original')

    # 均值滤波
    result1 = cv2.blur(noise_img, (5, 5))

    plt.subplot(3, 2, 3)
    plt.imshow(result1)
    plt.axis('off')
    plt.title('mean')

    # 方框滤波
    result2 = cv2.boxFilter(noise_img, -1, (5, 5), normalize=1)

    plt.subplot(3, 2, 4)
    plt.imshow(result2)
    plt.axis('off')
    plt.title('box')

    # 高斯滤波
    result3 = cv2.GaussianBlur(noise_img, (5, 5), 0)

    plt.subplot(3, 2, 5)
    plt.imshow(result3)
    plt.axis('off')
    plt.title('gaussian')

    # 中值滤波
    result4 = cv2.medianBlur(noise_img, 3)

    plt.subplot(3, 2, 6)
    plt.imshow(result4)
    plt.axis('off')
    plt.title('median')
    plt.show()

    img_name = os.path.basename(img_path).split('.')[0]
    save_fold = os.path.join(save_path, img_name)
    utils.makedirs(save_fold)
    print(save_fold)
    ori_path = os.path.join(save_fold, 'ori.png')
    blur_path = os.path.join(save_fold, 'blur.png')
    boxFilter_path = os.path.join(save_fold, 'boxFilter.png')
    GaussianBlurr_path = os.path.join(save_fold, 'GaussianBlur.png')
    medianBlur_path = os.path.join(save_fold, 'medianBlur.png')
    print(ori_path)
    cv2.imwrite(ori_path, noise_img)
    cv2.imwrite(blur_path, result1)
    cv2.imwrite(boxFilter_path, result2)
    cv2.imwrite(GaussianBlurr_path, result3)
    cv2.imwrite(medianBlur_path, result4)

if __name__ == "__main__":
    root_path = r'D:\code\python\EMHD\own1\val\lr'
    save_path = r'D:\code\python\EMHD\own1\val\denoise'
    utils.makedirs(save_path)
    imgs = os.listdir(root_path)
    for name in imgs:
        img_path = os.path.join(root_path, name)
        denoising(img_path, save_path)


