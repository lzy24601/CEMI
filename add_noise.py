import os

import skimage
import numpy as np
import matplotlib.pyplot as plt
import cv2

def Gaussnoise_func(image, mean=0, var=0.003, scale=1):
    '''
    添加高斯噪声
    mean : 均值
    var : 方差
    '''
    image = np.array(image / 255, dtype=float)  # 将像素值归一
    noise = np.random.normal(mean, var ** 0.5, image.shape)  # 产生高斯噪声
    out = image + noise  # 直接将归一化的图片与噪声相加

    '''
    将值限制在(-1/0,1)间，然后乘255恢复
    '''
    if out.min() < 0:
        low_clip = -1.
    else:
        low_clip = 0.

    out = np.clip(out, low_clip, 1.0)
    out = np.uint8(out * 255)
    return out


def exponent_noise(img, a=10.0):
    a = 10.0
    noiseExponent = np.random.exponential(scale=a, size=img.shape)
    imgExponentNoise = img + noiseExponent
    imgExponentNoise = np.uint8(cv2.normalize(imgExponentNoise, None, 0, 255, cv2.NORM_MINMAX))  # 归一化为 [0,255]
    return imgExponentNoise


def gamma(img, a, b):
    noiseGamma = np.random.gamma(shape=b, scale=a, size=img.shape)
    imgGammaNoise = img + noiseGamma
    imgGammaNoise = np.uint8(cv2.normalize(imgGammaNoise, None, 0, 255, cv2.NORM_MINMAX))  # 归一化为 [0,255]
    return imgGammaNoise

def add_noise():
    origin = cv2.imread(r"D:\1\gt\7.png")
    # noisy = skimage.util.random_noise(origin, mode='gaussian', var=0.005)
    var = 0.0001
    for i in range(20, 40):
        save_path = os.path.join(r'D:\1\noise', str(i)+'.png')
        #origin1 = cv2.GaussianBlur(origin, (3, 3), 0)
        noisy = Gaussnoise_func(origin, mean=0, var=var*i,scale=i)
        #noisy = gamma(origin, a=i+1, b=2.5)
        cv2.imwrite(save_path, noisy)
        # skimage.io.imshow(noisy)

def cal_ssim():
    from skimage.metrics import structural_similarity as compare_ssim
    from skimage.metrics import peak_signal_noise_ratio as compare_psnr
    from skimage.metrics import mean_squared_error as compare_mse
    import cv2
    root_path = r'D:\1\noise'
    lr = cv2.imread(r'D:\1\lr\7.png')
    noise_imgs = os.listdir(root_path)
    for name in noise_imgs:
        img_path = os.path.join(root_path, name)
        noise_img = cv2.imread(img_path)
        psnr = compare_psnr(lr, noise_img)
        ssim = compare_ssim(lr, noise_img, multichannel=True)  # 对于多通道图像(RGB、HSV等)关键词multichannel要设置为True
        mse = compare_mse(lr, noise_img)
        print('{}  PSNR：{}，SSIM：{}，MSE：{}'.format(name, psnr, ssim, mse))


# cal_ssim()
add_noise()

