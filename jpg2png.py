import cv2
import os



def resize_img_keep_ratio(img, target_size):
    # img = cv2.imread(img_name) # 读取图片
    old_size = img.shape[0:2]  # 原始图像大小
    ratio = min(float(target_size[i]) / (old_size[i]) for i in range(len(old_size)))  # 计算原始图像宽高与目标图像大小的比例，并取其中的较小值
    if ratio > 1:
        ratio = 1
    new_size = tuple([int(i * ratio) for i in old_size])  # 根据上边求得的比例计算在保持比例前提下得到的图像大小
    img = cv2.resize(img, (new_size[1], new_size[0]))  # 根据上边的大小进行放缩
    pad_w = target_size[1] - new_size[1]  # 计算需要填充的像素数目（图像的宽这一维度上）
    pad_h = target_size[0] - new_size[0]  # 计算需要填充的像素数目（图像的高这一维度上）
    top, bottom = pad_h // 2, pad_h - (pad_h // 2)
    left, right = pad_w // 2, pad_w - (pad_w // 2)
    img_new = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, None, (0, 0, 0))
    return img_new


def resize_img():
    root = r'C:\tmp\segmentation\data\data_path_percent\train'
    imgpath_list = os.listdir(root)
    for i in imgpath_list:
        print(os.path.join(root, i))
        pic_org = cv2.imread(os.path.join(root, i))
        pic_new = resize_img_keep_ratio(pic_org, (704, 1024))
        pic_new_name = root + r'\\' + i.split('.')[0] + '.png'
        print(pic_new_name)
        cv2.imwrite(pic_new_name, pic_new)
        # os.remove(os.path.join(root, i))


def main():
    resize_img()


if __name__ == '__main__':
    main()
