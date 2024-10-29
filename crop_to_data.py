import os
import cv2
import numpy as np
import json


'''
制作一个只包含分类标注的标签图像，假如我们分类的标签为cat和dog，那么该标签图像中，Background为0，cat为1，dog为2。
我们首先要创建一个和原图大小一致的空白图像，该图像所有像素都是0，这表示在该图像中所有的内容都是Background。
然后根据标签对应的区域使用与之对应的类别索引来填充该图像，也就是说，将cat对应的区域用1填充，dog对应的区域用2填充。
特别注意的是，一定要有Background这一项且一定要放在index为0的位置。
'''

# 图像分割 生成mask图片
# https://blog.csdn.net/oJiWuXuan/article/details/119569038

# 分类标签，一定要包含'Background'且必须放在最前面
category_types = ['Background', 'batten']
# 将图片标注json文件批量生成训练所需的标签图像png
img_root = r'C:\tmp\segmentation\data\sr\all'
json_root = r'C:\tmp\segmentation\data\sr\json\\'
mask_root = r'C:\tmp\segmentation\data\sr\mask\\'
imgpath_list = os.listdir(img_root)
for img_path in imgpath_list:
    img_name = img_path.split('.')[0]
    img = cv2.imread(os.path.join(img_root, img_path))
    h, w = img.shape[:2]
    # 创建一个大小和原图相同的空白图像
    mask = np.zeros([h, w, 1], np.uint8)

    with open(json_root + img_name+'.json', encoding='utf-8') as f:
        label = json.load(f)

    shapes = label['shapes']
    for shape in shapes:
        category = shape['label']
        points = shape['points']
        # 将图像标记填充至空白图像
        points_array = np.array(points, dtype=np.int32)
        print(category_types.index(category))
        mask = cv2.fillPoly(mask, [points_array], category_types.index(category))
        # if category == 'batten':
        #     # 调试时将某种标注的填充颜色改为255，便于查看用，实际时不需进行该操作
        #     mask = cv2.fillPoly(mask, [points_array], 125)
    # cv2.imshow('mask', mask)
    # cv2.waitKey(0)
    # 生成的标注图像必须为png格式
    cv2.imwrite(mask_root+img_name+'.png', mask)
