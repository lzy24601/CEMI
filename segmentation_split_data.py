import os
import random
import shutil

# 图像分割数据集划分
'''
├── data(按照7:2:1比例划分)
│   ├── train 存放用于训练的图片
│   ├── trainannot 存放用于训练的图片标注
│   ├── val 存放用于验证的图片
│   ├── valannot 存放用于验证的图片标注
│   ├── test 存放用于测试的图片
│   ├── testannot 存放用于测试的图片标注
'''

# 创建数据集文件夹
save_path_root = r'C:\tmp\segmentation\data'
dirpath_list = ['SR/train', 'SR/trainannot', 'SR/val', 'SR/valannot', 'SR/test', 'SR/testannot']
all_img_path = r'C:\tmp\segmentation\data\SR\all'
all_mask_path = r'C:\tmp\segmentation\data\SR\mask'

for dirpath in dirpath_list:
    dirpath = os.path.join(save_path_root,dirpath)
    if os.path.exists(dirpath):
        shutil.rmtree(dirpath)   # 删除原有的文件夹
        os.makedirs(dirpath)   # 创建文件夹
    elif not os.path.exists(dirpath):
        os.makedirs(dirpath)

# 训练集、验证集、测试集所占比例
train_percent = 0.7
val_percent = 0.1
test_percent = 0.2

# 数据集原始图片所存放的文件夹，必须为png文件
imagefilepath = 'data_mask/images'
total_img = os.listdir(all_img_path)
# 所有数据集的图片名列表
total_name_list = [row.split('.')[0] for row in total_img]
num = len(total_name_list)
num_list = range(num)
# 训练集、验证集、测试集所包含的图片数目
train_tol = int(num * train_percent)
val_tol = int(num * val_percent)
test_tol = int(num * test_percent)

# 训练集在total_name_list中的index
train_numlist = random.sample(num_list, train_tol)
# 验证集在total_name_list中的index
val_test_numlist = list(set(num_list) - set(train_numlist))
val_numlist = random.sample(val_test_numlist, val_tol)
# 测试集在total_name_list中的index
test_numlist = list(set(val_test_numlist) - set(val_numlist))

# 将数据集和标签图片安装分类情况依次复制到对应的文件夹
for i in train_numlist:
    img_path = os.path.join(all_img_path, total_name_list[i]+'.png')
    new_path = os.path.join(save_path_root, dirpath_list[0], total_name_list[i]+'.png')
    shutil.copy(img_path, new_path)
    img_path = os.path.join(all_mask_path, total_name_list[i]+'.png')
    new_path = os.path.join(save_path_root, dirpath_list[1], total_name_list[i]+'.png')
    shutil.copy(img_path, new_path)
for i in val_numlist:
    img_path = os.path.join(all_img_path, total_name_list[i]+'.png')
    new_path = os.path.join(save_path_root, dirpath_list[2], total_name_list[i]+'.png')
    shutil.copy(img_path, new_path)
    img_path = os.path.join(all_mask_path, total_name_list[i]+'.png')
    new_path = os.path.join(save_path_root, dirpath_list[3], total_name_list[i]+'.png')
    shutil.copy(img_path, new_path)
for i in test_numlist:
    img_path = os.path.join(all_img_path, total_name_list[i]+'.png')
    new_path = os.path.join(save_path_root, dirpath_list[4], total_name_list[i]+'.png')
    shutil.copy(img_path, new_path)
    img_path = os.path.join(all_mask_path, total_name_list[i]+'.png')
    new_path = os.path.join(save_path_root, dirpath_list[5], total_name_list[i]+'.png')
    shutil.copy(img_path, new_path)
