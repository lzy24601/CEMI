import os
import shutil

# 根目录
root_dir = r'D:\code\python\EMHD\SR\SR\tmp'
dst_dir = r'D:\code\python\EMHD\SR\SR\re'

# 初始图片编号
initial_nums = [1, 55, 110]

# 遍历根目录下的所有子文件夹
for i, sub_dir in enumerate(os.listdir(root_dir), start=1):
    # 计算当前子文件夹需要抽取的图片编号
    nums = [num + 5 * (i - 1) for num in initial_nums]

    # 遍历需要抽取的图片编号
    for num in nums:
        # 原图片路径
        src_path = os.path.join(root_dir, sub_dir, "{:03d}".format(num)+".png")
        # 目标图片路径
        dst_path = os.path.join(dst_dir, f'{i + 19}_{"{:03d}".format(num)}.png')
        print(src_path)
        print(dst_path)
        # 复制图片
        shutil.copy(src_path, dst_path)
