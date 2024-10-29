import os
import shutil

from utils import makedirs

def main():
    root_path = r'C:\SR\Real-ESRGAN\experiments\train_RealESRNetx1plus_own1_syndata_1000k_B6G1\visualization'
    save_path = r'C:\SR\cal\syn_net'
    num_one_img_fold = os.path.join(root_path, '1')
    num_one_imgs = os.listdir(num_one_img_fold)
    for img_name in num_one_imgs:
        print(img_name)
        index = img_name.split('.')[0].split('_')[1]
        save_fold = os.path.join(save_path, index)
        makedirs(save_fold)
        shutil.copyfile(os.path.join(num_one_img_fold, img_name), os.path.join(save_fold, '1'+'.png'))
        for i in range(2, 8):
            num_img_fold = os.path.join(root_path, str(i))
            shutil.copyfile(os.path.join(num_img_fold, str(i)+'_'+index+'.png'), os.path.join(save_fold, str(i) + '.png'))


# 将训练日志中保存的图像[1_1,1_2,1_3],[2_1,2_2,2_3],[3_1,3_2,3_3]按照[1_1,2_1,3_1],[1_2,2_2,3_2],[1_3,2_3,3_3]保存
if __name__ == "__main__":
    main()
