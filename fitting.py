'''
Author: CloudSir
Date: 2021-08-01 13:40:50
LastEditTime: 2021-08-02 09:41:54
LastEditors: CloudSir
Description: Python拟合多项式
https://github.com/cloudsir
'''
import matplotlib
matplotlib.rc("font",family='YouYuan')

import matplotlib.pyplot as plt
import numpy as np
 
x = [0.263, 0.285, 0.2743, 0.2940, 0.2328, 0.2325, 0.2353, 0.2650, 0.2462]
y = [5.53, 5.558, 5.553, 5.399, 5.431, 5.354, 5.29, 5.332, 5.22]
z1 = np.polyfit(x, y, 1) #用3次多项式拟合，输出系数从高到0
p1 = np.poly1d(z1) #使用次数合成多项式
y_pre = p1(x)
plt.xlabel('板条状占比') #x轴坐标名称及字体样式
plt.ylabel('硬度') #y轴坐标名称及字体样式
 
plt.plot(x,y,'.')
plt.plot(x,y_pre)
plt.show()
