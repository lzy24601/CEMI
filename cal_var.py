import cv2
import  matplotlib as plt
hr = cv2.imread(r"D:\1\lr\2.png")
lr = cv2.imread(r"D:\1\gt\2.png")
sub = cv2.subtract(hr,lr);
width=sub.shape[1]
heigt= sub.shape[0]
mean = cv2.mean(sub)[0]
print(mean)
(mean , stddv) = cv2.meanStdDev(hr)
print(mean/255)
print(stddv/255)
