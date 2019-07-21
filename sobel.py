# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 18:33:48 2019

@author: CZW 边缘检测
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2


def sobel_suanzi(matImg):
    [r,c] = np.shape(matImg)   
    newImgX = np.zeros((r,c))
    newImgY = np.zeros((r,c))
    newImg = np.zeros((r,c))
    corematX = np.array([[-1, 0, 1],
                         [-2, 0, 2],
                         [-1, 0, 1]])
    
    corematY = np.array([[-1,-2,-1],
                         [ 0, 0, 0],
                         [ 1, 2, 1]])
    
    for i in range(r-2):
        for j in range(c-2):
            newImgX[i+1,j+1] = abs(np.sum(img[i:i+3 , j:j+3] * corematX))
            newImgY[i+1,j+1] = abs(np.sum(img[i:i+3 , j:j+3] * corematY))
            newImg[i+1,j+1] = (newImgX[i+1,j+1]*newImgX[i+1,j+1] + newImgY[i+1,j+1]*newImgY[i+1,j+1])**0.5
    
    return np.uint8(newImg)

#读取图片
img = Image.open(r"C:\Users\CZW\Pictures\Saved Pictures\timg1.jpg")
#img = cv2.imread(r"C:\Users\CZW\Pictures\Saved Pictures\timg1.jpg")
plt.figure("Image")
plt.imshow(img)
plt.axis('on')
plt.title('image')
plt.show()

#转化灰度图
img = img.convert('L')
plt.figure("Image")
plt.imshow(img,cmap='gray')  #不然是伪彩图
plt.axis('on')
plt.title('image')
plt.show()


#sobel算子
img = np.array(img)
out_sobel = sobel_suanzi(img)
plt.figure("Image")
plt.imshow(out_sobel,cmap='gray')
plt.axis('on')
plt.title('image')
plt.show()



